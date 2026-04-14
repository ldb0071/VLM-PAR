import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from vlmpar.registery import MODELS

class AttributeDecoupler(nn.Module):
    """
    Decouples attribute-specific features from general features
    without requiring separate encoders for each attribute.
    """
    def __init__(self, hidden_size, num_attributes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attributes = num_attributes
        
        # Attribute-specific projection matrices
        self.attribute_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_attributes)
        ])
        
        # Shared projection for category-invariant features
        self.category_projection = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanism to control feature mixing
        self.attribute_gates = nn.ModuleList([
            nn.Linear(hidden_size, 1)
            for _ in range(num_attributes)
        ])
        
    def forward(self, features, attribute_idx):
        """
        Args:
            features: Input features [batch, seq_len, hidden_size]
            attribute_idx: Index of the attribute to decouple
            
        Returns:
            Decoupled features [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = features.shape
        
        # Get category-invariant features
        category_features = self.category_projection(features)
        
        # Get attribute-specific features
        attribute_features = self.attribute_projections[attribute_idx](features)
        
        # Compute gating value (how much to focus on attribute vs. category)
        gate = torch.sigmoid(self.attribute_gates[attribute_idx](features))
        
        # Mix features based on gate value
        decoupled_features = gate * attribute_features + (1 - gate) * category_features
        
        return decoupled_features


class EnhancedAttention(nn.Module):
    """
    Enhanced attention mechanism for better feature extraction
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and residual connection
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input features [batch, seq_len, hidden_size]
            
        Returns:
            Enhanced features [batch, seq_len, hidden_size]
        """
        # Self-attention with residual connection
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attention(x, x, x)
        x = x + residual
        
        # Feed-forward with residual connection
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for aligning positive samples and pushing away negative samples
    """
    def __init__(self, temperature=0.07, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, anchor_features, positive_features, negative_features=None):
        """
        Compute contrastive loss with positive and negative samples
        
        Args:
            anchor_features: Anchor features [batch, hidden]
            positive_features: Positive features [batch, hidden]
            negative_features: Negative features [batch, hidden] or None
            
        Returns:
            Contrastive loss value
        """
        # Normalize features
        anchor_features = F.normalize(anchor_features, dim=-1)
        positive_features = F.normalize(positive_features, dim=-1)
        
        # Positive similarity
        pos_sim = torch.sum(anchor_features * positive_features, dim=-1)
        
        if negative_features is not None:
            # Normalize negative features
            negative_features = F.normalize(negative_features, dim=-1)
            
            # Negative similarity
            neg_sim = torch.sum(anchor_features * negative_features, dim=-1)
            
            # Triplet-style contrastive loss with margin
            loss = torch.clamp(neg_sim - pos_sim + self.margin, min=0.0)
            
            return loss.mean()
        else:
            # If no negative features, just maximize positive similarity
            return -pos_sim.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Compute focal loss
        Args:
            logits: Predicted logits [batch, num_classes]
            targets: Target labels [batch]
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


@MODELS.register('vlmpar_decoupled')
class VLMPARDecoupled(nn.Module):
    """
    VLMPAR model with decoupled attribute features without separate encoders
    Inspired by Control-CLIP architecture
    """
    # Default questions for person attribute recognition
    DEFAULT_QUESTIONS = {
        "upper_body_color": "What is the color of the person's upper body clothing?",
        "lower_body_color": "What is the color of the person's lower body clothing?",
        "gender": "What is the gender of the person?",
        "bag": "Is the person carrying a bag?",
        "hat": "Is the person wearing a hat?"
    }

    def __init__(self, 
                 model_name, 
                 num_classes_dict, 
                 questions=None,
                 neg_questions=None,
                 contrastive_weight=0.5,
                 temperature=0.07,
                 margin=0.2,
                 use_focal_loss=False,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 use_label_smoothing=False,
                 smoothing=0.1):
        super().__init__()
        
        # Initialize vision-language model
        self.vlm_base = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Store configuration
        self.num_classes_dict = num_classes_dict
        self.hidden_size = self.vlm_base.vision_model.config.hidden_size
        self.questions = questions if questions is not None else self.DEFAULT_QUESTIONS
        self.neg_questions = neg_questions
        self.contrastive_weight = contrastive_weight
        self.use_focal_loss = use_focal_loss
        self.use_label_smoothing = use_label_smoothing
        self.smoothing = smoothing
        
        # Validate questions
        for key in num_classes_dict.keys():
            if key not in self.questions:
                raise ValueError(f"Key '{key}' in num_classes_dict has no corresponding question. Please provide a question for this key.")
        
        # Initialize attribute decoupler
        self.attribute_decoupler = AttributeDecoupler(
            hidden_size=self.hidden_size,
            num_attributes=len(num_classes_dict)
        )
        
        # Initialize enhanced attention
        self.enhanced_attention = EnhancedAttention(
            hidden_size=self.hidden_size,
            num_heads=8,
            dropout=0.1
        )
        
        # Initialize contrastive loss
        self.contrastive_loss_fn = ContrastiveLoss(
            temperature=temperature,
            margin=margin
        )
        
        # Initialize focal loss if enabled
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # Initialize classifier heads
        self.classifier_heads = nn.ModuleDict({
            key: nn.Linear(self.hidden_size, num_labels)
            for key, num_labels in num_classes_dict.items()
        })
        
        # Attribute-specific feature extractors
        self.attribute_extractors = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            for key in num_classes_dict.keys()
        })
        
        # Create mapping from attribute name to index
        self.attribute_to_idx = {attr: idx for idx, attr in enumerate(num_classes_dict.keys())}

    def _process_image(self, pixel_values):
        """Process image input"""
        # Get vision embeddings
        vision_outputs = self.vlm_base.vision_model(pixel_values=pixel_values)
        image_seq = vision_outputs.last_hidden_state
        image_pooled = vision_outputs.pooler_output
        
        return image_seq, image_pooled
    
    def _process_text(self, question, device, attribute_name=None):
        """Process text input with optional attribute decoupling"""
        # Tokenize the question
        text_inputs = self.tokenizer(question, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # Get text embeddings
        text_outputs = self.vlm_base.text_model(**text_inputs)
        text_seq = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output
        
        # Apply attribute decoupling if attribute name is provided
        if attribute_name is not None:
            attribute_idx = self.attribute_to_idx[attribute_name]
            text_seq = self.attribute_decoupler(text_seq, attribute_idx)
        
        return text_seq, text_pooled
    
    def _get_negative_text(self, attribute_name, device):
        """Get negative text embeddings for an attribute"""
        if self.neg_questions is None or attribute_name not in self.neg_questions:
            return None, None
        
        neg_question = self.neg_questions[attribute_name]
        neg_text_seq, neg_text_pooled = self._process_text(neg_question, device)
        
        return neg_text_seq, neg_text_pooled
    
    def _extract_attribute_features(self, image_features, attribute_name):
        """Extract attribute-specific features from image features"""
        return self.attribute_extractors[attribute_name](image_features)
    
    def forward(self, pixel_values, question_type=None, return_dict=True, labels=None):
        """
        Forward pass
        Args:
            pixel_values: Input images
            question_type: Type of question to process ('all' or specific type)
            return_dict: Whether to return a dictionary (True) or just answers (False)
            labels: Ground truth labels for loss calculation
        Returns:
            Dictionary with answers, similarities, etc. for each question type
        """
        device = pixel_values.device
        batch_size = pixel_values.shape[0]
        
        # Process images
        image_seq, image_pooled = self._process_image(pixel_values)
        
        # Apply enhanced attention to image sequence
        enhanced_image_seq = self.enhanced_attention(image_seq)
        
        # Use the CLS token from enhanced sequence
        enhanced_image_features = enhanced_image_seq[:, 0, :]
        
        if question_type is not None and question_type != 'all':
            # Process a single attribute
            if question_type not in self.questions:
                raise ValueError(f"Invalid question type. Must be one of {list(self.questions.keys())}")
            
            return self._forward_single_attribute(
                enhanced_image_seq, 
                enhanced_image_features,
                question_type, 
                device, 
                return_dict,
                labels
            )
        else:
            # Process all attributes
            return self._forward_all_attributes(
                enhanced_image_seq, 
                enhanced_image_features,
                device, 
                return_dict,
                labels
            )
    
    def _forward_single_attribute(self, image_seq, image_features, attribute_name, device, return_dict, labels=None):
        """Process a single attribute"""
        # Get positive text embeddings with attribute decoupling
        pos_text_seq, pos_text_pooled = self._process_text(
            self.questions[attribute_name], 
            device, 
            attribute_name
        )
        
        # Get negative text embeddings if available
        neg_text_seq, neg_text_pooled = self._get_negative_text(attribute_name, device)
        
        # Extract attribute-specific image features
        attr_image_features = self._extract_attribute_features(image_features, attribute_name)
        
        # Calculate positive similarity
        pos_similarity = torch.cosine_similarity(attr_image_features, pos_text_pooled, dim=-1)
        
        # Calculate negative similarity if available
        if neg_text_pooled is not None:
            neg_similarity = torch.cosine_similarity(attr_image_features, neg_text_pooled, dim=-1)
            contrastive_score = pos_similarity - neg_similarity
        else:
            neg_similarity = torch.zeros_like(pos_similarity)
            contrastive_score = pos_similarity
        
        # Get classification logits
        logits = self.classifier_heads[attribute_name](attr_image_features)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        
        # Calculate loss if labels provided
        loss_dict = {}
        if labels is not None and attribute_name in labels:
            # Classification loss
            if self.use_focal_loss:
                cls_loss = self.focal_loss(logits, labels[attribute_name])
            elif self.use_label_smoothing:
                cls_loss = F.cross_entropy(logits, labels[attribute_name], label_smoothing=self.smoothing)
            else:
                cls_loss = F.cross_entropy(logits, labels[attribute_name])
            
            # Contrastive loss if negative text is available
            if neg_text_pooled is not None:
                cont_loss = self.contrastive_loss_fn(
                    attr_image_features,
                    pos_text_pooled,
                    neg_text_pooled
                )
            else:
                cont_loss = self.contrastive_loss_fn(
                    attr_image_features,
                    pos_text_pooled
                )
            
            # Total loss
            total_loss = cls_loss + self.contrastive_weight * cont_loss
            
            loss_dict = {
                "cls_loss": cls_loss,
                "contrastive_loss": cont_loss,
                "total_loss": total_loss
            }
        
        # Generate human-readable answers
        answers = self._generate_answers(attribute_name, pred)
        
        if return_dict:
            result = {
                "answer": answers,
                "pos_similarity": pos_similarity.tolist(),
                "neg_similarity": neg_similarity.tolist(),
                "contrastive_score": contrastive_score.tolist(),
                "confidence": torch.max(probs, dim=-1)[0].tolist(),
                "logits": logits,
                "probs": probs,
                "pred": pred
            }
            
            if loss_dict:
                result["losses"] = loss_dict
                
            return result
        else:
            return answers
    
    def _forward_all_attributes(self, image_seq, image_features, device, return_dict, labels=None):
        """Process all attributes"""
        answers = {}
        total_loss = 0.0
        loss_components = {}
        
        for key in self.questions.keys():
            # Process each attribute individually
            result = self._forward_single_attribute(
                image_seq,
                image_features,
                key,
                device,
                True,  # Always return dict for internal processing
                labels
            )
            
            # Store results
            answers[key] = result
            
            # Accumulate losses
            if labels is not None and "losses" in result:
                total_loss += result["losses"]["total_loss"]
                for loss_name, loss_value in result["losses"].items():
                    if loss_name not in loss_components:
                        loss_components[loss_name] = 0.0
                    loss_components[loss_name] += loss_value
        
        if return_dict:
            if labels is not None:
                answers["losses"] = loss_components
                answers["total_loss"] = total_loss
            return answers
        else:
            # Return concatenated text answers for all attributes
            return " ".join([v["answer"][0] for v in answers.values()])
    
    def _generate_answers(self, attribute_name, pred):
        """Generate human-readable answers for predictions"""
        batch_answers = []
        
        for p in pred:
            p_item = p.item()
            
            if attribute_name in ['upper_body_color', 'lower_body_color']:
                answer = f"The {attribute_name.replace('_', ' ')} is {self._get_color_name(p_item)}."
            elif attribute_name == 'gender':
                answer = f"The person is {'male' if p_item == 1 else 'female'}."
            else:
                num_classes = self.num_classes_dict[attribute_name]
                if num_classes == 2:
                    answer = f"{'Yes' if p_item == 1 else 'No'}, the person {'is' if p_item == 1 else 'is not'} {attribute_name.replace('_', ' ')}."
                else:
                    answer = f"The person's {attribute_name.replace('_', ' ')} is {self._get_class_name(attribute_name, p_item)}."
            
            batch_answers.append(answer)
        
        return batch_answers
    
    def _get_color_name(self, color_idx):
        """Map color index to name"""
        color_map = {
            0: "black",
            1: "white",
            2: "red",
            3: "blue",
            4: "green",
            5: "yellow",
            6: "brown",
            7: "purple",
            8: "gray",
            9: "orange",
            10: "pink"
        }
        return color_map.get(color_idx, "unknown")
    
    def _get_class_name(self, key, class_idx):
        """Map class index to name for multi-class attributes"""
        class_names = {
            'age': ['young', 'teenager', 'adult', 'old']
        }
        return class_names.get(key, [f"class_{i}" for i in range(self.num_classes_dict[key])])[class_idx]
    
    def compute_loss(self, outputs, labels):
        """
        Compute combined loss for training
        Args:
            outputs: Model outputs with logits and losses
            labels: Ground truth labels
        Returns:
            Total loss value
        """
        if "total_loss" in outputs:
            return outputs["total_loss"], outputs["losses"]
        
        # If losses not pre-computed, calculate them here
        total_loss = 0.0
        loss_components = {"cls_loss": 0.0, "contrastive_loss": 0.0}
        
        for key in self.questions.keys():
            if key in labels and key in outputs:
                # Classification loss
                attr_labels = labels[key]
                attr_logits = outputs[key]["logits"]
                
                if self.use_focal_loss:
                    cls_loss = self.focal_loss(attr_logits, attr_labels)
                elif self.use_label_smoothing:
                    cls_loss = F.cross_entropy(attr_logits, attr_labels, label_smoothing=self.smoothing)
                else:
                    cls_loss = F.cross_entropy(attr_logits, attr_labels)
                
                loss_components["cls_loss"] += cls_loss
                total_loss += cls_loss
        
        return total_loss, loss_components
    
    def predict(self, pixel_values, question_type=None):
        """
        Make predictions on input images
        Args:
            pixel_values: Input images
            question_type: Type of question to process
        Returns:
            Dictionary with class indices and class names
        """
        with torch.no_grad():
            outputs = self.forward(pixel_values, question_type=question_type, return_dict=True)
            
            classes_idx = {}
            classes = {}
            
            if question_type is not None and question_type != 'all':
                # Single attribute prediction
                pred = outputs["pred"].cpu().numpy()[0]
                classes_idx[question_type] = int(pred)
                
                if question_type in ['upper_body_color', 'lower_body_color']:
                    classes[question_type] = self._get_color_name(pred)
                elif question_type == 'gender':
                    classes[question_type] = 'male' if pred == 1 else 'female'
                else:
                    num_classes = self.num_classes_dict[question_type]
                    if num_classes == 2:
                        classes[question_type] = 'yes' if pred == 1 else 'no'
                    else:
                        classes[question_type] = self._get_class_name(question_type, pred)
            else:
                # All attributes prediction
                for key in self.questions.keys():
                    pred = outputs[key]["pred"].cpu().numpy()[0]
                    classes_idx[key] = int(pred)
                    
                    if key in ['upper_body_color', 'lower_body_color']:
                        classes[key] = self._get_color_name(pred)
                    elif key == 'gender':
                        classes[key] = 'male' if pred == 1 else 'female'
                    else:
                        num_classes = self.num_classes_dict[key]
                        if num_classes == 2:
                            classes[key] = 'yes' if pred == 1 else 'no'
                        else:
                            classes[key] = self._get_class_name(key, pred)
            
            return classes_idx, classes
    
    def load_model(self, model_path):
        """Load model weights from checkpoint"""
        weights = torch.load(model_path)["model_state_dict"]
        new_weights = {}
        for key, value in weights.items():
            new_key = key.replace("module.", "")
            new_weights[new_key] = value
        self.load_state_dict(new_weights) 