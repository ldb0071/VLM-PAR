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
        batch_size, seq_len, hidden_size = features.shape
        
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


@MODELS.register('vlmpar_cross_decoupled')
class VLMPARCrossDecoupled(nn.Module):
    """
    VLMPAR model that combines cross-attention with attribute decoupling
    for improved attribute-specific feature extraction
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
        
        # Initialize cross-attention components
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        # Add projection layers for better feature alignment
        self.image_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.text_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
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
            # Skip attribute decoupling if dimensions don't match
            # This allows the model to work with different embedding dimensions
            if text_seq.shape[-1] == self.hidden_size:
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
    
    def _apply_cross_attention(self, image_seq, text_seq):
        """Apply cross-attention between image and text features"""
        # Project features
        image_seq_proj = self.image_proj(image_seq)  # [batch, img_seq_len, hidden_size]
        
        # Handle different dimensions between image and text features
        if text_seq.shape[-1] != self.hidden_size:
            # Add a projection layer to match dimensions
            text_proj = nn.Linear(text_seq.shape[-1], self.hidden_size).to(image_seq.device)
            text_seq = text_proj(text_seq)
            
        text_seq_proj = self.text_proj(text_seq)     # [batch, txt_seq_len, hidden_size]
        
        # Ensure both have the same batch size
        if image_seq_proj.shape[0] != text_seq_proj.shape[0]:
            if text_seq_proj.shape[0] == 1:
                text_seq_proj = text_seq_proj.expand(image_seq_proj.shape[0], -1, -1)
            else:
                raise ValueError(f"Batch size mismatch: image_seq_proj {image_seq_proj.shape}, text_seq_proj {text_seq_proj.shape}")
        
        # Apply cross-attention
        attended_features, _ = self.cross_attention(
            query=image_seq_proj,  # [batch, img_seq_len, hidden_size]
            key=text_seq_proj,     # [batch, txt_seq_len, hidden_size]
            value=text_seq_proj    # [batch, txt_seq_len, hidden_size]
        )
        
        return attended_features
    
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
        
        # Extract attribute-specific features
        attribute_features = self._extract_attribute_features(image_features, attribute_name)
        
        # Apply cross-attention between image and text
        attended_features = self._apply_cross_attention(image_seq, pos_text_seq)
        
        # Pool attended features (use CLS token)
        attended_pooled = attended_features[:, 0, :]
        
        # Combine attended features with attribute-specific features
        combined_features = attended_pooled + attribute_features
        
        # Apply classifier head
        logits = self.classifier_heads[attribute_name](combined_features)
        
        # Get predictions
        pred = torch.argmax(logits, dim=-1)
        
        # Calculate contrastive loss if negative text is available
        contrastive_loss = 0.0
        if neg_text_pooled is not None:
            contrastive_loss = self.contrastive_loss_fn(
                attribute_features, 
                pos_text_pooled,
                neg_text_pooled
            )
        
        if not return_dict:
            return pred
        
        # Generate text answers
        answers = self._generate_answers(attribute_name, pred)
        
        result = {
            "logits": logits,
            "predictions": pred,
            "answers": answers,
        }
        
        if contrastive_loss > 0:
            result["contrastive_loss"] = contrastive_loss
        
        return result
    
    def _forward_all_attributes(self, image_seq, image_features, device, return_dict, labels=None):
        """Process all attributes"""
        results = {}
        
        # Process each attribute
        for attribute_name in self.questions.keys():
            result = self._forward_single_attribute(
                image_seq,
                image_features,
                attribute_name,
                device,
                return_dict,
                labels[attribute_name] if labels is not None else None
            )
            results[attribute_name] = result
        
        # Collect contrastive losses
        contrastive_losses = []
        for attr, result in results.items():
            if "contrastive_loss" in result:
                contrastive_losses.append(result["contrastive_loss"])
        
        # Add combined contrastive loss if available
        if contrastive_losses:
            avg_contrastive_loss = torch.mean(torch.stack(contrastive_losses))
            results["losses"] = {"contrastive_loss": avg_contrastive_loss * self.contrastive_weight}
        
        return results
    
    def _generate_answers(self, attribute_name, pred):
        """Generate text answers based on predictions"""
        batch_size = pred.shape[0]
        answers = []
        
        for i in range(batch_size):
            class_idx = pred[i].item()
            
            if attribute_name.startswith("up") and attribute_name != "up":
                # Color of upper body
                color_name = self._get_color_name(class_idx)
                answers.append(f"The person is wearing {color_name} upper body clothing.")
            elif attribute_name.startswith("down") and attribute_name != "down":
                # Color of lower body
                color_name = self._get_color_name(class_idx)
                answers.append(f"The person is wearing {color_name} lower body clothing.")
            else:
                # Other attributes
                class_name = self._get_class_name(attribute_name, class_idx)
                answers.append(class_name)
        
        return answers
    
    def _get_color_name(self, color_idx):
        """Get color name from index"""
        colors = {
            0: "not",
            1: "black",
            2: "white",
            3: "red",
            4: "purple",
            5: "yellow",
            6: "gray",
            7: "blue",
            8: "green",
            9: "brown",
            10: "pink"
        }
        return colors.get(color_idx, "unknown")
    
    def _get_class_name(self, key, class_idx):
        """Get class name from attribute key and class index"""
        if key == "gender":
            return "male" if class_idx == 0 else "female"
        elif key in ["bag", "hat", "backpack", "handbag"]:
            return f"The person is {'not ' if class_idx == 0 else ''}carrying a {key}."
        else:
            return f"Class {class_idx} for {key}"
    
    def compute_loss(self, outputs, labels):
        """Compute loss for all outputs"""
        total_loss = 0.0
        valid_samples = 0
        
        # Add contrastive loss if available
        if "losses" in outputs and "contrastive_loss" in outputs["losses"]:
            total_loss += outputs["losses"]["contrastive_loss"]
        
        # Compute classification loss for each attribute
        for attribute_name, output in outputs.items():
            if attribute_name == "losses":
                continue
                
            if "logits" not in output:
                continue
                
            logits = output["logits"]
            attribute_labels = labels[attribute_name]
            
            # Skip invalid labels
            valid_mask = attribute_labels != -1
            if not valid_mask.any():
                continue
                
            valid_logits = logits[valid_mask]
            valid_labels = attribute_labels[valid_mask]
            
            # Use appropriate loss function
            if self.use_focal_loss:
                loss = self.focal_loss(valid_logits, valid_labels)
            elif self.use_label_smoothing:
                loss = nn.CrossEntropyLoss(label_smoothing=self.smoothing)(valid_logits, valid_labels)
            else:
                loss = nn.CrossEntropyLoss()(valid_logits, valid_labels)
                
            total_loss += loss * valid_mask.sum()
            valid_samples += valid_mask.sum()
        
        if valid_samples > 0:
            total_loss = total_loss / valid_samples
            
        return total_loss
    
    def predict(self, pixel_values, question_type=None):
        """
        Make predictions for the given images
        Args:
            pixel_values: Input images
            question_type: Type of question to process ('all' or specific type)
        Returns:
            Dictionary with predictions and answers
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(pixel_values, question_type=question_type)
            
            if question_type is not None and question_type != 'all':
                # Single attribute
                return {
                    "predictions": outputs["predictions"],
                    "answers": outputs["answers"]
                }
            else:
                # All attributes
                results = {}
                for attr, output in outputs.items():
                    if attr == "losses":
                        continue
                        
                    results[attr] = {
                        "predictions": output["predictions"],
                        "answers": output["answers"]
                    }
                return results
    
    def load_model(self, model_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Load model state dict
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)
            
        return self 