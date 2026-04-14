import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch
from vlmpar.registery import MODELS

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Project queries, keys, and values
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.out_proj(attn_output)

@MODELS.register('vlmpar')
class VLMPAR(nn.Module):
    # Default questions for person attribute recognition
    DEFAULT_QUESTIONS = {
        "upper_body_color": "What is the color of the person's upper body clothing?",
        "lower_body_color": "What is the color of the person's lower body clothing?",
        "gender": "What is the gender of the person?",
        "bag": "Is the person carrying a bag?",
        "hat": "Is the person wearing a hat?"
    }

    def __init__(self, model_name, num_classes_dict, questions=None, use_cross_attention=False):
        super().__init__()
        
        self.vlm_base = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_classes_dict = num_classes_dict
        self.hidden_size = self.vlm_base.vision_model.config.hidden_size
        self.use_cross_attention = use_cross_attention
        
        self.questions = questions if questions is not None else self.DEFAULT_QUESTIONS
        
        for key in num_classes_dict.keys():
            if key not in self.questions:
                raise ValueError(f"Key '{key}' in num_classes_dict has no corresponding question. Please provide a question for this key.")
        
        # Add cross-attention layer only if enabled
        if self.use_cross_attention:
            self.cross_attention = CrossAttention(self.hidden_size)
        
        self.classifier_heads = nn.ModuleDict({
            key: nn.Linear(self.hidden_size, num_labels)
            for key, num_labels in num_classes_dict.items()
        })

    def forward(self, pixel_values, question_type=None, return_dict=True):
        vision_outputs = self.vlm_base.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.pooler_output
        
        if question_type is not None and question_type != 'all':
            if question_type not in self.questions:
                raise ValueError(f"Invalid question type. Must be one of {list(self.questions.keys())}")
            
            question = self.questions[question_type]
            text_inputs = self.tokenizer(question, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(pixel_values.device) for k, v in text_inputs.items()}
            
            text_outputs = self.vlm_base.text_model(**text_inputs)
            text_features = text_outputs.pooler_output
            
            # Apply cross-attention if enabled
            if self.use_cross_attention:
                features = self.cross_attention(
                    query=image_features.unsqueeze(1),
                    key=text_features.unsqueeze(1),
                    value=text_features.unsqueeze(1)
                ).squeeze(1)
            else:
                features = image_features
            
            similarity = torch.cosine_similarity(features, text_features, dim=-1)
            
            logits = self.classifier_heads[question_type](features)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            
            if question_type in ['upper_body_color', 'lower_body_color']:
                answers = [f"The {question_type.replace('_', ' ')} is {self._get_color_name(p.item())}." for p in pred]
            elif question_type == 'gender':
                answers = [f"The person is {'male' if p.item() == 1 else 'female'}." for p in pred]
            elif question_type in ['bag', 'hat']:
                answers = [f"{'Yes' if p.item() == 1 else 'No'}, the person {'is' if p.item() == 1 else 'is not'} {'carrying a bag' if question_type == 'bag' else 'wearing a hat'}." for p in pred]
            
            if return_dict:
                return {
                    "answer": answers,
                    "similarity": similarity.tolist(),
                    "confidence": torch.max(probs, dim=-1)[0].tolist(),
                    "logits": logits,
                    "probs": probs
                }
            return answers
        
        answers = {}
        for key, question in self.questions.items():
            text_inputs = self.tokenizer(question, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(pixel_values.device) for k, v in text_inputs.items()}
            
            text_outputs = self.vlm_base.text_model(**text_inputs)
            text_features = text_outputs.pooler_output
            
            # Apply cross-attention if enabled
            if self.use_cross_attention:
                features = self.cross_attention(
                    query=image_features.unsqueeze(1),
                    key=text_features.unsqueeze(1),
                    value=text_features.unsqueeze(1)
                ).squeeze(1)
            else:
                features = image_features
            
            similarity = torch.cosine_similarity(features, text_features, dim=-1)
            
            logits = self.classifier_heads[key](features)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            
            if key in ['upper_body_color', 'lower_body_color']:
                answers[key] = {
                    "answer": [f"The {key.replace('_', ' ')} is {self._get_color_name(p.item())}." for p in pred],
                    "similarity": similarity.tolist(),
                    "confidence": torch.max(probs, dim=-1)[0].tolist(),
                    "logits": logits,
                    "probs": probs
                }
            elif key == 'gender':
                answers[key] = {
                    "answer": [f"The person is {'male' if p.item() == 1 else 'female'}." for p in pred],
                    "similarity": similarity.tolist(),
                    "confidence": torch.max(probs, dim=-1)[0].tolist(),
                    "logits": logits,
                    "probs": probs
                }
            elif key in ['bag', 'hat']:
                answers[key] = {
                    "answer": [f"{'Yes' if p.item() == 1 else 'No'}, the person {'is' if p.item() == 1 else 'is not'} {'carrying a bag' if key == 'bag' else 'wearing a hat'}." for p in pred],
                    "similarity": similarity.tolist(),
                    "confidence": torch.max(probs, dim=-1)[0].tolist(),
                    "logits": logits,
                    "probs": probs
                }
        
        if return_dict:
            return answers
        else:
            return " ".join([v["answer"][0] for v in answers.values()])

    def _get_color_name(self, color_idx):
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
    
    def predict(self, pixel_values, question_type=None):
        with torch.no_grad():
            outputs = self.forward(pixel_values, question_type=question_type, return_dict=True)
            classes_idx = {}
            classes = {}
            for key in outputs.keys():
                classes_idx[key] = outputs[key]['logits']
                classes_idx[key] = torch.argmax(classes_idx[key], dim=-1).cpu().numpy()[0]
                if key == 'upper_body_color' or key == 'lower_body_color':
                    classes[key] = self._get_color_name(classes_idx[key])
                    classes_idx[key] += 1
                elif key == 'gender':
                    classes[key] = 'male' if classes_idx[key] == 1 else 'female'
                elif key == 'bag' or key == 'hat':
                    classes[key] = 'yes' if classes_idx[key] == 1 else 'no'
                else:
                    classes[key] = 'unknown'
                
                classes_idx[key] = int(classes_idx[key])
        return classes_idx, classes
            
    
    def load_model(self, model_path):
        weights = torch.load(model_path)["model_state_dict"]
        new_weights = {}
        for key, value in weights.items():
            new_key = key.replace("module.", "")
            new_weights[new_key] = value
        self.load_state_dict(new_weights)