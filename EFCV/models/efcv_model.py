"""
EFCV Model Implementation

Core implementation of Entity-aware Fusion of Consistency and Visual Clues (EFCV)
for multimodal fake news detection.

This module contains:
- ConsistencyModule: Entity-aware dual consistency measurement with multi-scale fusion
- VisualClueModule: Weight-guided visual evidence extraction
- FusionModule: Evidence theory-based adaptive fusion
- EFCVModel: Complete model integrating all components

Input: Preprocessed entity features (CLIP-encoded)
Output: Classification logits and evidence

Reference:
Jin et al., "EFCV: Entity-aware Fusion of Consistency and Visual Clues 
for Multimodal Fake News Detection", Neurocomputing, 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyModule(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.cosine_enhancement = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.consistency_calculator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.cosine_weight = nn.Parameter(torch.tensor(0.1))
        self.txt_enhance_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.img_enhance_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.evidence_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )
        self.entity_consistency_calculator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2)
        )
        self.entity_evidence_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2)
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, txt_features, img_features):
        txt_features, img_features = txt_features.float(), img_features.float()
        batch_size = txt_features.size(0)
        
        if img_features.dtype != txt_features.dtype:
            img_features = img_features.to(txt_features.dtype)
        
        txt_norm = F.normalize(txt_features, p=2, dim=-1)
        img_norm = F.normalize(img_features, p=2, dim=-1)
        D = txt_norm.size(-1)
        scale = D ** 0.5
        
        t0 = txt_norm[:, :1, :]
        v0 = img_norm[:, :1, :]
        
        s_img_intra = (img_norm * v0).sum(dim=-1) / scale
        s_txt_intra = (txt_norm * t0).sum(dim=-1) / scale
        s_img2txt = (img_norm * t0).sum(dim=-1) / scale
        s_txt2img = (txt_norm * v0).sum(dim=-1) / scale
        
        alpha_s_img = torch.softmax(s_img_intra, dim=-1).unsqueeze(-1)
        beta_s_txt = torch.softmax(s_txt_intra, dim=-1).unsqueeze(-1)
        alpha_cM_img = torch.softmax(s_img2txt, dim=-1).unsqueeze(-1)
        beta_cM_txt = torch.softmax(s_txt2img, dim=-1).unsqueeze(-1)
        alpha_cI_img = torch.softmax(-s_img2txt, dim=-1).unsqueeze(-1)
        beta_cI_txt = torch.softmax(-s_txt2img, dim=-1).unsqueeze(-1)
        
        alpha_M_img = 0.5 * (alpha_s_img + alpha_cM_img)
        beta_M_txt = 0.5 * (beta_s_txt + beta_cM_txt)
        alpha_I_img = 0.5 * (alpha_s_img + alpha_cI_img)
        beta_I_txt = 0.5 * (beta_s_txt + beta_cI_txt)
        
        gamma = self.cosine_weight
        img_scaled = img_features * (1.0 + gamma * alpha_M_img)
        txt_scaled = txt_features * (1.0 + gamma * beta_M_txt)
        
        txt_input = (beta_M_txt * txt_scaled).squeeze(-1) if txt_scaled.dim() == 3 else txt_scaled
        img_input = (alpha_M_img * img_scaled).squeeze(-1) if img_scaled.dim() == 3 else img_scaled
        
        txt_enhanced = txt_features + self.txt_enhance_mlp(txt_input.float())
        img_enhanced = img_features + self.img_enhance_mlp(img_input.float())
        
        txt_attended, _ = self.attention(txt_enhanced, img_enhanced, img_enhanced)
        img_attended, _ = self.attention(img_enhanced, txt_enhanced, txt_enhanced)
        
        txt_encoded = self.transformer(txt_attended)
        img_encoded = self.transformer(img_attended)
        
        txt_global_avg = txt_encoded.mean(dim=1)
        img_global_avg = img_encoded.mean(dim=1)
        
        txt_weights = beta_M_txt.squeeze(-1)
        img_weights = alpha_M_img.squeeze(-1)
        txt_global_weighted = torch.sum(txt_encoded * txt_weights.unsqueeze(-1), dim=1)
        img_global_weighted = torch.sum(img_encoded * img_weights.unsqueeze(-1), dim=1)
        
        txt_entity_features = txt_encoded * txt_weights.unsqueeze(-1)
        img_entity_features = img_encoded * img_weights.unsqueeze(-1)
        txt_entity_global = txt_entity_features.mean(dim=1)
        img_entity_global = img_entity_features.mean(dim=1)
        
        gate_input = torch.cat([txt_global_avg, txt_global_weighted, img_global_avg, img_global_weighted], dim=1)
        scale_weights = self.scale_gate(gate_input)
        
        txt_global = scale_weights[:, 0:1] * txt_global_avg + scale_weights[:, 1:2] * txt_global_weighted + scale_weights[:, 2:3] * txt_entity_global
        img_global = scale_weights[:, 0:1] * img_global_avg + scale_weights[:, 1:2] * img_global_weighted + scale_weights[:, 2:3] * img_entity_global
        
        consistency_features = self.consistency_calculator(torch.cat([txt_global, img_global], dim=1))
        entity_consistency_features = self.entity_consistency_calculator(torch.cat([txt_entity_global, img_entity_global], dim=1))
        
        global_evidence = self.evidence_head(torch.cat([txt_global, img_global], dim=1))
        entity_evidence = self.entity_evidence_head(entity_consistency_features)
        consistency_evidence = global_evidence + 0.3 * entity_evidence
        
        return {
            'txt_global': txt_global,
            'img_global': img_global,
            'consistency_features': consistency_features,
            'entity_consistency_features': entity_consistency_features,
            'evidence': consistency_evidence,
            'txt_encoded': txt_encoded,
            'img_encoded': img_encoded,
            'scale_weights': scale_weights,
            'consistency_weights': {
                'alpha_M_img': alpha_M_img,
                'beta_M_txt': beta_M_txt,
                'alpha_I_img': alpha_I_img,
                'beta_I_txt': beta_I_txt
            },
            'similarity_scores': {
                's_img_intra': s_img_intra,
                's_txt_intra': s_txt_intra,
                's_img2txt': s_img2txt,
                's_txt2img': s_txt2img
            }
        }


class VisualClueModule(nn.Module):
    def __init__(self, d_model=512, hidden_dim=256, dropout=0.1, feature_out=32):
        super().__init__()
        self.d_model = d_model
        self.feature_out = feature_out
        
        self.max_seq_len = 100
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.encoder_layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(3)
        ])
        
        self.txt_projection = nn.Linear(d_model, d_model)
        self.img_projection = nn.Linear(d_model, d_model)
        
        self.visual_detector = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_out)
        )
        
        self.evidence_generator = nn.Sequential(
            nn.Linear(feature_out, feature_out),
            nn.ReLU(),
            nn.Linear(feature_out, 2)
        )
        
        self.weight_projection_txt = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.weight_projection_img = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.weight_scale = nn.Parameter(torch.tensor(0.15))
    
    def forward(self, txt_features, img_features, consistency_weights=None):
        txt_features, img_features = txt_features.float(), img_features.float()
        batch_size = txt_features.size(0)
        
        if consistency_weights is not None:
            alpha_M_img = consistency_weights['alpha_M_img']
            beta_M_txt = consistency_weights['beta_M_txt']
            scale = self.weight_scale
            
            txt_weighted = txt_features * (1.0 + scale * beta_M_txt)
            img_weighted = img_features * (1.0 + scale * alpha_M_img)
            
            beta_weight = beta_M_txt if beta_M_txt.dim() == 3 else beta_M_txt.unsqueeze(-1)
            alpha_weight = alpha_M_img if alpha_M_img.dim() == 3 else alpha_M_img.unsqueeze(-1)
            
            txt_proj = self.txt_projection(txt_weighted) + self.weight_projection_txt(txt_features * beta_weight)
            img_proj = self.img_projection(img_weighted) + self.weight_projection_img(img_features * alpha_weight)
        else:
            txt_proj = self.txt_projection(txt_features)
            img_proj = self.img_projection(img_features)
        
        seq_len = txt_proj.size(1) + img_proj.size(1)
        if seq_len > self.max_seq_len:
            seq_len = self.max_seq_len
        
        txt_with_pos = txt_proj + self.pos_embed[:, :txt_proj.size(1), :]
        img_with_pos = img_proj + self.pos_embed[:, :img_proj.size(1), :]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        combined_features = torch.cat([cls_tokens, txt_with_pos, img_with_pos], dim=1)
        
        encoded_features = self.encoder_layers(combined_features)
        cls_output = encoded_features[:, 0, :]
        
        visual_features = self.visual_detector(torch.cat([cls_output, cls_output], dim=1))
        evidence = F.softplus(self.evidence_generator(visual_features)) + 1
        
        return {
            'visual_features': visual_features,
            'evidence': evidence,
            'encoded_features': encoded_features
        }


class FusionModule(nn.Module):
    def __init__(self, consistency_dim=256, visual_dim=32, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
        self.consistency_classifier = nn.Sequential(
            nn.Linear(consistency_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        self.visual_classifier = nn.Sequential(
            nn.Linear(visual_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self.evidence_fusion = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.Softplus()
        )
        
        self.evidence_gate = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, consistency_output, visual_output):
        consistency_evidence = consistency_output['evidence']
        visual_evidence = visual_output['evidence']
        
        alpha_consistency = F.softplus(consistency_evidence) + 1
        alpha_visual = F.softplus(visual_evidence) + 1
        
        gate_weights = self.evidence_gate(torch.cat([alpha_consistency, alpha_visual], dim=1))
        combined_alpha = gate_weights[:, 0:1] * alpha_consistency + gate_weights[:, 1:2] * alpha_visual
        
        S_combined = torch.sum(combined_alpha, dim=1, keepdim=True)
        prob = combined_alpha / S_combined
        logits = torch.log(combined_alpha / S_combined + 1e-8)
        
        return {
            'logits': logits,
            'prob': prob,
            'alpha': combined_alpha,
            'consistency_evidence': consistency_evidence,
            'visual_evidence': visual_evidence,
            'alpha_consistency': alpha_consistency,
            'alpha_visual': alpha_visual,
            'gate_weights': gate_weights
        }


class EFCVModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, dropout=0.1, feature_out=32, num_classes=2):
        super().__init__()
        
        self.consistency_module = ConsistencyModule(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.visual_module = VisualClueModule(
            d_model=d_model,
            feature_out=feature_out,
            dropout=dropout
        )
        
        self.fusion_module = FusionModule(
            consistency_dim=d_model // 2,
            visual_dim=feature_out,
            num_classes=num_classes
        )
    
    def forward(self, batch_data):
        txt_features = batch_data['text_features']
        img_features = batch_data['img_features']
        
        consistency_output = self.consistency_module(txt_features, img_features)
        
        visual_output = self.visual_module(
            txt_features,
            img_features,
            consistency_weights=consistency_output.get('consistency_weights')
        )
        
        fusion_output = self.fusion_module(consistency_output, visual_output)
        
        return {
            'logits': fusion_output['logits'],
            'prob': fusion_output['prob'],
            'alpha': fusion_output['alpha'],
            'consistency_features': consistency_output['consistency_features'],
            'visual_features': visual_output['visual_features'],
            'evidence': fusion_output['consistency_evidence'] + fusion_output['visual_evidence'],
            'scale_weights': consistency_output['scale_weights'],
            'gate_weights': fusion_output['gate_weights']
        }


def create_efcv_model(d_model=512, nhead=8, num_layers=3, dropout=0.1, feature_out=32, num_classes=2):
    return EFCVModel(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        feature_out=feature_out,
        num_classes=num_classes
    )
