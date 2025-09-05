"""
Multi-Modal Action Recognizer
Main recognizer that integrates backbone, fusion, and classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.uniformerv2 import create_uniformerv2_multimodal
from models.fusion.early_fusion import create_early_fusion
from models.fusion.late_fusion import create_late_fusion
from models.fusion.meta_fusion import create_meta_fusion
from models.heads.classification_head import create_classification_head
from configs.datasets.drive_act import ACTION_CLASSES


class MultiModalActionRecognizer(nn.Module):
    """Multi-modal action recognizer for car cabin environment."""
    
    def __init__(self, num_classes=34, fusion_type='early', backbone_config=None,
                 fusion_config=None, head_config=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.action_classes = ACTION_CLASSES
        
        # Set default configurations
        if backbone_config is None:
            backbone_config = {
                'img_size': 224,
                'patch_size': 16,
                'in_chans': 4 if fusion_type == 'early' else 3,
                'num_classes': 0,  # No classifier in backbone
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'tubelet_size': 1
            }
        
        if fusion_config is None:
            fusion_config = {'fusion_type': 'simple'}
            
        if head_config is None:
            head_config = {'head_type': 'linear'}
        
        # Initialize components based on fusion type
        if fusion_type == 'early':
            self._init_early_fusion(backbone_config, fusion_config, head_config)
        elif fusion_type == 'late':
            self._init_late_fusion(backbone_config, fusion_config, head_config)
        elif fusion_type == 'meta':
            self._init_meta_fusion(backbone_config, fusion_config, head_config)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def _init_early_fusion(self, backbone_config, fusion_config, head_config):
        """Initialize early fusion architecture."""
        # Fusion module (optional preprocessing)
        self.fusion_module = create_early_fusion(**fusion_config)
        
        # Backbone with 4-channel input (RGB + thermal)
        # Create a clean config without conflicts
        clean_backbone_config = backbone_config.copy()
        clean_backbone_config['in_chans'] = 4
        clean_backbone_config['num_classes'] = 0  # No classifier in backbone
        
        self.backbone = create_uniformerv2_multimodal(**clean_backbone_config)
        
        # Classification head
        feature_dim = backbone_config.get('embed_dim', 768)
        self.classifier = create_classification_head(
            input_dim=feature_dim, num_classes=self.num_classes, **head_config
        )
    
    def _init_late_fusion(self, backbone_config, fusion_config, head_config):
        """Initialize late fusion architecture."""
        # Separate backbones for each modality
        rgb_config = backbone_config.copy()
        rgb_config['in_chans'] = 3  # RGB
        rgb_config['num_classes'] = 0  # No classifier in backbone
        self.rgb_backbone = create_uniformerv2_multimodal(**rgb_config)
        
        thermal_config = backbone_config.copy()
        thermal_config['in_chans'] = 1  # Thermal
        thermal_config['num_classes'] = 0  # No classifier in backbone
        self.thermal_backbone = create_uniformerv2_multimodal(**thermal_config)
        
        # Fusion module
        feature_dim = backbone_config.get('embed_dim', 768)
        fusion_config.update({
            'feature_dim': feature_dim,
            'num_modalities': 2
        })
        self.fusion_module = create_late_fusion(**fusion_config)
        
        # Classification head
        self.classifier = create_classification_head(
            input_dim=self.fusion_module.output_dim, 
            num_classes=self.num_classes, 
            **head_config
        )
    
    def _init_meta_fusion(self, backbone_config, fusion_config, head_config):
        """Initialize meta fusion architecture."""
        # Separate backbones for each modality
        rgb_config = backbone_config.copy()
        rgb_config['in_chans'] = 3  # RGB
        rgb_config['num_classes'] = 0  # No classifier in backbone
        self.rgb_backbone = create_uniformerv2_multimodal(**rgb_config)
        
        thermal_config = backbone_config.copy()
        thermal_config['in_chans'] = 1  # Thermal
        thermal_config['num_classes'] = 0  # No classifier in backbone
        self.thermal_backbone = create_uniformerv2_multimodal(**thermal_config)
        
        # Meta fusion module
        feature_dim = backbone_config.get('embed_dim', 768)
        fusion_config.update({
            'feature_dim': feature_dim,
            'num_modalities': 2
        })
        self.fusion_module = create_meta_fusion(**fusion_config)
        
        # Classification head
        self.classifier = create_classification_head(
            input_dim=self.fusion_module.output_dim,
            num_classes=self.num_classes,
            **head_config
        )
    
    def forward(self, video_data, return_features=False):
        """
        Forward pass through the multi-modal recognizer.
        
        Args:
            video_data: Dict containing video tensors for different modalities
                       For early fusion: {'fused': [B, 4, T, H, W]}
                       For late/meta fusion: {'rgb': [B, 3, T, H, W], 'kir': [B, 1, T, H, W]}
            return_features: Whether to return intermediate features
        Returns:
            Class logits [B, num_classes] and optionally features
        """
        if self.fusion_type == 'early':
            return self._forward_early_fusion(video_data, return_features)
        elif self.fusion_type == 'late':
            return self._forward_late_fusion(video_data, return_features)
        elif self.fusion_type == 'meta':
            return self._forward_meta_fusion(video_data, return_features)
    
    def _forward_early_fusion(self, video_data, return_features=False):
        """Forward pass for early fusion."""
        if 'fused' in video_data:
            fused_input = video_data['fused']  # [B, 4, T, H, W]
        else:
            # Fuse RGB and thermal on the fly
            rgb_input = video_data['rgb']      # [B, 3, T, H, W]
            thermal_input = video_data['kir']  # [B, 1, T, H, W]
            fused_input = self.fusion_module(rgb_input, thermal_input)
        
        # Extract features using backbone
        features = self.backbone.forward_features(fused_input)  # [B, D]
        
        # Classify
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def _forward_late_fusion(self, video_data, return_features=False):
        """Forward pass for late fusion."""
        rgb_input = video_data['rgb']      # [B, 3, T, H, W]
        thermal_input = video_data['kir']  # [B, 1, T, H, W]
        
        # Extract features from separate backbones
        rgb_features = self.rgb_backbone.forward_features(rgb_input)      # [B, D]
        thermal_features = self.thermal_backbone.forward_features(thermal_input)  # [B, D]
        
        # Fuse features
        fused_features = self.fusion_module([rgb_features, thermal_features])  # [B, D']
        
        # Classify
        logits = self.classifier(fused_features)
        
        if return_features:
            return logits, {'rgb': rgb_features, 'thermal': thermal_features, 'fused': fused_features}
        return logits
    
    def _forward_meta_fusion(self, video_data, return_features=False):
        """Forward pass for meta fusion."""
        rgb_input = video_data['rgb']      # [B, 3, T, H, W]
        thermal_input = video_data['kir']  # [B, 1, T, H, W]
        
        # Extract features from separate backbones
        rgb_features = self.rgb_backbone.forward_features(rgb_input)      # [B, D]
        thermal_features = self.thermal_backbone.forward_features(thermal_input)  # [B, D]
        
        # Meta fusion
        fused_features = self.fusion_module([rgb_features, thermal_features])  # [B, D']
        
        # Classify
        logits = self.classifier(fused_features)
        
        if return_features:
            return logits, {'rgb': rgb_features, 'thermal': thermal_features, 'fused': fused_features}
        return logits
    
    def predict(self, video_data, return_probabilities=True):
        """
        Make predictions on input data.
        
        Args:
            video_data: Input video data
            return_probabilities: Whether to return probabilities or logits
        Returns:
            Predictions and class names
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(video_data)
            
            if return_probabilities:
                probs = F.softmax(logits, dim=1)
                predicted_indices = probs.argmax(dim=1)
                confidence_scores = probs.max(dim=1)[0]
                
                predicted_classes = [self.action_classes[idx] for idx in predicted_indices.cpu().numpy()]
                
                return {
                    'predictions': predicted_classes,
                    'indices': predicted_indices.cpu().numpy(),
                    'probabilities': probs.cpu().numpy(),
                    'confidence': confidence_scores.cpu().numpy()
                }
            else:
                predicted_indices = logits.argmax(dim=1)
                predicted_classes = [self.action_classes[idx] for idx in predicted_indices.cpu().numpy()]
                
                return {
                    'predictions': predicted_classes,
                    'indices': predicted_indices.cpu().numpy(),
                    'logits': logits.cpu().numpy()
                }
    
    def get_model_size(self):
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.rgb_backbone.parameters():
                param.requires_grad = False
            for param in self.thermal_backbone.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for param in self.rgb_backbone.parameters():
                param.requires_grad = True
            for param in self.thermal_backbone.parameters():
                param.requires_grad = True


def create_recognizer(fusion_type='early', num_classes=34, **kwargs):
    """Factory function to create multi-modal recognizer."""
    return MultiModalActionRecognizer(
        num_classes=num_classes,
        fusion_type=fusion_type,
        **kwargs
    )


def load_pretrained_recognizer(checkpoint_path, fusion_type='early', num_classes=34, **kwargs):
    """Load pretrained recognizer from checkpoint."""
    model = create_recognizer(fusion_type=fusion_type, num_classes=num_classes, **kwargs)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model