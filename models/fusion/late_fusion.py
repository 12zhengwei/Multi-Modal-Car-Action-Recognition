"""
Late Fusion Strategy
Combine features from separate modality-specific networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusion(nn.Module):
    """Late fusion by combining features from separate streams."""
    
    def __init__(self, feature_dim=768, num_modalities=2, fusion_method='concat',
                 hidden_dim=512, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            self.output_dim = feature_dim * num_modalities
        elif fusion_method in ['average', 'max', 'min']:
            self.output_dim = feature_dim
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim, num_heads=8, dropout=dropout
            )
            self.output_dim = feature_dim
        elif fusion_method == 'mlp':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(feature_dim * num_modalities, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim)
            )
            self.output_dim = feature_dim
        elif fusion_method == 'bilinear':
            # Only works for 2 modalities
            assert num_modalities == 2, "Bilinear fusion only supports 2 modalities"
            self.bilinear = nn.Bilinear(feature_dim, feature_dim, feature_dim)
            self.output_dim = feature_dim
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of feature tensors [B, D] for each modality
        Returns:
            Fused features: [B, output_dim]
        """
        if self.fusion_method == 'concat':
            return torch.cat(modality_features, dim=1)
        
        elif self.fusion_method == 'average':
            return torch.stack(modality_features, dim=0).mean(dim=0)
        
        elif self.fusion_method == 'max':
            return torch.stack(modality_features, dim=0).max(dim=0)[0]
        
        elif self.fusion_method == 'min':
            return torch.stack(modality_features, dim=0).min(dim=0)[0]
        
        elif self.fusion_method == 'attention':
            # Stack features as sequence: [num_modalities, B, D]
            features_seq = torch.stack(modality_features, dim=0)
            
            # Self-attention across modalities
            attended_features, _ = self.attention(features_seq, features_seq, features_seq)
            
            # Average across modalities
            return attended_features.mean(dim=0)
        
        elif self.fusion_method == 'mlp':
            concatenated = torch.cat(modality_features, dim=1)
            return self.fusion_mlp(concatenated)
        
        elif self.fusion_method == 'bilinear':
            return self.bilinear(modality_features[0], modality_features[1])
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class WeightedLateFusion(nn.Module):
    """Late fusion with learnable weights for each modality."""
    
    def __init__(self, feature_dim=768, num_modalities=2, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Modality weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=1)
        )
        
        self.output_dim = feature_dim
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of feature tensors [B, D] for each modality
        Returns:
            Weighted fused features: [B, D]
        """
        # Concatenate features for weight prediction
        concatenated = torch.cat(modality_features, dim=1)  # [B, D*num_modalities]
        
        # Predict modality weights
        weights = self.weight_predictor(concatenated)  # [B, num_modalities]
        
        # Apply weights and sum
        weighted_features = []
        for i, features in enumerate(modality_features):
            weight = weights[:, i:i+1]  # [B, 1]
            weighted_features.append(features * weight)
        
        return sum(weighted_features)


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention fusion for late integration."""
    
    def __init__(self, feature_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Cross-modal attention layers
        self.rgb_to_thermal = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout
        )
        self.thermal_to_rgb = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Final fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.output_dim = feature_dim
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of [rgb_features, thermal_features], each [B, D]
        Returns:
            Cross-attended fused features: [B, D]
        """
        rgb_features, thermal_features = modality_features
        
        # Add sequence dimension for attention: [1, B, D]
        rgb_seq = rgb_features.unsqueeze(0)
        thermal_seq = thermal_features.unsqueeze(0)
        
        # Cross-modal attention
        rgb_attended, _ = self.rgb_to_thermal(rgb_seq, thermal_seq, thermal_seq)
        thermal_attended, _ = self.thermal_to_rgb(thermal_seq, rgb_seq, rgb_seq)
        
        # Remove sequence dimension: [B, D]
        rgb_attended = rgb_attended.squeeze(0)
        thermal_attended = thermal_attended.squeeze(0)
        
        # Apply layer normalization
        rgb_attended = self.norm1(rgb_attended + rgb_features)
        thermal_attended = self.norm2(thermal_attended + thermal_features)
        
        # Final fusion
        concatenated = torch.cat([rgb_attended, thermal_attended], dim=1)
        fused = self.fusion_layer(concatenated)
        
        return fused


class HierarchicalLateFusion(nn.Module):
    """Hierarchical late fusion with multiple levels of integration."""
    
    def __init__(self, feature_dim=768, num_modalities=2, levels=2, 
                 hidden_dims=[512, 256], dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.levels = levels
        
        # Multi-level fusion layers
        self.fusion_layers = nn.ModuleList()
        
        input_dim = feature_dim * num_modalities
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fusion_layers.append(layer)
            input_dim = hidden_dim
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dims[-1], feature_dim)
        self.output_dim = feature_dim
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of feature tensors [B, D] for each modality
        Returns:
            Hierarchically fused features: [B, D]
        """
        # Initial concatenation
        x = torch.cat(modality_features, dim=1)
        
        # Apply hierarchical fusion layers
        for layer in self.fusion_layers:
            x = layer(x)
        
        # Final output
        return self.output_layer(x)


def create_late_fusion(fusion_type='concat', feature_dim=768, num_modalities=2, **kwargs):
    """Factory function to create late fusion modules."""
    
    if fusion_type in ['concat', 'average', 'max', 'min', 'attention', 'mlp', 'bilinear']:
        return LateFusion(
            feature_dim=feature_dim, 
            num_modalities=num_modalities, 
            fusion_method=fusion_type, 
            **kwargs
        )
    elif fusion_type == 'weighted':
        return WeightedLateFusion(
            feature_dim=feature_dim, 
            num_modalities=num_modalities, 
            **kwargs
        )
    elif fusion_type == 'cross_attention':
        return CrossModalAttentionFusion(feature_dim=feature_dim, **kwargs)
    elif fusion_type == 'hierarchical':
        return HierarchicalLateFusion(
            feature_dim=feature_dim, 
            num_modalities=num_modalities, 
            **kwargs
        )
    else:
        raise ValueError(f"Unknown late fusion type: {fusion_type}")


# For backward compatibility
def get_late_fusion_model(fusion_config):
    """Get late fusion model from configuration."""
    fusion_type = fusion_config.get('type', 'concat')
    return create_late_fusion(fusion_type, **fusion_config)