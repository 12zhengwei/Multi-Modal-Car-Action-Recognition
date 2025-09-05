"""
Classification Head for Action Recognition
Various classification heads for multi-modal action recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LinearClassificationHead(nn.Module):
    """Simple linear classification head."""
    
    def __init__(self, input_dim=768, num_classes=34, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
        
        # Initialize weights
        init.normal_(self.classifier.weight, std=0.02)
        init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, input_dim]
        Returns:
            Class logits [B, num_classes]
        """
        x = self.dropout(x)
        return self.classifier(x)


class MLPClassificationHead(nn.Module):
    """MLP classification head with hidden layers."""
    
    def __init__(self, input_dim=768, hidden_dims=[512, 256], num_classes=34, 
                 dropout=0.1, activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, input_dim]
        Returns:
            Class logits [B, num_classes]
        """
        return self.classifier(x)


class AttentionClassificationHead(nn.Module):
    """Classification head with self-attention mechanism."""
    
    def __init__(self, input_dim=768, num_classes=34, num_heads=8, 
                 dropout=0.1, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(input_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Final classifier
        self.classifier = nn.Linear(input_dim, num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.ones_(m.weight)
            init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, input_dim]
        Returns:
            Class logits [B, num_classes]
        """
        # Add sequence dimension for attention
        x_seq = x.unsqueeze(1)  # [B, 1, input_dim]
        x_seq = x_seq.transpose(0, 1)  # [1, B, input_dim]
        
        # Self-attention
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        x_attn = self.norm1(attn_output.squeeze(0) + x)  # [B, input_dim]
        
        # Feed-forward
        x_ffn = self.ffn(x_attn)
        x_out = self.norm2(x_ffn + x_attn)
        
        # Classification
        return self.classifier(x_out)


class MultiScaleClassificationHead(nn.Module):
    """Multi-scale classification head with different receptive fields."""
    
    def __init__(self, input_dim=768, num_classes=34, scales=[1, 2, 4], 
                 dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scales = scales
        
        # Multi-scale feature extractors
        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // scale),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for scale in scales
        ])
        
        # Fusion layer
        total_dim = sum([input_dim // scale for scale in scales])
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classifier
        self.classifier = nn.Linear(input_dim, num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, input_dim]
        Returns:
            Class logits [B, num_classes]
        """
        # Extract multi-scale features
        scale_features = []
        for extractor in self.scale_extractors:
            scale_features.append(extractor(x))
        
        # Concatenate and fuse
        concatenated = torch.cat(scale_features, dim=1)
        fused_features = self.fusion(concatenated)
        
        # Classification
        return self.classifier(fused_features)


class HierarchicalClassificationHead(nn.Module):
    """Hierarchical classification head for structured predictions."""
    
    def __init__(self, input_dim=768, num_classes=34, hierarchy_levels=[17, 34],
                 dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Total number of fine-grained classes
            hierarchy_levels: Number of classes at each hierarchy level
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hierarchy_levels = hierarchy_levels
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hierarchy-specific classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(input_dim, level_classes) 
            for level_classes in hierarchy_levels
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, input_dim]
        Returns:
            List of logits for each hierarchy level
        """
        # Extract shared features
        shared = self.shared_features(x)
        
        # Get predictions at each hierarchy level
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier(shared))
        
        return predictions


class UncertaintyClassificationHead(nn.Module):
    """Classification head with uncertainty estimation."""
    
    def __init__(self, input_dim=768, num_classes=34, hidden_dim=512,
                 dropout=0.1, num_mc_samples=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_mc_samples = num_mc_samples
        
        # Feature extractor with dropout for MC sampling
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Uncertainty estimation layer
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.zeros_(m.bias)
    
    def forward(self, x, return_uncertainty=False):
        """
        Args:
            x: Input features [B, input_dim]
            return_uncertainty: Whether to return uncertainty estimates
        Returns:
            Class logits [B, num_classes] and optionally uncertainty [B, 1]
        """
        if return_uncertainty and self.training:
            # Monte Carlo sampling for uncertainty estimation
            predictions = []
            uncertainties = []
            
            for _ in range(self.num_mc_samples):
                features = self.features(x)
                pred = self.classifier(features)
                unc = self.uncertainty_head(features)
                
                predictions.append(pred)
                uncertainties.append(unc)
            
            # Average predictions and compute uncertainty
            mean_pred = torch.stack(predictions).mean(dim=0)
            pred_var = torch.stack(predictions).var(dim=0).mean(dim=1, keepdim=True)
            epistemic_unc = torch.stack(uncertainties).mean(dim=0)
            
            total_uncertainty = pred_var + epistemic_unc
            
            return mean_pred, total_uncertainty
        else:
            # Standard forward pass
            features = self.features(x)
            logits = self.classifier(features)
            
            if return_uncertainty:
                uncertainty = self.uncertainty_head(features)
                return logits, uncertainty
            else:
                return logits


def create_classification_head(head_type='linear', input_dim=768, num_classes=34, **kwargs):
    """Factory function to create classification heads."""
    
    if head_type == 'linear':
        return LinearClassificationHead(input_dim, num_classes, **kwargs)
    elif head_type == 'mlp':
        return MLPClassificationHead(input_dim, num_classes=num_classes, **kwargs)
    elif head_type == 'attention':
        return AttentionClassificationHead(input_dim, num_classes, **kwargs)
    elif head_type == 'multiscale':
        return MultiScaleClassificationHead(input_dim, num_classes, **kwargs)
    elif head_type == 'hierarchical':
        return HierarchicalClassificationHead(input_dim, num_classes, **kwargs)
    elif head_type == 'uncertainty':
        return UncertaintyClassificationHead(input_dim, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown classification head type: {head_type}")


# For backward compatibility
def get_classification_head(head_config):
    """Get classification head from configuration."""
    head_type = head_config.get('type', 'linear')
    return create_classification_head(head_type, **head_config)