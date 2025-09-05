"""
Meta Fusion Strategy
Adaptive fusion with learnable attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MetaFusionNetwork(nn.Module):
    """Meta-learning based fusion network."""
    
    def __init__(self, feature_dim=768, num_modalities=2, hidden_dim=256, 
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_modalities)
        ])
        
        # Meta-attention mechanism
        self.meta_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        
        # Fusion controller
        self.fusion_controller = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.output_dim = feature_dim
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of feature tensors [B, D] for each modality
        Returns:
            Meta-fused features: [B, D]
        """
        batch_size = modality_features[0].shape[0]
        
        # Encode each modality
        encoded_features = []
        for i, features in enumerate(modality_features):
            encoded = self.modality_encoders[i](features)
            encoded_features.append(encoded)
        
        # Stack for attention: [num_modalities, B, hidden_dim]
        features_stack = torch.stack(encoded_features, dim=0)
        
        # Apply meta-attention
        attended_features, attention_weights = self.meta_attention(
            features_stack, features_stack, features_stack
        )
        
        # Compute fusion weights using controller
        concatenated = torch.cat(encoded_features, dim=1)  # [B, hidden_dim * num_modalities]
        fusion_weights = self.fusion_controller(concatenated)  # [B, num_modalities]
        
        # Apply fusion weights
        weighted_features = []
        for i in range(self.num_modalities):
            weight = fusion_weights[:, i:i+1]  # [B, 1]
            weighted = attended_features[i] * weight  # [B, hidden_dim]
            weighted_features.append(weighted)
        
        # Sum weighted features
        fused_features = sum(weighted_features)  # [B, hidden_dim]
        
        # Project to output dimension
        output = self.output_proj(fused_features)
        
        return output


class AdaptiveModalityFusion(nn.Module):
    """Adaptive fusion that handles missing or unreliable modalities."""
    
    def __init__(self, feature_dim=768, num_modalities=2, confidence_threshold=0.1,
                 hidden_dim=256, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.confidence_threshold = confidence_threshold
        
        # Modality confidence estimator
        self.confidence_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # Feature refinement networks
        self.feature_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim)
            ) for _ in range(num_modalities)
        ])
        
        # Adaptive fusion layer
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim * num_modalities + num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.output_dim = feature_dim
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of feature tensors [B, D] for each modality
        Returns:
            Adaptively fused features: [B, D]
        """
        batch_size = modality_features[0].shape[0]
        
        # Estimate confidence for each modality
        confidences = []
        refined_features = []
        
        for i, features in enumerate(modality_features):
            # Estimate confidence
            confidence = self.confidence_estimator[i](features)  # [B, 1]
            confidences.append(confidence)
            
            # Refine features
            refined = self.feature_refiners[i](features)
            refined_features.append(refined)
        
        # Apply confidence-based weighting
        weighted_features = []
        for i, (features, confidence) in enumerate(zip(refined_features, confidences)):
            # Only use features above confidence threshold
            mask = (confidence > self.confidence_threshold).float()
            weighted = features * confidence * mask
            weighted_features.append(weighted)
        
        # Concatenate weighted features and confidences
        concatenated_features = torch.cat(weighted_features, dim=1)
        concatenated_confidences = torch.cat(confidences, dim=1)
        fusion_input = torch.cat([concatenated_features, concatenated_confidences], dim=1)
        
        # Apply fusion network
        fused_output = self.fusion_net(fusion_input)
        
        return fused_output


class ContextualMetaFusion(nn.Module):
    """Context-aware meta fusion using scene understanding."""
    
    def __init__(self, feature_dim=768, num_modalities=2, context_dim=128,
                 hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.context_dim = context_dim
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_dim)
        )
        
        # Context-conditioned attention
        self.context_attention = nn.Sequential(
            nn.Linear(context_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=1)
        )
        
        # Modality-specific transformers
        self.modality_transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                dropout=dropout, batch_first=True
            ) for _ in range(num_modalities)
        ])
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim * num_modalities + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.output_dim = feature_dim
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of feature tensors [B, D] for each modality
        Returns:
            Context-aware fused features: [B, D]
        """
        batch_size = modality_features[0].shape[0]
        
        # Encode context from all modalities
        concatenated = torch.cat(modality_features, dim=1)
        context = self.context_encoder(concatenated)  # [B, context_dim]
        
        # Transform each modality with context
        transformed_features = []
        for i, features in enumerate(modality_features):
            # Add sequence dimension for transformer
            features_seq = features.unsqueeze(1)  # [B, 1, D]
            
            # Apply modality-specific transformer
            transformed = self.modality_transformers[i](features_seq)
            transformed = transformed.squeeze(1)  # [B, D]
            transformed_features.append(transformed)
        
        # Compute context-conditioned attention weights
        attention_weights = []
        for features in transformed_features:
            context_features = torch.cat([context, features], dim=1)
            weights = self.context_attention(context_features)  # [B, num_modalities]
            attention_weights.append(weights)
        
        # Average attention weights across modalities
        avg_attention = sum(attention_weights) / len(attention_weights)
        
        # Apply attention weights
        weighted_features = []
        for i, features in enumerate(transformed_features):
            weight = avg_attention[:, i:i+1]  # [B, 1]
            weighted = features * weight
            weighted_features.append(weighted)
        
        # Final fusion with context
        final_input = torch.cat(weighted_features + [context], dim=1)
        fused_output = self.final_fusion(final_input)
        
        return fused_output


class TemporalMetaFusion(nn.Module):
    """Temporal-aware meta fusion for video sequences."""
    
    def __init__(self, feature_dim=768, num_modalities=2, temporal_dim=16,
                 hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.temporal_dim = temporal_dim
        
        # Temporal encoding
        self.temporal_encoder = nn.LSTM(
            input_size=feature_dim * num_modalities,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional LSTM
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Modality fusion
        self.modality_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.output_dim = feature_dim
    
    def forward(self, modality_features_sequence):
        """
        Args:
            modality_features_sequence: List of [B, T, D] tensors for each modality
        Returns:
            Temporal-aware fused features: [B, D]
        """
        # Concatenate modalities along feature dimension
        concatenated_sequence = torch.cat(modality_features_sequence, dim=2)  # [B, T, D*num_modalities]
        
        # Apply temporal encoding
        temporal_encoded, _ = self.temporal_encoder(concatenated_sequence)  # [B, T, hidden_dim*2]
        
        # Apply temporal attention
        attended_features, _ = self.temporal_attention(
            temporal_encoded, temporal_encoded, temporal_encoded
        )  # [B, T, hidden_dim*2]
        
        # Global temporal pooling (average over time)
        pooled_features = attended_features.mean(dim=1)  # [B, hidden_dim*2]
        
        # Final fusion
        fused_output = self.modality_fusion(pooled_features)
        
        return fused_output


def create_meta_fusion(fusion_type='basic', **kwargs):
    """Factory function to create meta fusion modules."""
    
    if fusion_type == 'basic':
        return MetaFusionNetwork(**kwargs)
    elif fusion_type == 'adaptive':
        return AdaptiveModalityFusion(**kwargs)
    elif fusion_type == 'contextual':
        return ContextualMetaFusion(**kwargs)
    elif fusion_type == 'temporal':
        return TemporalMetaFusion(**kwargs)
    else:
        raise ValueError(f"Unknown meta fusion type: {fusion_type}")


# For backward compatibility
def get_meta_fusion_model(fusion_config):
    """Get meta fusion model from configuration."""
    fusion_type = fusion_config.get('type', 'basic')
    return create_meta_fusion(fusion_type, **fusion_config)