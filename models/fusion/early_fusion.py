"""
Early Fusion Strategy
Concatenate modalities at input level before feeding to the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusion(nn.Module):
    """Early fusion by channel concatenation."""
    
    def __init__(self, rgb_channels=3, thermal_channels=1, fusion_method='concat'):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.thermal_channels = thermal_channels
        self.fusion_method = fusion_method
        self.output_channels = rgb_channels + thermal_channels
        
        if fusion_method == 'weighted_concat':
            self.rgb_weight = nn.Parameter(torch.ones(1))
            self.thermal_weight = nn.Parameter(torch.ones(1))
        elif fusion_method == 'learned_concat':
            self.fusion_conv = nn.Conv3d(
                rgb_channels + thermal_channels, 
                rgb_channels + thermal_channels,
                kernel_size=1, bias=True
            )
    
    def forward(self, rgb_input, thermal_input):
        """
        Args:
            rgb_input: [B, 3, T, H, W]
            thermal_input: [B, 1, T, H, W]
        Returns:
            Fused input: [B, 4, T, H, W]
        """
        if self.fusion_method == 'concat':
            # Simple concatenation
            return torch.cat([rgb_input, thermal_input], dim=1)
        
        elif self.fusion_method == 'weighted_concat':
            # Weighted concatenation
            weighted_rgb = rgb_input * torch.sigmoid(self.rgb_weight)
            weighted_thermal = thermal_input * torch.sigmoid(self.thermal_weight)
            return torch.cat([weighted_rgb, weighted_thermal], dim=1)
        
        elif self.fusion_method == 'learned_concat':
            # Learned fusion with 1x1 convolution
            concatenated = torch.cat([rgb_input, thermal_input], dim=1)
            return self.fusion_conv(concatenated)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class EarlyFusionWithAlignment(nn.Module):
    """Early fusion with temporal and spatial alignment."""
    
    def __init__(self, rgb_channels=3, thermal_channels=1, align_method='interpolation'):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.thermal_channels = thermal_channels
        self.align_method = align_method
        
        if align_method == 'learned_alignment':
            # Learnable alignment module
            self.spatial_align = nn.Conv2d(thermal_channels, thermal_channels, 
                                         kernel_size=3, padding=1)
            self.temporal_align = nn.Conv1d(thermal_channels, thermal_channels, 
                                          kernel_size=3, padding=1)
    
    def spatial_alignment(self, thermal_input, rgb_input):
        """Align thermal input spatially to RGB input."""
        if thermal_input.shape[-2:] != rgb_input.shape[-2:]:
            # Resize thermal to match RGB spatial dimensions
            B, C, T, H_t, W_t = thermal_input.shape
            _, _, _, H_r, W_r = rgb_input.shape
            
            thermal_resized = F.interpolate(
                thermal_input.view(B * C * T, 1, H_t, W_t),
                size=(H_r, W_r), mode='bilinear', align_corners=False
            ).view(B, C, T, H_r, W_r)
            
            return thermal_resized
        return thermal_input
    
    def temporal_alignment(self, thermal_input, rgb_input):
        """Align thermal input temporally to RGB input."""
        if thermal_input.shape[2] != rgb_input.shape[2]:
            # Interpolate thermal to match RGB temporal dimension
            B, C, T_t, H, W = thermal_input.shape
            T_r = rgb_input.shape[2]
            
            thermal_aligned = F.interpolate(
                thermal_input.permute(0, 1, 3, 4, 2).contiguous().view(B * C * H * W, T_t),
                size=T_r, mode='linear', align_corners=False
            ).view(B, C, H, W, T_r).permute(0, 1, 4, 2, 3)
            
            return thermal_aligned
        return thermal_input
    
    def forward(self, rgb_input, thermal_input):
        """
        Args:
            rgb_input: [B, 3, T, H, W]
            thermal_input: [B, 1, T', H', W']
        Returns:
            Fused input: [B, 4, T, H, W]
        """
        # Align thermal to RGB dimensions
        thermal_aligned = self.spatial_alignment(thermal_input, rgb_input)
        thermal_aligned = self.temporal_alignment(thermal_aligned, rgb_input)
        
        if self.align_method == 'learned_alignment':
            # Apply learned alignment
            B, C, T, H, W = thermal_aligned.shape
            thermal_spatial = self.spatial_align(
                thermal_aligned.view(B * T, C, H, W)
            ).view(B, C, T, H, W)
            
            thermal_temporal = self.temporal_align(
                thermal_spatial.view(B * H * W, C, T)
            ).view(B, C, T, H, W)
            
            thermal_aligned = thermal_temporal
        
        # Concatenate aligned modalities
        return torch.cat([rgb_input, thermal_aligned], dim=1)


class AdaptiveEarlyFusion(nn.Module):
    """Adaptive early fusion with learned modality importance."""
    
    def __init__(self, rgb_channels=3, thermal_channels=1, hidden_dim=64):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.thermal_channels = thermal_channels
        
        # Modality importance network
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(rgb_channels + thermal_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # RGB and thermal importance weights
            nn.Softmax(dim=1)
        )
        
        # Feature enhancement
        self.rgb_enhance = nn.Conv3d(rgb_channels, rgb_channels, 
                                   kernel_size=3, padding=1, groups=rgb_channels)
        self.thermal_enhance = nn.Conv3d(thermal_channels, thermal_channels, 
                                       kernel_size=3, padding=1, groups=thermal_channels)
    
    def forward(self, rgb_input, thermal_input):
        """
        Args:
            rgb_input: [B, 3, T, H, W]
            thermal_input: [B, 1, T, H, W]
        Returns:
            Adaptively fused input: [B, 4, T, H, W]
        """
        # Simple concatenation for importance estimation
        concat_input = torch.cat([rgb_input, thermal_input], dim=1)
        
        # Compute modality importance weights
        importance_weights = self.importance_net(concat_input)  # [B, 2]
        rgb_weight = importance_weights[:, 0:1].view(-1, 1, 1, 1, 1)
        thermal_weight = importance_weights[:, 1:2].view(-1, 1, 1, 1, 1)
        
        # Enhance features
        rgb_enhanced = self.rgb_enhance(rgb_input)
        thermal_enhanced = self.thermal_enhance(thermal_input)
        
        # Apply importance weighting
        rgb_weighted = rgb_enhanced * rgb_weight
        thermal_weighted = thermal_enhanced * thermal_weight
        
        # Final fusion
        return torch.cat([rgb_weighted, thermal_weighted], dim=1)


def create_early_fusion(fusion_type='simple', **kwargs):
    """Factory function to create early fusion modules."""
    
    if fusion_type == 'simple':
        return EarlyFusion(fusion_method='concat', **kwargs)
    elif fusion_type == 'weighted':
        return EarlyFusion(fusion_method='weighted_concat', **kwargs)
    elif fusion_type == 'learned':
        return EarlyFusion(fusion_method='learned_concat', **kwargs)
    elif fusion_type == 'aligned':
        return EarlyFusionWithAlignment(**kwargs)
    elif fusion_type == 'adaptive':
        return AdaptiveEarlyFusion(**kwargs)
    else:
        raise ValueError(f"Unknown early fusion type: {fusion_type}")


# For backward compatibility
def get_early_fusion_model(fusion_config):
    """Get early fusion model from configuration."""
    fusion_type = fusion_config.get('type', 'simple')
    return create_early_fusion(fusion_type, **fusion_config)