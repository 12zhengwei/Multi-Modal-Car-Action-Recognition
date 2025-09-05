"""
UniFormerV2 Implementation for Multi-Modal Video Understanding
Adapted for multi-modal car action recognition with early fusion support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class PatchEmbed3D(nn.Module):
    """3D patch embedding for video input."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, tubelet_size=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            Embedded patches: [B, T', H'*W', D]
        """
        B, C, T, H, W = x.shape
        
        # Apply patch embedding
        x = self.proj(x)  # [B, D, T', H', W']
        
        # Reshape to sequence format
        x = rearrange(x, 'b d t h w -> b t (h w) d')
        
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron with GELU activation."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Block(nn.Module):
    """UniFormer block with local and global attention."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, use_local=True, local_k=7):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop
        )
        
        self.use_local = use_local
        if use_local:
            self.local_k = local_k
            self.local_conv = nn.Conv3d(
                dim, dim, kernel_size=(1, local_k, local_k), 
                padding=(0, local_k//2, local_k//2), groups=dim
            )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, N, D] where N is number of spatial patches
        """
        B, T, N, D = x.shape
        
        # Local convolution branch (if enabled)
        if self.use_local:
            # Reshape for local conv: [B, D, T, H, W]
            H = W = int(N ** 0.5)  # Assume square patches
            x_local = x.permute(0, 3, 1, 2).reshape(B, D, T, H, W)
            x_local = self.local_conv(x_local)
            x_local = x_local.reshape(B, D, T, N).permute(0, 2, 3, 1)
        else:
            x_local = 0
        
        # Global attention branch
        x_flat = x.reshape(B * T, N, D)
        x_attn = self.attn(self.norm1(x_flat))
        x_attn = x_attn.reshape(B, T, N, D)
        
        # Combine local and global features
        x = x + self.drop_path(x_attn + x_local)
        
        # MLP
        x_flat = x.reshape(B * T, N, D)
        x_mlp = self.mlp(self.norm2(x_flat))
        x_mlp = x_mlp.reshape(B, T, N, D)
        x = x + self.drop_path(x_mlp)
        
        return x


class UniFormerV2(nn.Module):
    """UniFormerV2 for multi-modal video understanding."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, tubelet_size=1,
                 use_learnable_pos_emb=True, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or nn.LayerNorm
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, tubelet_size=tubelet_size
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Positional embeddings
        self.use_learnable_pos_emb = use_learnable_pos_emb
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_local=(i < depth // 2), local_k=7
            ) for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        
        # Only initialize head weights if it's a Linear layer
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        """Extract features without classification head."""
        B, C, T, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, T', N, D]
        B, T_new, N, D = x.shape
        
        # Add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(1)  # Broadcast across time
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=(1, 2))  # Average over time and spatial dimensions
        
        return x
    
    def forward(self, x):
        """Forward pass with classification."""
        x = self.forward_features(x)
        x = self.head(x)
        return x


def uniformerv2_small(**kwargs):
    """UniFormerV2 Small model."""
    model = UniFormerV2(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        **kwargs
    )
    return model


def uniformerv2_base(**kwargs):
    """UniFormerV2 Base model."""
    # Set base configuration defaults
    base_config = {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4
    }
    
    # Update with provided kwargs, avoiding duplicates
    for key, value in kwargs.items():
        base_config[key] = value
    
    model = UniFormerV2(**base_config)
    return model


def uniformerv2_large(**kwargs):
    """UniFormerV2 Large model."""
    model = UniFormerV2(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        **kwargs
    )
    return model


def create_uniformerv2_multimodal(num_classes=34, in_chans=4, **kwargs):
    """Create UniFormerV2 for multi-modal input (RGB + thermal)."""
    # Remove conflicting parameters
    kwargs.pop('num_classes', None)  # Remove if exists to avoid conflict
    kwargs.pop('in_chans', None)     # Remove if exists to avoid conflict
    
    return uniformerv2_base(
        num_classes=num_classes,
        in_chans=in_chans,  # 4 channels for early fusion (RGB + thermal)
        **kwargs
    )