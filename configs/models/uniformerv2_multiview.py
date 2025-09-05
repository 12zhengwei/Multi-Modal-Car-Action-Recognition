"""
UniFormerV2 Multi-View Configuration
Configuration for multi-modal UniFormerV2 model with early fusion.
"""

import torch.nn as nn

UNIFORMERV2_MULTIVIEW_CONFIG = {
    'model_name': 'uniformerv2_multiview',
    'num_classes': 34,  # Drive and Act dataset classes
    'input_channels': 4,  # RGB (3) + KIR (1) for early fusion
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4.0,
    'qkv_bias': True,
    'qk_scale': None,
    'drop_rate': 0.0,
    'attn_drop_rate': 0.0,
    'drop_path_rate': 0.1,
    'norm_layer': nn.LayerNorm,
    'init_values': 0.0,
    'use_learnable_pos_emb': True,
    'tubelet_size': 1,
    'use_mean_pooling': True,
    'pretrained': True,
    'pretrained_path': None
}

# Fusion strategy configuration
FUSION_CONFIG = {
    'early_fusion': {
        'type': 'channel_concat',
        'fusion_layer': 'input',
        'rgb_channels': 3,
        'kir_channels': 1,
        'output_channels': 4
    },
    'late_fusion': {
        'type': 'feature_concat',
        'fusion_layer': 'before_classifier',
        'aggregation': 'concat'  # or 'average', 'max'
    },
    'meta_fusion': {
        'type': 'attention_based',
        'num_modalities': 2,
        'attention_dim': 256,
        'hidden_dim': 512
    }
}

# Training specific configurations
TRAINING_CONFIG = {
    'optimizer': 'adamw',
    'base_lr': 1e-4,
    'min_lr': 1e-6,
    'warmup_lr': 1e-6,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'epochs': 100,
    'batch_size': 8,
    'accumulate_grad_batches': 1,
    'gradient_clip_val': 1.0,
    'scheduler': {
        'type': 'cosine',
        'T_max': 100,
        'eta_min': 1e-6
    }
}