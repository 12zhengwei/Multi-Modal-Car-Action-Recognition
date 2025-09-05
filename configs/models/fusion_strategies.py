"""
Fusion Strategies Configuration
Configuration for different fusion approaches in multi-modal learning.
"""

FUSION_STRATEGIES = {
    'early_fusion': {
        'name': 'Early Fusion',
        'description': 'Concatenate modalities at input level',
        'fusion_point': 'input',
        'advantages': ['Simple implementation', 'End-to-end learning'],
        'disadvantages': ['Fixed fusion strategy', 'Potential modality dominance']
    },
    
    'late_fusion': {
        'name': 'Late Fusion',
        'description': 'Combine features from separate modality streams',
        'fusion_point': 'features',
        'advantages': ['Modality-specific learning', 'Flexible combination'],
        'disadvantages': ['More parameters', 'Training complexity']
    },
    
    'meta_fusion': {
        'name': 'Meta Fusion',
        'description': 'Learnable attention-based fusion mechanism',
        'fusion_point': 'adaptive',
        'advantages': ['Adaptive weighting', 'Handles missing modalities'],
        'disadvantages': ['Computational overhead', 'More complex training']
    }
}

# Configuration for the selected early fusion approach
EARLY_FUSION_CONFIG = {
    'input_preprocessing': {
        'rgb_normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'kir_normalization': {
            'mean': [0.5],
            'std': [0.5]
        },
        'resize_size': 256,
        'crop_size': 224,
        'temporal_sampling': {
            'num_frames': 16,
            'sampling_rate': 4
        }
    },
    
    'fusion_mechanism': {
        'type': 'channel_concatenation',
        'input_dims': {
            'rgb': [3, 16, 224, 224],  # [C, T, H, W]
            'kir': [1, 16, 224, 224]
        },
        'output_dim': [4, 16, 224, 224],  # Concatenated channels
        'alignment_strategy': 'temporal_interpolation'
    }
}