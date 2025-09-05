"""
Data Utilities
Utility functions for data processing and handling.
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict


def check_dataset_structure(root_dir):
    """Check if the dataset structure is correct."""
    expected_structure = {
        'splits': ['split0', 'split1', 'split2'],
        'video_types': ['video_train', 'video_val', 'video_test'],
        'modalities': ['rgb', 'kir']
    }
    
    issues = []
    
    if not os.path.exists(root_dir):
        issues.append(f"Root directory {root_dir} does not exist")
        return issues
    
    # Check splits
    for split in expected_structure['splits']:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            issues.append(f"Split directory {split_path} does not exist")
            continue
        
        # Check video types
        for video_type in expected_structure['video_types']:
            video_type_path = os.path.join(split_path, video_type)
            if not os.path.exists(video_type_path):
                issues.append(f"Video type directory {video_type_path} does not exist")
                continue
            
            # Check modalities
            for modality in expected_structure['modalities']:
                modality_path = os.path.join(video_type_path, modality)
                if not os.path.exists(modality_path):
                    issues.append(f"Modality directory {modality_path} does not exist")
    
    return issues


def count_videos_per_class(root_dir, split='split0', video_type='video_train'):
    """Count the number of videos per class for a given split and video type."""
    from configs.datasets.drive_act import ACTION_CLASSES
    
    video_counts = defaultdict(dict)
    
    for modality in ['rgb', 'kir']:
        video_counts[modality] = {}
        
        for class_name in ACTION_CLASSES:
            class_path = os.path.join(root_dir, split, video_type, modality, class_name)
            if os.path.exists(class_path):
                video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
                video_counts[modality][class_name] = len(video_files)
            else:
                video_counts[modality][class_name] = 0
    
    return video_counts


def analyze_dataset_balance(root_dir):
    """Analyze dataset balance across classes and modalities."""
    from configs.datasets.drive_act import ACTION_CLASSES
    
    analysis = {}
    
    for split in ['split0', 'split1', 'split2']:
        analysis[split] = {}
        
        for video_type in ['video_train', 'video_val', 'video_test']:
            counts = count_videos_per_class(root_dir, split, video_type)
            analysis[split][video_type] = counts
    
    return analysis


def get_class_weights(root_dir, split='split0', video_type='video_train'):
    """Calculate class weights for handling imbalanced dataset."""
    from configs.datasets.drive_act import ACTION_CLASSES
    
    # Count samples per class (using RGB modality as reference)
    class_counts = []
    
    for class_name in ACTION_CLASSES:
        class_path = os.path.join(root_dir, split, video_type, 'rgb', class_name)
        if os.path.exists(class_path):
            video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
            class_counts.append(len(video_files))
        else:
            class_counts.append(0)
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts)
    num_classes = len(ACTION_CLASSES)
    
    if total_samples == 0:
        return torch.ones(num_classes)
    
    weights = []
    for count in class_counts:
        if count == 0:
            weights.append(0.0)  # No samples for this class
        else:
            weights.append(total_samples / (num_classes * count))
    
    return torch.FloatTensor(weights)


def save_dataset_statistics(root_dir, output_file='dataset_stats.json'):
    """Save dataset statistics to a JSON file."""
    stats = {
        'structure_check': check_dataset_structure(root_dir),
        'balance_analysis': analyze_dataset_balance(root_dir)
    }
    
    # Calculate total videos
    stats['total_videos'] = {}
    for split in ['split0', 'split1', 'split2']:
        stats['total_videos'][split] = {}
        for video_type in ['video_train', 'video_val', 'video_test']:
            total_rgb = sum(stats['balance_analysis'][split][video_type]['rgb'].values())
            total_kir = sum(stats['balance_analysis'][split][video_type]['kir'].values())
            stats['total_videos'][split][video_type] = {
                'rgb': total_rgb,
                'kir': total_kir
            }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def create_class_mapping(output_file='class_mapping.json'):
    """Create and save class name to index mapping."""
    from configs.datasets.drive_act import ACTION_CLASSES
    
    class_mapping = {
        'idx_to_class': {idx: cls for idx, cls in enumerate(ACTION_CLASSES)},
        'class_to_idx': {cls: idx for idx, cls in enumerate(ACTION_CLASSES)},
        'num_classes': len(ACTION_CLASSES)
    }
    
    with open(output_file, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    return class_mapping


def calculate_dataset_mean_std(dataloader, modality='fused'):
    """Calculate mean and standard deviation for the dataset."""
    mean = torch.zeros(3 if modality == 'rgb' else (1 if modality == 'kir' else 4))
    std = torch.zeros(3 if modality == 'rgb' else (1 if modality == 'kir' else 4))
    total_samples = 0
    
    for batch in dataloader:
        video_data = batch['video'][modality]  # [B, T, C, H, W]
        batch_samples = video_data.size(0) * video_data.size(1)  # B * T
        
        # Reshape to [B*T*H*W, C]
        video_data = video_data.permute(0, 1, 3, 4, 2).contiguous()
        video_data = video_data.view(-1, video_data.size(-1))
        
        mean += video_data.mean(dim=0) * batch_samples
        std += video_data.std(dim=0) * batch_samples
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()


def verify_data_alignment(root_dir, split='split0', video_type='video_train', max_samples=10):
    """Verify that RGB and KIR videos are properly aligned."""
    from configs.datasets.drive_act import ACTION_CLASSES
    import cv2
    
    misaligned_samples = []
    
    for class_name in ACTION_CLASSES[:max_samples]:  # Check first few classes
        rgb_path = os.path.join(root_dir, split, video_type, 'rgb', class_name)
        kir_path = os.path.join(root_dir, split, video_type, 'kir', class_name)
        
        if os.path.exists(rgb_path) and os.path.exists(kir_path):
            rgb_files = set([f.replace('.mp4', '') for f in os.listdir(rgb_path) if f.endswith('.mp4')])
            kir_files = set([f.replace('.mp4', '') for f in os.listdir(kir_path) if f.endswith('.mp4')])
            
            # Check for missing correspondences
            rgb_only = rgb_files - kir_files
            kir_only = kir_files - rgb_files
            
            if rgb_only or kir_only:
                misaligned_samples.append({
                    'class': class_name,
                    'rgb_only': list(rgb_only),
                    'kir_only': list(kir_only),
                    'common': len(rgb_files & kir_files)
                })
    
    return misaligned_samples


def split_data_by_ratio(data_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data list into train, validation, and test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.shuffle(data_list)
    total_samples = len(data_list)
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_data = data_list[:train_size]
    val_data = data_list[train_size:train_size + val_size]
    test_data = data_list[train_size + val_size:]
    
    return train_data, val_data, test_data