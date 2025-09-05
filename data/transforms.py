"""
Data Transforms for Multi-Modal Video Processing
Transforms for preprocessing RGB and thermal infrared video data.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
from configs.datasets.drive_act import DATASET_CONFIG


class VideoNormalize:
    """Normalize video frames for different modalities."""
    
    def __init__(self, modality='rgb'):
        self.modality = modality
        if modality == 'rgb':
            self.mean = torch.tensor(DATASET_CONFIG['mean']).view(1, 3, 1, 1)
            self.std = torch.tensor(DATASET_CONFIG['std']).view(1, 3, 1, 1)
        else:  # kir
            self.mean = torch.tensor(DATASET_CONFIG['kir_mean']).view(1, 1, 1, 1)
            self.std = torch.tensor(DATASET_CONFIG['kir_std']).view(1, 1, 1, 1)
    
    def __call__(self, video_tensor):
        """
        Args:
            video_tensor: Tensor of shape [T, C, H, W]
        Returns:
            Normalized tensor
        """
        return (video_tensor - self.mean) / self.std


class VideoResize:
    """Resize video frames to target size."""
    
    def __init__(self, size=(224, 224), interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, video_tensor):
        """
        Args:
            video_tensor: Tensor of shape [T, C, H, W]
        Returns:
            Resized tensor
        """
        T, C, H, W = video_tensor.shape
        video_tensor = video_tensor.view(-1, C, H, W)  # [T*C, H, W] -> [T, C, H, W]
        
        video_tensor = F.interpolate(
            video_tensor, 
            size=self.size, 
            mode=self.interpolation,
            align_corners=False if self.interpolation == 'bilinear' else None
        )
        
        return video_tensor.view(T, C, *self.size)


class VideoRandomCrop:
    """Random crop for video frames."""
    
    def __init__(self, size=(224, 224)):
        self.size = size
    
    def __call__(self, video_tensor):
        """
        Args:
            video_tensor: Tensor of shape [T, C, H, W]
        Returns:
            Randomly cropped tensor
        """
        T, C, H, W = video_tensor.shape
        crop_h, crop_w = self.size
        
        if H <= crop_h or W <= crop_w:
            return F.interpolate(
                video_tensor.view(-1, C, H, W), 
                size=self.size, 
                mode='bilinear',
                align_corners=False
            ).view(T, C, crop_h, crop_w)
        
        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)
        
        return video_tensor[:, :, top:top + crop_h, left:left + crop_w]


class VideoCenterCrop:
    """Center crop for video frames."""
    
    def __init__(self, size=(224, 224)):
        self.size = size
    
    def __call__(self, video_tensor):
        """
        Args:
            video_tensor: Tensor of shape [T, C, H, W]
        Returns:
            Center cropped tensor
        """
        T, C, H, W = video_tensor.shape
        crop_h, crop_w = self.size
        
        if H <= crop_h or W <= crop_w:
            return F.interpolate(
                video_tensor.view(-1, C, H, W), 
                size=self.size, 
                mode='bilinear',
                align_corners=False
            ).view(T, C, crop_h, crop_w)
        
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        
        return video_tensor[:, :, top:top + crop_h, left:left + crop_w]


class VideoHorizontalFlip:
    """Random horizontal flip for video frames."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, video_tensor):
        """
        Args:
            video_tensor: Tensor of shape [T, C, H, W]
        Returns:
            Horizontally flipped tensor (with probability p)
        """
        if np.random.random() < self.p:
            return torch.flip(video_tensor, [-1])  # Flip along width dimension
        return video_tensor


class MultiModalTransform:
    """Transform for multi-modal video data."""
    
    def __init__(self, mode='train', img_size=224):
        self.mode = mode
        self.img_size = img_size
        
        # Define transforms for each mode
        if mode == 'train':
            self.spatial_transforms = T.Compose([
                VideoResize((256, 256)),
                VideoRandomCrop((img_size, img_size)),
                VideoHorizontalFlip(0.5)
            ])
        else:
            self.spatial_transforms = T.Compose([
                VideoResize((img_size, img_size))
            ])
        
        # Normalization for different modalities
        self.rgb_normalize = VideoNormalize('rgb')
        self.kir_normalize = VideoNormalize('kir')
    
    def __call__(self, video_data):
        """
        Args:
            video_data: Dict containing video tensors for different modalities
        Returns:
            Transformed video data dict
        """
        transformed_data = {}
        
        for modality, video_tensor in video_data.items():
            if modality in ['rgb', 'kir']:
                # Apply spatial transforms
                video_tensor = self.spatial_transforms(video_tensor)
                
                # Apply modality-specific normalization
                if modality == 'rgb':
                    video_tensor = self.rgb_normalize(video_tensor)
                else:  # kir
                    video_tensor = self.kir_normalize(video_tensor)
                
                transformed_data[modality] = video_tensor
        
        # Create fused input for early fusion
        if 'rgb' in transformed_data and 'kir' in transformed_data:
            transformed_data['fused'] = torch.cat([
                transformed_data['rgb'], 
                transformed_data['kir']
            ], dim=1)  # Concatenate along channel dimension
        
        return transformed_data


class TemporalAugmentation:
    """Temporal augmentation for video sequences."""
    
    def __init__(self, temporal_jitter=0.1, frame_rate_jitter=0.1):
        self.temporal_jitter = temporal_jitter
        self.frame_rate_jitter = frame_rate_jitter
    
    def __call__(self, video_tensor):
        """
        Args:
            video_tensor: Tensor of shape [T, C, H, W]
        Returns:
            Temporally augmented tensor
        """
        T, C, H, W = video_tensor.shape
        
        # Temporal jittering
        if self.temporal_jitter > 0:
            jitter = int(T * self.temporal_jitter)
            if jitter > 0:
                start_offset = np.random.randint(-jitter, jitter + 1)
                start_idx = max(0, min(T - 1, start_offset))
                end_idx = min(T, start_idx + T)
                
                if end_idx - start_idx < T:
                    # Pad if necessary
                    padding = T - (end_idx - start_idx)
                    video_tensor = F.pad(video_tensor[start_idx:end_idx], 
                                       (0, 0, 0, 0, 0, 0, 0, padding), 
                                       mode='replicate')
                else:
                    video_tensor = video_tensor[start_idx:end_idx]
        
        return video_tensor


def get_transforms(mode='train', img_size=224):
    """Get transforms for the specified mode."""
    return MultiModalTransform(mode=mode, img_size=img_size)


def get_train_transforms(img_size=224):
    """Get training transforms with data augmentation."""
    return get_transforms('train', img_size)


def get_val_transforms(img_size=224):
    """Get validation transforms without data augmentation."""
    return get_transforms('val', img_size)


def get_test_transforms(img_size=224):
    """Get test transforms without data augmentation."""
    return get_transforms('test', img_size)