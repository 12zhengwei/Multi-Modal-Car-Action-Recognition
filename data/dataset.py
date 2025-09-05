"""
Drive and Act Dataset Implementation
Multi-modal dataset loader for car action recognition using RGB and thermal infrared data.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from configs.datasets.drive_act import ACTION_CLASSES, DATASET_CONFIG


class DriveActDataset(Dataset):
    """Drive and Act multi-modal dataset for car action recognition."""
    
    def __init__(self, root_dir, split='split0', video_type='video_train', 
                 transform=None, num_frames=16, sampling_rate=4, modalities=['rgb', 'kir']):
        """
        Args:
            root_dir (str): Root directory of the dataset
            split (str): Data split (split0, split1, split2)
            video_type (str): Type of videos (video_train, video_val, video_test)
            transform: Transform to apply to the videos
            num_frames (int): Number of frames to sample from each video
            sampling_rate (int): Frame sampling rate
            modalities (list): List of modalities to load ['rgb', 'kir']
        """
        self.root_dir = root_dir
        self.split = split
        self.video_type = video_type
        self.transform = transform
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.modalities = modalities
        
        self.action_classes = ACTION_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.action_classes)}
        
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load all video samples from the dataset."""
        samples = []
        
        for class_name in self.action_classes:
            class_samples = {}
            
            # Load samples for each modality
            for modality in self.modalities:
                modality_path = os.path.join(
                    self.root_dir, self.split, self.video_type, modality, class_name
                )
                
                if os.path.exists(modality_path):
                    video_files = glob.glob(os.path.join(modality_path, '*.mp4'))
                    for video_file in video_files:
                        video_id = os.path.splitext(os.path.basename(video_file))[0]
                        if video_id not in class_samples:
                            class_samples[video_id] = {
                                'class_name': class_name,
                                'class_idx': self.class_to_idx[class_name],
                                'modalities': {}
                            }
                        class_samples[video_id]['modalities'][modality] = video_file
            
            # Only keep samples that have all required modalities
            for video_id, sample in class_samples.items():
                if all(mod in sample['modalities'] for mod in self.modalities):
                    samples.append(sample)
        
        return samples
    
    def _load_video(self, video_path, modality='rgb'):
        """Load video frames from file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if modality == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:  # kir (thermal)
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _sample_frames(self, frames, num_frames, sampling_rate):
        """Sample frames from video."""
        total_frames = len(frames)
        
        if total_frames <= num_frames * sampling_rate:
            # If video is too short, repeat frames
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        else:
            # Sample frames with specified sampling rate
            start_idx = np.random.randint(0, total_frames - num_frames * sampling_rate + 1)
            indices = np.arange(start_idx, start_idx + num_frames * sampling_rate, sampling_rate)
        
        return [frames[i] for i in indices]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_data = {}
        
        # Load each modality
        for modality in self.modalities:
            video_path = sample['modalities'][modality]
            frames = self._load_video(video_path, modality)
            frames = self._sample_frames(frames, self.num_frames, self.sampling_rate)
            
            # Convert to tensor
            if modality == 'rgb':
                video_tensor = torch.stack([
                    torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    for frame in frames
                ])  # Shape: [T, C, H, W]
            else:  # kir
                video_tensor = torch.stack([
                    torch.from_numpy(frame).unsqueeze(0).float() / 255.0
                    for frame in frames
                ])  # Shape: [T, 1, H, W]
            
            video_data[modality] = video_tensor
        
        # Apply transforms if provided
        if self.transform:
            video_data = self.transform(video_data)
        
        # For early fusion, concatenate channels
        if len(self.modalities) > 1 and 'rgb' in video_data and 'kir' in video_data:
            # Concatenate RGB and KIR channels: [T, C_rgb + C_kir, H, W]
            video_data['fused'] = torch.cat([video_data['rgb'], video_data['kir']], dim=1)
        
        return {
            'video': video_data,
            'label': sample['class_idx'],
            'class_name': sample['class_name']
        }


def create_dataloader(root_dir, split='split0', video_type='video_train', 
                     batch_size=8, num_workers=4, shuffle=True, transform=None):
    """Create DataLoader for Drive and Act dataset."""
    
    dataset = DriveActDataset(
        root_dir=root_dir,
        split=split,
        video_type=video_type,
        transform=transform,
        num_frames=DATASET_CONFIG['num_frames'],
        sampling_rate=DATASET_CONFIG['sampling_rate'],
        modalities=DATASET_CONFIG['modalities']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(video_type == 'video_train')
    )
    
    return dataloader


def get_dataloaders(root_dir, batch_size=8, num_workers=4, transform=None):
    """Get train, validation, and test dataloaders."""
    
    dataloaders = {}
    
    for split in ['split0']:  # Can extend to use all splits
        for video_type, shuffle in [('video_train', True), ('video_val', False), ('video_test', False)]:
            key = video_type.replace('video_', '')
            dataloaders[key] = create_dataloader(
                root_dir=root_dir,
                split=split,
                video_type=video_type,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                transform=transform
            )
    
    return dataloaders