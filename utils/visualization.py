"""
Visualization Utilities
Visualization tools for multi-modal action recognition.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import io
import base64
from pathlib import Path
from configs.datasets.drive_act import ACTION_CLASSES


def denormalize_video(video_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize video tensor for visualization."""
    if len(video_tensor.shape) == 5:  # [B, C, T, H, W]
        video_tensor = video_tensor[0]  # Take first batch
    
    if len(video_tensor.shape) == 4:  # [C, T, H, W]
        C, T, H, W = video_tensor.shape
        
        # Handle different channel configurations
        if C == 3:  # RGB
            mean = torch.tensor(mean).view(3, 1, 1, 1)
            std = torch.tensor(std).view(3, 1, 1, 1)
        elif C == 1:  # Grayscale/Thermal
            mean = torch.tensor([0.5]).view(1, 1, 1, 1)
            std = torch.tensor([0.5]).view(1, 1, 1, 1)
        elif C == 4:  # RGB + Thermal
            mean = torch.tensor(mean + [0.5]).view(4, 1, 1, 1)
            std = torch.tensor(std + [0.5]).view(4, 1, 1, 1)
        
        video_tensor = video_tensor * std + mean
        video_tensor = torch.clamp(video_tensor, 0, 1)
    
    return video_tensor


def create_video_grid(video_tensor, nrow=4, padding=2):
    """Create a grid of video frames."""
    if len(video_tensor.shape) == 4:  # [C, T, H, W]
        C, T, H, W = video_tensor.shape
        
        # Take subset of frames if too many
        if T > 16:
            indices = np.linspace(0, T-1, 16).astype(int)
            video_tensor = video_tensor[:, indices]
            T = 16
        
        # Rearrange to create grid: [T, C, H, W]
        frames = video_tensor.permute(1, 0, 2, 3)
        
        # Create grid
        from torchvision.utils import make_grid
        grid = make_grid(frames, nrow=nrow, padding=padding, normalize=False)
        
        return grid.permute(1, 2, 0).numpy()
    
    return None


def visualize_multimodal_sample(rgb_video, thermal_video, prediction, target, 
                               class_names=None, save_path=None, figsize=(15, 8)):
    """Visualize multi-modal video sample with prediction."""
    class_names = class_names or ACTION_CLASSES
    
    # Denormalize videos
    rgb_viz = denormalize_video(rgb_video)
    thermal_viz = denormalize_video(thermal_video)
    
    # Create grids
    rgb_grid = create_video_grid(rgb_viz)
    thermal_grid = create_video_grid(thermal_viz)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot RGB frames
    axes[0].imshow(rgb_grid)
    axes[0].set_title('RGB Video Frames', fontsize=14)
    axes[0].axis('off')
    
    # Plot thermal frames
    if thermal_grid.shape[-1] == 1:  # Grayscale thermal
        axes[1].imshow(thermal_grid.squeeze(-1), cmap='hot')
    else:
        axes[1].imshow(thermal_grid)
    axes[1].set_title('Thermal Video Frames', fontsize=14)
    axes[1].axis('off')
    
    # Add prediction information
    pred_class = class_names[prediction] if prediction < len(class_names) else f"Class_{prediction}"
    target_class = class_names[target] if target < len(class_names) else f"Class_{target}"
    
    fig.suptitle(f'Prediction: {pred_class} | Ground Truth: {target_class}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                        save_path=None, figsize=(12, 5)):
    """Plot training and validation curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_class_distribution(dataset, class_names=None, save_path=None, figsize=(15, 6)):
    """Plot class distribution in dataset."""
    class_names = class_names or ACTION_CLASSES
    
    # Count samples per class
    class_counts = {}
    for sample in dataset.samples:
        class_name = sample['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_classes)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(classes)), counts, alpha=0.7)
    
    # Color bars by count (gradient)
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.title('Class Distribution in Dataset', fontsize=16)
    plt.xlabel('Action Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count), 
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def visualize_attention_weights(attention_weights, input_video, save_path=None, figsize=(12, 8)):
    """Visualize attention weights on video frames."""
    # This is a placeholder for attention visualization
    # Actual implementation would depend on the attention mechanism used
    
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    
    # Plot sample frames with attention weights
    for i in range(8):
        row, col = i // 4, i % 4
        
        if i < len(attention_weights):
            # Overlay attention on frame
            axes[row, col].imshow(attention_weights[i], cmap='hot', alpha=0.7)
            axes[row, col].set_title(f'Attention Frame {i+1}')
        else:
            axes[row, col].axis('off')
    
    plt.suptitle('Attention Weights Visualization', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_predictions(predictions, probabilities, targets, class_names=None, 
                          top_k=10, save_path=None, figsize=(12, 8)):
    """Plot model predictions with confidence scores."""
    class_names = class_names or ACTION_CLASSES
    
    # Get top-k most confident predictions
    confidence_scores = np.max(probabilities, axis=1)
    top_indices = np.argsort(confidence_scores)[-top_k:][::-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot confidence scores
    confidences = confidence_scores[top_indices]
    pred_classes = [class_names[predictions[i]] for i in top_indices]
    true_classes = [class_names[targets[i]] for i in top_indices]
    
    colors = ['green' if predictions[i] == targets[i] else 'red' for i in top_indices]
    
    bars = ax1.barh(range(top_k), confidences, color=colors, alpha=0.7)
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels([f'{pred} | {true}' for pred, true in zip(pred_classes, true_classes)])
    ax1.set_xlabel('Confidence Score')
    ax1.set_title(f'Top {top_k} Most Confident Predictions')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot prediction distribution
    unique_preds, counts = np.unique(predictions, return_counts=True)
    pred_names = [class_names[i] for i in unique_preds]
    
    ax2.pie(counts, labels=pred_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Prediction Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_model_summary_plot(model_metrics, save_path=None, figsize=(10, 6)):
    """Create a summary plot of model metrics."""
    metrics_names = list(model_metrics.keys())
    metrics_values = list(model_metrics.values())
    
    # Filter out non-numeric metrics
    numeric_metrics = {}
    for name, value in model_metrics.items():
        if isinstance(value, (int, float)):
            numeric_metrics[name] = value
    
    if not numeric_metrics:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(numeric_metrics.keys(), numeric_metrics.values(), 
                 color=plt.cm.Set3(np.linspace(0, 1, len(numeric_metrics))), alpha=0.8)
    
    ax.set_title('Model Performance Metrics', fontsize=16)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_video_frames(video_tensor, save_dir, prefix='frame', max_frames=16):
    """Save video frames as individual images."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize video
    video_viz = denormalize_video(video_tensor)
    
    if len(video_viz.shape) == 4:  # [C, T, H, W]
        C, T, H, W = video_viz.shape
        
        # Limit number of frames
        num_frames = min(T, max_frames)
        
        for t in range(num_frames):
            frame = video_viz[:, t, :, :]  # [C, H, W]
            
            # Convert to PIL Image
            if C == 3:  # RGB
                frame_img = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_img)
            elif C == 1:  # Grayscale
                frame_img = (frame.squeeze(0).numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_img, mode='L')
            else:  # Multi-channel, take first 3
                frame_img = (frame[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_img)
            
            # Save image
            save_path = save_dir / f'{prefix}_{t:03d}.png'
            pil_img.save(save_path)
    
    return save_dir


def plot_feature_distributions(features_dict, save_path=None, figsize=(15, 10)):
    """Plot feature distributions for different modalities."""
    num_modalities = len(features_dict)
    
    fig, axes = plt.subplots(2, num_modalities, figsize=figsize)
    if num_modalities == 1:
        axes = axes.reshape(2, 1)
    
    for i, (modality, features) in enumerate(features_dict.items()):
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # Feature magnitude distribution
        feature_magnitudes = np.linalg.norm(features, axis=1)
        axes[0, i].hist(feature_magnitudes, bins=50, alpha=0.7, color=f'C{i}')
        axes[0, i].set_title(f'{modality.capitalize()} Feature Magnitudes')
        axes[0, i].set_xlabel('Magnitude')
        axes[0, i].set_ylabel('Count')
        axes[0, i].grid(True, alpha=0.3)
        
        # Feature component distribution (sample)
        feature_sample = features[:, :100] if features.shape[1] > 100 else features
        axes[1, i].boxplot(feature_sample.T)
        axes[1, i].set_title(f'{modality.capitalize()} Feature Components')
        axes[1, i].set_xlabel('Feature Dimension (sample)')
        axes[1, i].set_ylabel('Value')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions Across Modalities', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig