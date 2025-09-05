"""
Inference Script for Multi-Modal Action Recognition
Run inference on single video or batch of videos.
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import json
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recognizer import load_pretrained_recognizer
from data.transforms import get_test_transforms
from configs.datasets.drive_act import DATASET_CONFIG, ACTION_CLASSES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on video(s)')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video file or directory')
    parser.add_argument('--rgb_video', type=str, default=None,
                       help='Path to RGB video (if separate from thermal)')
    parser.add_argument('--thermal_video', type=str, default=None,
                       help='Path to thermal video (if separate from RGB)')
    parser.add_argument('--fusion_type', type=str, default='early',
                       choices=['early', 'late', 'meta'],
                       help='Fusion strategy used in the model')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--output', type=str, default='inference_results.json',
                       help='Output file for predictions')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to return')
    parser.add_argument('--batch_process', action='store_true',
                       help='Process all videos in input directory')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save sample frames with predictions')
    
    return parser.parse_args()


def setup_device(device_str):
    """Setup compute device."""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    return device


def load_model(checkpoint_path, fusion_type, device):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Model configuration
    backbone_config = {
        'img_size': DATASET_CONFIG['img_size'],
        'patch_size': 16,
        'in_chans': 4 if fusion_type == 'early' else 3,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'tubelet_size': 1
    }
    
    fusion_config = {'fusion_type': 'simple'}
    head_config = {'head_type': 'linear'}
    
    model = load_pretrained_recognizer(
        checkpoint_path,
        fusion_type=fusion_type,
        num_classes=len(ACTION_CLASSES),
        backbone_config=backbone_config,
        fusion_config=fusion_config,
        head_config=head_config
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    return model


def load_video_frames(video_path, num_frames=16, img_size=224):
    """Load and preprocess video frames."""
    cap = cv2.VideoCapture(str(video_path))
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < num_frames:
        print(f"Warning: Video has only {frame_count} frames, requested {num_frames}")
    
    # Calculate frame indices for uniform sampling
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames).astype(int)
    else:
        # If video is shorter, repeat frames
        indices = np.linspace(0, frame_count - 1, num_frames).astype(int)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {i}")
            if frames:  # Use last frame if available
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            continue
        
        # Resize frame
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) != num_frames:
        print(f"Warning: Expected {num_frames} frames, got {len(frames)}")
        # Pad or truncate as needed
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3), dtype=np.uint8))
        frames = frames[:num_frames]
    
    return np.array(frames)


def preprocess_video(frames, modality='rgb', transform=None):
    """Preprocess video frames for model input."""
    # Convert to torch tensor: [T, H, W, C] -> [C, T, H, W]
    video_tensor = torch.from_numpy(frames).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
    
    # Handle different modalities
    if modality == 'kir' and video_tensor.shape[0] == 3:
        # Convert RGB to grayscale for thermal simulation
        video_tensor = torch.mean(video_tensor, dim=0, keepdim=True)
    
    # Apply transforms if provided
    if transform:
        video_data = {modality: video_tensor.unsqueeze(0)}  # Add batch dimension
        video_data = transform(video_data)
        video_tensor = video_data[modality].squeeze(0)  # Remove batch dimension
    
    return video_tensor


def predict_single_video(model, video_data, device, top_k=5):
    """Run prediction on a single video."""
    model.eval()
    
    with torch.no_grad():
        # Move to device
        if isinstance(video_data, dict):
            video_input = {k: v.unsqueeze(0).to(device) for k, v in video_data.items()}
        else:
            video_input = video_data.unsqueeze(0).to(device)
        
        # Forward pass
        start_time = time.time()
        outputs = model(video_input)
        inference_time = time.time() - start_time
        
        # Get probabilities and predictions
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Convert to numpy
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Create results
        predictions = []
        for i in range(top_k):
            predictions.append({
                'class_index': int(top_indices[i]),
                'class_name': ACTION_CLASSES[top_indices[i]],
                'confidence': float(top_probs[i])
            })
        
        result = {
            'predictions': predictions,
            'inference_time': inference_time,
            'top_prediction': predictions[0]
        }
        
        return result


def process_single_video_file(video_path, model, device, fusion_type, transform, args):
    """Process a single video file."""
    print(f"Processing: {video_path}")
    
    try:
        # Load video frames
        frames = load_video_frames(
            video_path, 
            num_frames=DATASET_CONFIG['num_frames'],
            img_size=DATASET_CONFIG['img_size']
        )
        
        if fusion_type == 'early':
            # For early fusion, we need both RGB and thermal
            # If only one video provided, simulate the other modality
            rgb_tensor = preprocess_video(frames, 'rgb', transform)
            
            if args.thermal_video:
                thermal_frames = load_video_frames(
                    args.thermal_video,
                    num_frames=DATASET_CONFIG['num_frames'],
                    img_size=DATASET_CONFIG['img_size']
                )
                thermal_tensor = preprocess_video(thermal_frames, 'kir', transform)
            else:
                # Simulate thermal by converting RGB to grayscale
                thermal_tensor = torch.mean(rgb_tensor, dim=0, keepdim=True)
            
            # Create fused input
            video_data = {'fused': torch.cat([rgb_tensor, thermal_tensor], dim=0)}
            
        else:  # late or meta fusion
            # Separate modalities
            rgb_tensor = preprocess_video(frames, 'rgb', transform)
            
            if args.thermal_video:
                thermal_frames = load_video_frames(
                    args.thermal_video,
                    num_frames=DATASET_CONFIG['num_frames'],
                    img_size=DATASET_CONFIG['img_size']
                )
                thermal_tensor = preprocess_video(thermal_frames, 'kir', transform)
            else:
                # Simulate thermal
                thermal_tensor = torch.mean(rgb_tensor, dim=0, keepdim=True)
            
            video_data = {
                'rgb': rgb_tensor,
                'kir': thermal_tensor
            }
        
        # Run prediction
        result = predict_single_video(model, video_data, device, args.top_k)
        
        # Add metadata
        result.update({
            'video_path': str(video_path),
            'video_name': video_path.name,
            'fusion_type': fusion_type,
            'num_frames': len(frames)
        })
        
        # Filter by confidence threshold
        filtered_predictions = [
            pred for pred in result['predictions'] 
            if pred['confidence'] >= args.confidence_threshold
        ]
        result['filtered_predictions'] = filtered_predictions
        
        print(f"  Top prediction: {result['top_prediction']['class_name']} "
              f"({result['top_prediction']['confidence']:.3f})")
        print(f"  Inference time: {result['inference_time']:.3f}s")
        
        return result
        
    except Exception as e:
        print(f"  Error processing {video_path}: {e}")
        return {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'error': str(e),
            'predictions': []
        }


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model
    model = load_model(args.checkpoint, args.fusion_type, device)
    
    # Create transforms
    transform = get_test_transforms(img_size=DATASET_CONFIG['img_size'])
    
    # Process video(s)
    results = []
    input_path = Path(args.input)
    
    if args.batch_process and input_path.is_dir():
        # Process all videos in directory
        print(f"Processing all videos in: {input_path}")
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(input_path.glob(f'*{ext}'))
            video_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(video_files)} video files")
        
        for video_file in video_files:
            result = process_single_video_file(
                video_file, model, device, args.fusion_type, transform, args
            )
            results.append(result)
            
    elif input_path.is_file():
        # Process single video
        result = process_single_video_file(
            input_path, model, device, args.fusion_type, transform, args
        )
        results.append(result)
        
    else:
        print(f"Error: Input path {input_path} not found or invalid")
        return
    
    # Compile summary statistics
    total_videos = len(results)
    successful_predictions = len([r for r in results if 'predictions' in r and r['predictions']])
    
    if successful_predictions > 0:
        avg_inference_time = np.mean([r['inference_time'] for r in results if 'inference_time' in r])
        
        # Count predictions by class
        class_counts = {}
        for result in results:
            if 'top_prediction' in result:
                class_name = result['top_prediction']['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Save results
    output_data = {
        'summary': {
            'total_videos': total_videos,
            'successful_predictions': successful_predictions,
            'fusion_type': args.fusion_type,
            'confidence_threshold': args.confidence_threshold,
            'avg_inference_time': avg_inference_time if successful_predictions > 0 else 0
        },
        'results': results
    }
    
    if successful_predictions > 0:
        output_data['summary']['class_distribution'] = class_counts
        output_data['summary']['most_common_prediction'] = max(class_counts, key=class_counts.get)
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print(f"Total videos processed: {total_videos}")
    print(f"Successful predictions: {successful_predictions}")
    
    if successful_predictions > 0:
        print(f"Average inference time: {avg_inference_time:.3f}s")
        print(f"Most common prediction: {output_data['summary']['most_common_prediction']}")
        
        print(f"\nTop 5 predicted classes:")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes[:5]:
            print(f"  {class_name}: {count} videos")
    
    print(f"\nResults saved to: {args.output}")
    print("="*50)


if __name__ == '__main__':
    main()