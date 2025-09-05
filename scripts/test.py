"""
Testing Script for Multi-Modal Action Recognition
Evaluate the trained model on test set and generate detailed analysis.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recognizer import load_pretrained_recognizer
from data.dataset import create_dataloader
from data.transforms import get_test_transforms
from utils.logger import setup_logger
from utils.metrics import MetricsCalculator
from utils.visualization import visualize_multimodal_sample, plot_model_predictions
from configs.datasets.drive_act import DATASET_CONFIG, ACTION_CLASSES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test multi-modal action recognition model')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--fusion_type', type=str, default='early',
                       choices=['early', 'late', 'meta'],
                       help='Fusion strategy used in the model')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Directory to save test results')
    parser.add_argument('--split', type=str, default='split0',
                       choices=['split0', 'split1', 'split2'],
                       help='Data split to use for testing')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save individual predictions to file')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--num_vis_samples', type=int, default=20,
                       help='Number of samples to visualize')
    
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Model configuration (should match training)
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
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total_params:,} parameters")
    
    if 'epoch' in checkpoint:
        print(f"Model was trained for {checkpoint['epoch']} epochs")
    
    return model


def test_model(model, test_loader, device, metrics_calc, save_predictions=False):
    """Test the model and collect predictions."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_samples_info = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')  # Get per-sample losses
    total_loss = 0.0
    sample_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            if isinstance(batch['video'], dict):
                video_data = {k: v.to(device) for k, v in batch['video'].items()}
            else:
                video_data = batch['video'].to(device)
            
            targets = batch['label'].to(device)
            
            # Forward pass
            outputs = model(video_data)
            batch_losses = criterion(outputs, targets)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            # Collect results
            batch_predictions = predictions.cpu().numpy()
            batch_targets = targets.cpu().numpy()
            batch_probabilities = probabilities.cpu().numpy()
            batch_losses_np = batch_losses.cpu().numpy()
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(batch_targets)
            all_probabilities.extend(batch_probabilities)
            sample_losses.extend(batch_losses_np)
            
            # Update metrics
            metrics_calc.update(predictions, targets, probabilities)
            
            total_loss += batch_losses.sum().item()
            
            # Save sample info if requested
            if save_predictions:
                for i in range(len(batch_targets)):
                    sample_info = {
                        'prediction': int(batch_predictions[i]),
                        'target': int(batch_targets[i]),
                        'predicted_class': ACTION_CLASSES[batch_predictions[i]],
                        'target_class': ACTION_CLASSES[batch_targets[i]],
                        'confidence': float(batch_probabilities[i].max()),
                        'loss': float(batch_losses_np[i]),
                        'correct': bool(batch_predictions[i] == batch_targets[i])
                    }
                    all_samples_info.append(sample_info)
            
            if batch_idx % 50 == 0:
                print(f'Processed batch {batch_idx}/{len(test_loader)}')
    
    avg_loss = total_loss / len(all_predictions)
    
    results = {
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probabilities),
        'sample_losses': np.array(sample_losses),
        'avg_loss': avg_loss,
        'samples_info': all_samples_info if save_predictions else None
    }
    
    return results


def analyze_results(results, metrics_calc, output_dir):
    """Analyze test results and save detailed analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("TEST RESULTS ANALYSIS")
    print("="*50)
    
    # Compute all metrics
    all_metrics = metrics_calc.compute_all_metrics()
    
    # Print main metrics
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {all_metrics['accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {all_metrics['top_5_accuracy']:.4f}")
    print(f"  Average Loss: {results['avg_loss']:.4f}")
    print(f"  Precision (macro): {all_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {all_metrics['recall_macro']:.4f}")
    print(f"  F1-score (macro): {all_metrics['f1_macro']:.4f}")
    
    if 'auc_macro' in all_metrics:
        print(f"  AUC (macro): {all_metrics['auc_macro']:.4f}")
    
    # Per-class analysis
    print(f"\nPer-Class Performance:")
    per_class_acc = metrics_calc.compute_per_class_accuracy()
    
    # Sort classes by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    
    print("  Best performing classes:")
    for class_name, acc in sorted_classes[:5]:
        print(f"    {class_name}: {acc:.4f}")
    
    print("  Worst performing classes:")
    for class_name, acc in sorted_classes[-5:]:
        print(f"    {class_name}: {acc:.4f}")
    
    # Error analysis
    print(f"\nError Analysis:")
    predictions = results['predictions']
    targets = results['targets']
    sample_losses = results['sample_losses']
    
    # Most confident correct predictions
    correct_mask = predictions == targets
    correct_confidences = results['probabilities'][correct_mask].max(axis=1)
    most_confident_correct = correct_confidences.argsort()[-5:][::-1]
    
    print("  Most confident correct predictions:")
    for idx in most_confident_correct:
        global_idx = np.where(correct_mask)[0][idx]
        pred_class = ACTION_CLASSES[predictions[global_idx]]
        conf = correct_confidences[idx]
        print(f"    {pred_class}: {conf:.4f}")
    
    # Most confident incorrect predictions
    incorrect_mask = predictions != targets
    if incorrect_mask.sum() > 0:
        incorrect_confidences = results['probabilities'][incorrect_mask].max(axis=1)
        most_confident_incorrect = incorrect_confidences.argsort()[-5:][::-1]
        
        print("  Most confident incorrect predictions:")
        for idx in most_confident_incorrect:
            global_idx = np.where(incorrect_mask)[0][idx]
            pred_class = ACTION_CLASSES[predictions[global_idx]]
            true_class = ACTION_CLASSES[targets[global_idx]]
            conf = incorrect_confidences[idx]
            print(f"    Predicted: {pred_class}, True: {true_class}, Confidence: {conf:.4f}")
    
    # Save detailed metrics
    metrics_file = output_dir / 'test_metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for key, value in all_metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                json_metrics[key] = value.item()
            else:
                json_metrics[key] = value
        
        json_metrics['avg_loss'] = results['avg_loss']
        json_metrics['per_class_accuracy'] = per_class_acc
        
        json.dump(json_metrics, f, indent=2)
    
    print(f"\nDetailed metrics saved to: {metrics_file}")
    
    # Save classification report
    classification_report = metrics_calc.compute_classification_report(output_dict=False)
    report_file = output_dir / 'classification_report.txt'
    with open(report_file, 'w') as f:
        f.write(classification_report)
    
    print(f"Classification report saved to: {report_file}")
    
    return all_metrics


def create_visualizations(model, test_loader, results, output_dir, num_samples=20):
    """Create and save visualizations."""
    output_dir = Path(output_dir)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating visualizations...")
    
    # Plot confusion matrix
    from utils.metrics import MetricsCalculator
    metrics_calc = MetricsCalculator(len(ACTION_CLASSES), ACTION_CLASSES)
    metrics_calc.all_predictions = results['predictions'].tolist()
    metrics_calc.all_targets = results['targets'].tolist()
    metrics_calc.all_probabilities = results['probabilities'].tolist()
    
    # Normalized confusion matrix
    cm_fig = metrics_calc.plot_confusion_matrix(
        normalize='true',
        save_path=vis_dir / 'confusion_matrix_normalized.png',
        title='Normalized Confusion Matrix'
    )
    
    # Raw confusion matrix
    cm_raw_fig = metrics_calc.plot_confusion_matrix(
        normalize=None,
        save_path=vis_dir / 'confusion_matrix_raw.png',
        title='Confusion Matrix (Raw Counts)'
    )
    
    # Per-class metrics
    per_class_fig = metrics_calc.plot_per_class_metrics(
        save_path=vis_dir / 'per_class_metrics.png'
    )
    
    # Model predictions summary
    pred_fig = plot_model_predictions(
        results['predictions'], results['probabilities'], results['targets'],
        class_names=ACTION_CLASSES, save_path=vis_dir / 'predictions_summary.png'
    )
    
    print(f"Visualizations saved to: {vis_dir}")


def main():
    """Main testing function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting multi-modal action recognition testing")
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model(args.checkpoint, args.fusion_type, device)
    
    # Create test data loader
    print("Creating test data loader...")
    
    test_transform = get_test_transforms(img_size=DATASET_CONFIG['img_size'])
    
    test_loader = create_dataloader(
        root_dir=args.data_root,
        split=args.split,
        video_type='video_test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        transform=test_transform
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Setup metrics calculator
    metrics_calc = MetricsCalculator(len(ACTION_CLASSES), ACTION_CLASSES)
    
    # Test the model
    print("Testing model...")
    results = test_model(
        model, test_loader, device, metrics_calc, 
        save_predictions=args.save_predictions
    )
    
    # Analyze results
    all_metrics = analyze_results(results, metrics_calc, output_dir)
    
    # Save predictions if requested
    if args.save_predictions and results['samples_info']:
        predictions_file = output_dir / 'predictions.json'
        with open(predictions_file, 'w') as f:
            json.dump(results['samples_info'], f, indent=2)
        print(f"Individual predictions saved to: {predictions_file}")
    
    # Create visualizations if requested
    if args.save_visualizations:
        create_visualizations(model, test_loader, results, output_dir, args.num_vis_samples)
    
    print("\n" + "="*50)
    print("TESTING COMPLETED")
    print(f"Final Accuracy: {all_metrics['accuracy']:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()