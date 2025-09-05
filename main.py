"""
Multi-Modal Car Action Recognition
Main entry point for training, testing, and inference.
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.train import main as train_main
from scripts.test import main as test_main
from scripts.inference import main as inference_main


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description='Multi-Modal Car Action Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python main.py train --data_root /path/to/data --fusion_type early --epochs 100

  # Testing
  python main.py test --data_root /path/to/data --checkpoint checkpoints/best_model.pth

  # Inference on single video
  python main.py inference --checkpoint checkpoints/best_model.pth --input video.mp4

  # Batch inference
  python main.py inference --checkpoint checkpoints/best_model.pth --input /path/to/videos --batch_process
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training subcommand
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_root', type=str, required=True,
                             help='Root directory of the dataset')
    train_parser.add_argument('--fusion_type', type=str, default='early',
                             choices=['early', 'late', 'meta'],
                             help='Fusion strategy to use')
    train_parser.add_argument('--batch_size', type=int, default=8,
                             help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=1e-4,
                             help='Weight decay')
    train_parser.add_argument('--num_workers', type=int, default=4,
                             help='Number of data loader workers')
    train_parser.add_argument('--device', type=str, default='auto',
                             help='Device to use (cuda, cpu, or auto)')
    train_parser.add_argument('--log_dir', type=str, default='logs',
                             help='Directory for logs and checkpoints')
    train_parser.add_argument('--experiment_name', type=str, default=None,
                             help='Name of the experiment')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    train_parser.add_argument('--save_freq', type=int, default=10,
                             help='Save checkpoint every N epochs')
    train_parser.add_argument('--eval_freq', type=int, default=1,
                             help='Evaluate every N epochs')
    train_parser.add_argument('--early_stop_patience', type=int, default=15,
                             help='Early stopping patience')
    train_parser.add_argument('--mixed_precision', action='store_true',
                             help='Use mixed precision training')
    train_parser.add_argument('--config', type=str, default=None,
                             help='Path to configuration file')
    
    # Testing subcommand
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--data_root', type=str, required=True,
                            help='Root directory of the dataset')
    test_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    test_parser.add_argument('--fusion_type', type=str, default='early',
                            choices=['early', 'late', 'meta'],
                            help='Fusion strategy used in the model')
    test_parser.add_argument('--batch_size', type=int, default=8,
                            help='Batch size for testing')
    test_parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of data loader workers')
    test_parser.add_argument('--device', type=str, default='auto',
                            help='Device to use (cuda, cpu, or auto)')
    test_parser.add_argument('--output_dir', type=str, default='test_results',
                            help='Directory to save test results')
    test_parser.add_argument('--split', type=str, default='split0',
                            choices=['split0', 'split1', 'split2'],
                            help='Data split to use for testing')
    test_parser.add_argument('--save_predictions', action='store_true',
                            help='Save individual predictions to file')
    test_parser.add_argument('--save_visualizations', action='store_true',
                            help='Save visualization plots')
    test_parser.add_argument('--num_vis_samples', type=int, default=20,
                            help='Number of samples to visualize')
    
    # Inference subcommand
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--checkpoint', type=str, required=True,
                                 help='Path to model checkpoint')
    inference_parser.add_argument('--input', type=str, required=True,
                                 help='Path to input video file or directory')
    inference_parser.add_argument('--rgb_video', type=str, default=None,
                                 help='Path to RGB video (if separate from thermal)')
    inference_parser.add_argument('--thermal_video', type=str, default=None,
                                 help='Path to thermal video (if separate from RGB)')
    inference_parser.add_argument('--fusion_type', type=str, default='early',
                                 choices=['early', 'late', 'meta'],
                                 help='Fusion strategy used in the model')
    inference_parser.add_argument('--device', type=str, default='auto',
                                 help='Device to use (cuda, cpu, or auto)')
    inference_parser.add_argument('--output', type=str, default='inference_results.json',
                                 help='Output file for predictions')
    inference_parser.add_argument('--confidence_threshold', type=float, default=0.5,
                                 help='Confidence threshold for predictions')
    inference_parser.add_argument('--top_k', type=int, default=5,
                                 help='Number of top predictions to return')
    inference_parser.add_argument('--batch_process', action='store_true',
                                 help='Process all videos in input directory')
    inference_parser.add_argument('--save_frames', action='store_true',
                                 help='Save sample frames with predictions')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Backup original sys.argv
    original_argv = sys.argv[:]
    
    try:
        # Modify sys.argv to match the expected format for each subcommand
        sys.argv = ['main.py'] + sys.argv[2:]  # Remove 'main.py' and the subcommand
        
        if args.command == 'train':
            print("Starting training...")
            train_main()
        elif args.command == 'test':
            print("Starting testing...")
            test_main()
        elif args.command == 'inference':
            print("Starting inference...")
            inference_main()
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == '__main__':
    main()