"""
Training Script for Multi-Modal Action Recognition
Train the multi-modal action recognition model on Drive and Act dataset.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recognizer import create_recognizer
from data.dataset import get_dataloaders
from data.transforms import get_train_transforms, get_val_transforms
from utils.logger import setup_logger, EarlyStopping
from utils.metrics import MetricsCalculator
from configs.datasets.drive_act import DATASET_CONFIG, TRAIN_CONFIG, ACTION_CLASSES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-modal action recognition model')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--fusion_type', type=str, default='early',
                       choices=['early', 'late', 'meta'],
                       help='Fusion strategy to use')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logs and checkpoints')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name of the experiment')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_str):
    """Setup compute device."""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device


def create_model(fusion_type, num_classes, device):
    """Create and initialize the model."""
    # Model configuration
    backbone_config = {
        'img_size': DATASET_CONFIG['img_size'],
        'patch_size': 16,
        'in_chans': 4 if fusion_type == 'early' else 3,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'tubelet_size': 1,
        'drop_path_rate': 0.1
    }
    
    fusion_config = {'fusion_type': 'simple'}
    head_config = {'head_type': 'linear', 'dropout': 0.1}
    
    model = create_recognizer(
        fusion_type=fusion_type,
        num_classes=num_classes,
        backbone_config=backbone_config,
        fusion_config=fusion_config,
        head_config=head_config
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    return model


def create_optimizer_and_scheduler(model, args):
    """Create optimizer and scheduler."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, 
               logger, metrics_calc, epoch, mixed_precision=False):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    batch_count = 0
    
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        if isinstance(batch['video'], dict):
            video_data = {k: v.to(device) for k, v in batch['video'].items()}
        else:
            video_data = batch['video'].to(device)
        
        targets = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(video_data)
                loss = criterion(outputs, targets)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(video_data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            predictions = outputs.argmax(dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            metrics_calc.update(predictions, targets, probabilities)
            
            running_loss += loss.item()
            batch_count += 1
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}'
                )
                logger.log_learning_rate(current_lr, logger.step_count)
        
        logger.increment_step()
    
    # Update scheduler
    scheduler.step()
    
    # Calculate epoch metrics
    avg_loss = running_loss / batch_count
    epoch_metrics = metrics_calc.compute_all_metrics()
    epoch_metrics['loss'] = avg_loss
    
    return epoch_metrics


def validate_epoch(model, val_loader, criterion, device, logger, metrics_calc, epoch):
    """Validate for one epoch."""
    model.eval()
    
    running_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move data to device
            if isinstance(batch['video'], dict):
                video_data = {k: v.to(device) for k, v in batch['video'].items()}
            else:
                video_data = batch['video'].to(device)
            
            targets = batch['label'].to(device)
            
            # Forward pass
            outputs = model(video_data)
            loss = criterion(outputs, targets)
            
            # Update metrics
            predictions = outputs.argmax(dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            metrics_calc.update(predictions, targets, probabilities)
            
            running_loss += loss.item()
            batch_count += 1
    
    # Calculate epoch metrics
    avg_loss = running_loss / batch_count
    epoch_metrics = metrics_calc.compute_all_metrics()
    epoch_metrics['loss'] = avg_loss
    
    return epoch_metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)
    
    # Setup device
    device = setup_device(args.device)
    
    # Setup logger
    logger = setup_logger(args.log_dir, args.experiment_name)
    logger.info("Starting multi-modal action recognition training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    train_transform = get_train_transforms(img_size=DATASET_CONFIG['img_size'])
    val_transform = get_val_transforms(img_size=DATASET_CONFIG['img_size'])
    
    dataloaders = get_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=train_transform  # Will be handled separately for train/val
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    num_classes = len(ACTION_CLASSES)
    model = create_model(args.fusion_type, num_classes, device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup metrics calculators
    train_metrics_calc = MetricsCalculator(num_classes, ACTION_CLASSES)
    val_metrics_calc = MetricsCalculator(num_classes, ACTION_CLASSES)
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=args.early_stop_patience, mode='min')
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training loop...")
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.increment_epoch()
        
        # Reset metrics
        train_metrics_calc.reset()
        val_metrics_calc.reset()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            logger, train_metrics_calc, epoch, args.mixed_precision
        )
        
        # Validate
        if (epoch + 1) % args.eval_freq == 0:
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, logger, val_metrics_calc, epoch
            )
            
            # Log metrics
            logger.log_metrics(train_metrics, epoch=epoch, prefix='train/')
            logger.log_metrics(val_metrics, epoch=epoch, prefix='val/')
            
            # Store for plotting
            train_losses.append(train_metrics['loss'])
            val_losses.append(val_metrics['loss'])
            train_accs.append(train_metrics['accuracy'])
            val_accs.append(val_metrics['accuracy'])
            
            # Check for best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0 or is_best:
                checkpoint_metrics = {
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                    'best_val_loss': best_val_loss
                }
                
                logger.save_checkpoint(
                    model, optimizer, scheduler, epoch, checkpoint_metrics, best=is_best
                )
            
            # Early stopping check
            if early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            # Log only training metrics
            logger.log_metrics(train_metrics, epoch=epoch, prefix='train/')
            train_losses.append(train_metrics['loss'])
            train_accs.append(train_metrics['accuracy'])
    
    # Save final metrics
    final_metrics = {
        'final_train_loss': train_losses[-1],
        'final_train_acc': train_accs[-1],
        'best_val_loss': best_val_loss
    }
    
    if val_losses:
        final_metrics.update({
            'final_val_loss': val_losses[-1],
            'final_val_acc': val_accs[-1]
        })
    
    # Log hyperparameters
    hparams = {
        'fusion_type': args.fusion_type,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs
    }
    
    logger.log_hyperparameters(hparams, final_metrics)
    
    # Save training curves
    if len(val_losses) > 0:
        from utils.visualization import plot_training_curves
        curves_path = logger.experiment_dir / 'training_curves.png'
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=curves_path)
        logger.info(f"Training curves saved to {curves_path}")
    
    logger.info("Training completed!")
    logger.close()


if __name__ == '__main__':
    main()