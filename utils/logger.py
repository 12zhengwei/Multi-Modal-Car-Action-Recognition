"""
Logger Utility
Custom logger for training and evaluation.
"""

import os
import sys
import logging
import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Custom logger for multi-modal action recognition."""
    
    def __init__(self, log_dir='logs', experiment_name=None, level=logging.INFO):
        self.log_dir = Path(log_dir)
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f'multimodal_action_recognition_{timestamp}'
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logger
        self.setup_file_logger(level)
        
        # Setup tensorboard writer
        self.tb_writer = SummaryWriter(log_dir=str(self.experiment_dir / 'tensorboard'))
        
        self.step_count = 0
        self.epoch_count = 0
        
    def setup_file_logger(self, level):
        """Setup file-based logger."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        log_file = self.experiment_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)
    
    def log_metrics(self, metrics, step=None, epoch=None, prefix=''):
        """Log metrics to both file and tensorboard."""
        if step is None:
            step = self.step_count
        
        if epoch is None:
            epoch = self.epoch_count
        
        # Log to file
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.info(f'{prefix}Epoch {epoch}, Step {step} - {metrics_str}')
        
        # Log to tensorboard
        for key, value in metrics.items():
            tag = f'{prefix}{key}' if prefix else key
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_learning_rate(self, lr, step=None):
        """Log learning rate."""
        if step is None:
            step = self.step_count
        
        self.tb_writer.add_scalar('learning_rate', lr, step)
        self.info(f'Learning rate: {lr:.6f}')
    
    def log_model_graph(self, model, input_tensor):
        """Log model graph to tensorboard."""
        try:
            self.tb_writer.add_graph(model, input_tensor)
            self.info('Model graph logged to tensorboard')
        except Exception as e:
            self.warning(f'Failed to log model graph: {e}')
    
    def log_confusion_matrix(self, cm, class_names, step=None, tag='confusion_matrix'):
        """Log confusion matrix to tensorboard."""
        if step is None:
            step = self.step_count
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Create confusion matrix plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Log to tensorboard
        self.tb_writer.add_figure(tag, plt.gcf(), step)
        plt.close()
    
    def log_histograms(self, model, step=None):
        """Log model parameter histograms."""
        if step is None:
            step = self.step_count
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.tb_writer.add_histogram(f'parameters/{name}', param, step)
                if param.grad is not None:
                    self.tb_writer.add_histogram(f'gradients/{name}', param.grad, step)
    
    def log_images(self, images, tag='images', step=None, max_images=8):
        """Log images to tensorboard."""
        if step is None:
            step = self.step_count
        
        # Limit number of images
        if len(images) > max_images:
            images = images[:max_images]
        
        self.tb_writer.add_images(tag, images, step)
    
    def log_video_samples(self, video_batch, predictions, targets, step=None, max_videos=4):
        """Log video samples with predictions."""
        if step is None:
            step = self.step_count
        
        # This is a placeholder - actual video logging would depend on the format
        self.info(f'Video batch shape: {video_batch.shape}')
        self.info(f'Predictions: {predictions[:max_videos]}')
        self.info(f'Targets: {targets[:max_videos]}')
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, 
                       checkpoint_dir='checkpoints', best=False):
        """Save model checkpoint."""
        checkpoint_dir = self.experiment_dir / checkpoint_dir
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'experiment_name': self.experiment_name
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.info(f'Best model saved at epoch {epoch}')
        
        self.info(f'Checkpoint saved: {checkpoint_path}')
        return str(checkpoint_path)
    
    def log_hyperparameters(self, hparams, metrics):
        """Log hyperparameters and final metrics."""
        # Convert all values to scalars for tensorboard
        hparams_dict = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparams_dict[key] = value
            else:
                hparams_dict[key] = str(value)
        
        self.tb_writer.add_hparams(hparams_dict, metrics)
    
    def increment_step(self):
        """Increment step counter."""
        self.step_count += 1
    
    def increment_epoch(self):
        """Increment epoch counter."""
        self.epoch_count += 1
        self.step_count = 0  # Reset step count for new epoch
    
    def close(self):
        """Close logger and tensorboard writer."""
        self.tb_writer.close()
        self.info('Logger closed')


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=7, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def setup_logger(log_dir='logs', experiment_name=None, level=logging.INFO):
    """Setup logger for training/evaluation."""
    return Logger(log_dir, experiment_name, level)