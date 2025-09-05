"""
Metrics Utility
Evaluation metrics for multi-modal action recognition.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from configs.datasets.drive_act import ACTION_CLASSES


class MetricsCalculator:
    """Calculate various metrics for action recognition."""
    
    def __init__(self, num_classes=34, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ACTION_CLASSES
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and targets."""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions, targets, probabilities=None):
        """
        Update metrics with new batch of predictions.
        
        Args:
            predictions: Predicted class indices [B]
            targets: True class indices [B]
            probabilities: Class probabilities [B, num_classes] (optional)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        self.all_predictions.extend(predictions.flatten())
        self.all_targets.extend(targets.flatten())
        
        if probabilities is not None:
            self.all_probabilities.extend(probabilities)
    
    def compute_accuracy(self, average='macro'):
        """Compute accuracy score."""
        if not self.all_predictions:
            return 0.0
        
        return accuracy_score(self.all_targets, self.all_predictions)
    
    def compute_precision(self, average='macro', zero_division=0):
        """Compute precision score."""
        if not self.all_predictions:
            return 0.0
        
        return precision_score(
            self.all_targets, self.all_predictions, 
            average=average, zero_division=zero_division
        )
    
    def compute_recall(self, average='macro', zero_division=0):
        """Compute recall score."""
        if not self.all_predictions:
            return 0.0
        
        return recall_score(
            self.all_targets, self.all_predictions, 
            average=average, zero_division=zero_division
        )
    
    def compute_f1_score(self, average='macro', zero_division=0):
        """Compute F1 score."""
        if not self.all_predictions:
            return 0.0
        
        return f1_score(
            self.all_targets, self.all_predictions, 
            average=average, zero_division=zero_division
        )
    
    def compute_top_k_accuracy(self, k=5):
        """Compute top-k accuracy."""
        if not self.all_probabilities:
            return 0.0
        
        probabilities = np.array(self.all_probabilities)
        targets = np.array(self.all_targets)
        
        top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
        
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_predictions[i]:
                correct += 1
        
        return correct / len(targets)
    
    def compute_confusion_matrix(self, normalize=None):
        """Compute confusion matrix."""
        if not self.all_predictions:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(
            self.all_targets, self.all_predictions, 
            labels=range(self.num_classes), normalize=normalize
        )
    
    def compute_classification_report(self, output_dict=False):
        """Compute classification report."""
        if not self.all_predictions:
            return {}
        
        return classification_report(
            self.all_targets, self.all_predictions, 
            target_names=self.class_names, 
            output_dict=output_dict, zero_division=0
        )
    
    def compute_auc_score(self, average='macro'):
        """Compute Area Under Curve (AUC) score."""
        if not self.all_probabilities or len(set(self.all_targets)) < 2:
            return 0.0
        
        try:
            targets_binarized = label_binarize(
                self.all_targets, classes=range(self.num_classes)
            )
            probabilities = np.array(self.all_probabilities)
            
            if targets_binarized.shape[1] == 1:
                # Binary classification case
                return roc_auc_score(targets_binarized, probabilities[:, 1])
            else:
                # Multi-class case
                return roc_auc_score(
                    targets_binarized, probabilities, 
                    average=average, multi_class='ovr'
                )
        except Exception:
            return 0.0
    
    def compute_per_class_accuracy(self):
        """Compute per-class accuracy."""
        if not self.all_predictions:
            return {}
        
        cm = self.compute_confusion_matrix()
        per_class_acc = {}
        
        for i, class_name in enumerate(self.class_names):
            if cm[i].sum() > 0:
                per_class_acc[class_name] = cm[i, i] / cm[i].sum()
            else:
                per_class_acc[class_name] = 0.0
        
        return per_class_acc
    
    def compute_all_metrics(self):
        """Compute all available metrics."""
        metrics = {
            'accuracy': self.compute_accuracy(),
            'precision_macro': self.compute_precision(average='macro'),
            'precision_micro': self.compute_precision(average='micro'),
            'precision_weighted': self.compute_precision(average='weighted'),
            'recall_macro': self.compute_recall(average='macro'),
            'recall_micro': self.compute_recall(average='micro'),
            'recall_weighted': self.compute_recall(average='weighted'),
            'f1_macro': self.compute_f1_score(average='macro'),
            'f1_micro': self.compute_f1_score(average='micro'),
            'f1_weighted': self.compute_f1_score(average='weighted'),
            'top_5_accuracy': self.compute_top_k_accuracy(k=5),
        }
        
        if self.all_probabilities:
            metrics['auc_macro'] = self.compute_auc_score(average='macro')
            metrics['auc_weighted'] = self.compute_auc_score(average='weighted')
        
        return metrics
    
    def plot_confusion_matrix(self, normalize=None, figsize=(12, 10), 
                            save_path=None, title='Confusion Matrix'):
        """Plot confusion matrix."""
        cm = self.compute_confusion_matrix(normalize=normalize)
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            cm, annot=True, 
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_per_class_metrics(self, save_path=None, figsize=(15, 8)):
        """Plot per-class precision, recall, and F1 scores."""
        report = self.compute_classification_report(output_dict=True)
        
        if not report:
            return None
        
        classes = self.class_names
        precision_scores = [report[cls]['precision'] for cls in classes]
        recall_scores = [report[cls]['recall'] for cls in classes]
        f1_scores = [report[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-class Precision, Recall, and F1-score')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def calculate_batch_metrics(predictions, targets, probabilities=None):
    """Calculate metrics for a single batch."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if probabilities is not None and isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average='macro', zero_division=0),
        'recall': recall_score(targets, predictions, average='macro', zero_division=0),
        'f1': f1_score(targets, predictions, average='macro', zero_division=0)
    }
    
    if probabilities is not None:
        # Top-5 accuracy
        top5_preds = np.argsort(probabilities, axis=1)[:, -5:]
        top5_correct = sum(targets[i] in top5_preds[i] for i in range(len(targets)))
        metrics['top5_accuracy'] = top5_correct / len(targets)
    
    return metrics


def compute_loss_metrics(outputs, targets, criterion):
    """Compute loss and accuracy for model outputs."""
    loss = criterion(outputs, targets)
    
    # Calculate predictions
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    accuracy = correct / total
    
    # Calculate probabilities for additional metrics
    probabilities = F.softmax(outputs, dim=1)
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'predictions': predicted,
        'probabilities': probabilities
    }


class AverageMeter:
    """Keeps track of running averages."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """Update with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressMeter:
    """Display progress during training."""
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        """Display current progress."""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        """Get batch format string."""
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'