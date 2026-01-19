"""
Helper functions to integrate multi-class metrics into existing validation and inference scripts.
Add these imports and functions to val_3D.py or other validation scripts.
"""

from code.utils import metrics
import torch
import numpy as np
from medpy import metric


def log_per_class_metrics_to_wandb(wandb, per_class_dice, iteration, phase='val'):
    """
    Log per-class metrics to W&B.
    
    Args:
        wandb: wandb module
        per_class_dice: List of 6 dice scores
        iteration: Current iteration
        phase: 'train' or 'val'
    """
    if not wandb:
        return
    
    wandb_dict = {f'{phase}/mean_dice': np.mean(per_class_dice)}
    for class_id, dice in enumerate(per_class_dice):
        class_name = metrics.CLASS_NAMES.get(class_id, f'Class_{class_id}')
        wandb_dict[f'{phase}/dice_class_{class_id}_{class_name}'] = dice
    
    wandb_dict['iteration'] = iteration
    wandb.log(wandb_dict)


def evaluate_multiclass_segmentation(predictions, targets, num_classes=6):
    """
    Comprehensive multi-class evaluation.
    
    Args:
        predictions: [D, H, W] class indices or [C, D, H, W] probabilities
        targets: [D, H, W] class indices
        num_classes: Number of classes
    
    Returns:
        results: Dictionary with per-class and overall metrics
    """
    # Convert probabilities to class indices if needed
    if predictions.ndim == 4:
        if isinstance(predictions, torch.Tensor):
            predictions = torch.argmax(predictions, dim=0).cpu().numpy()
        else:
            predictions = np.argmax(predictions, axis=0)
    
    # Convert to numpy if tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    results = {
        'per_class_dice': [],
        'per_class_precision': [],
        'per_class_recall': [],
        'per_class_jaccard': [],
    }
    
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id).astype(np.float32)
        target_mask = (targets == class_id).astype(np.float32)
        
        # Dice
        intersection = np.sum(pred_mask * target_mask)
        pred_sum = np.sum(pred_mask)
        target_sum = np.sum(target_mask)
        
        if pred_sum + target_sum > 0:
            dice = (2 * intersection) / (pred_sum + target_sum + 1e-5)
        else:
            dice = 1.0 if target_sum == 0 else 0.0
        results['per_class_dice'].append(dice)
        
        # Precision and Recall
        if pred_sum > 0:
            precision = intersection / (pred_sum + 1e-5)
        else:
            precision = 1.0 if target_sum == 0 else 0.0
        results['per_class_precision'].append(precision)
        
        if target_sum > 0:
            recall = intersection / (target_sum + 1e-5)
        else:
            recall = 1.0 if pred_sum == 0 else 0.0
        results['per_class_recall'].append(recall)
        
        # Jaccard (IoU)
        union = pred_sum + target_sum - intersection
        if union > 0:
            jaccard = intersection / (union + 1e-5)
        else:
            jaccard = 1.0 if (pred_sum + target_sum) == 0 else 0.0
        results['per_class_jaccard'].append(jaccard)
    
    # Calculate averages
    results['mean_dice'] = np.mean(results['per_class_dice'])
    results['mean_precision'] = np.mean(results['per_class_precision'])
    results['mean_recall'] = np.mean(results['per_class_recall'])
    results['mean_jaccard'] = np.mean(results['per_class_jaccard'])
    
    return results


def print_multiclass_results(results, results_dict=None):
    """
    Print comprehensive multi-class results.
    
    Args:
        results: Dictionary from evaluate_multiclass_segmentation()
        results_dict: Optional dict to accumulate results across samples
    """
    print("\n" + "=" * 80)
    print("Multi-Class Segmentation Results")
    print("=" * 80)
    
    print(f"\n{'Class':<12} {'Name':<15} {'Dice':<12} {'Prec':<12} {'Rec':<12} {'IoU':<12}")
    print("-" * 80)
    
    for class_id in range(len(results['per_class_dice'])):
        class_name = metrics.CLASS_NAMES.get(class_id, f'Class {class_id}')
        dice = results['per_class_dice'][class_id]
        prec = results['per_class_precision'][class_id]
        rec = results['per_class_recall'][class_id]
        iou = results['per_class_jaccard'][class_id]
        
        print(f"{class_id:<12} {class_name:<15} {dice:<12.4f} {prec:<12.4f} "
              f"{rec:<12.4f} {iou:<12.4f}")
    
    print("-" * 80)
    print(f"{'Mean':<12} {'':<15} {results['mean_dice']:<12.4f} "
          f"{results['mean_precision']:<12.4f} {results['mean_recall']:<12.4f} "
          f"{results['mean_jaccard']:<12.4f}")
    print("=" * 80 + "\n")
    
    return results


def process_batch_multiclass(model, batch_images, batch_targets, num_classes=6, device='cuda'):
    """
    Process a batch and compute multi-class metrics.
    
    Args:
        model: PyTorch model
        batch_images: [B, C, D, H, W]
        batch_targets: [B, D, H, W]
        num_classes: Number of classes
        device: 'cuda' or 'cpu'
    
    Returns:
        per_class_dice: Per-class dice averaged over batch
        batch_results: Metrics for each sample in batch
    """
    model.eval()
    device = torch.device(device)
    
    batch_images = batch_images.to(device)
    batch_targets = batch_targets.to(device)
    
    batch_size = batch_images.size(0)
    all_per_class_dice = [[] for _ in range(num_classes)]
    batch_results = []
    
    with torch.no_grad():
        outputs = model(batch_images)  # [B, C, D, H, W]
        predictions = torch.softmax(outputs, dim=1)  # [B, C, D, H, W]
        pred_classes = torch.argmax(predictions, dim=1)  # [B, D, H, W]
    
    # Evaluate each sample in batch
    for b in range(batch_size):
        pred = pred_classes[b].cpu().numpy()
        target = batch_targets[b].cpu().numpy()
        
        sample_results = evaluate_multiclass_segmentation(pred, target, num_classes)
        batch_results.append(sample_results)
        
        for class_id in range(num_classes):
            all_per_class_dice[class_id].append(sample_results['per_class_dice'][class_id])
    
    # Average across batch
    per_class_dice = [np.mean(dices) if len(dices) > 0 else 0.0 
                      for dices in all_per_class_dice]
    
    return per_class_dice, batch_results


# Example usage functions

def example_validation_loop():
    """Example of how to use these functions in a validation loop."""
    
    code = '''
from code.utils import metrics
from helpers import (evaluate_multiclass_segmentation, 
                    print_multiclass_results,
                    log_per_class_metrics_to_wandb)
import wandb

# During validation
model.eval()
all_results = {
    'per_class_dice': [[] for _ in range(6)],
    'per_class_jaccard': [[] for _ in range(6)],
}

for batch in val_loader:
    images, targets = batch
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
    
    # Evaluate batch
    results = evaluate_multiclass_segmentation(predictions, targets, num_classes=6)
    
    # Accumulate results
    for i in range(6):
        all_results['per_class_dice'][i].append(results['per_class_dice'][i])
        all_results['per_class_jaccard'][i].append(results['per_class_jaccard'][i])

# Average results
avg_dice = [np.mean(d) for d in all_results['per_class_dice']]
avg_jaccard = [np.mean(d) for d in all_results['per_class_jaccard']]

# Print results
print_multiclass_results({
    'per_class_dice': avg_dice,
    'per_class_jaccard': avg_jaccard,
    'mean_dice': np.mean(avg_dice),
})

# Log to W&B
log_per_class_metrics_to_wandb(wandb, avg_dice, iteration=iter_num, phase='val')
    '''
    
    print(code)


def example_inference():
    """Example of using multi-class metrics in inference."""
    
    code = '''
from helpers import evaluate_multiclass_segmentation, print_multiclass_results

# After inference on a volume
predictions = np.argmax(model_output, axis=0)  # [D, H, W]
targets = load_ground_truth()  # [D, H, W]

# Evaluate
results = evaluate_multiclass_segmentation(predictions, targets, num_classes=6)

# Print detailed results
print_multiclass_results(results)

# Per-class results
for class_id in range(6):
    class_name = metrics.CLASS_NAMES[class_id]
    dice = results['per_class_dice'][class_id]
    print(f"{class_name}: {dice:.4f}")
    '''
    
    print(code)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Integration Examples for Multi-Class Metrics")
    print("=" * 80)
    
    print("\n[Example 1] Validation Loop Integration:")
    print("-" * 80)
    example_validation_loop()
    
    print("\n[Example 2] Inference Integration:")
    print("-" * 80)
    example_inference()
    
    print("\n" + "=" * 80)
    print("Copy these functions into your validation/inference scripts")
    print("=" * 80 + "\n")
