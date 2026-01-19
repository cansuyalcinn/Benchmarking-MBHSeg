"""
Multi-class segmentation metrics and visualization utilities.
Supports 6 classes: 0=Background, 1=EDH, 2=IPH, 3=IVH, 4=SAH, 5=SDH
"""

import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric


# Class names mapping
CLASS_NAMES = {
    0: 'Background',
    1: 'EDH',
    2: 'IPH',
    3: 'IVH',
    4: 'SAH',
    5: 'SDH'
}

CLASS_COLORS = {
    0: [0, 0, 0],          # Black - Background
    1: [255, 0, 0],        # Red - EDH
    2: [0, 255, 0],        # Green - IPH
    3: [0, 0, 255],        # Blue - IVH
    4: [255, 255, 0],      # Yellow - SAH
    5: [255, 0, 255],      # Magenta - SDH
}


def compute_per_class_dice(predictions, targets, num_classes=6, smooth=1e-5):
    """
    Compute per-class Dice score.
    
    Args:
        predictions: Predictions [B, C, D, H, W] (softmax probabilities) or [B, D, H, W] (argmax)
        targets: Ground truth [B, D, H, W]
        num_classes: Number of classes
        smooth: Smoothing constant
    
    Returns:
        per_class_dice: List of dice scores for each class
        mean_dice: Mean dice across FOREGROUND classes (1-5, excluding background)
    """
    if predictions.dim() == 4:
        # argmax predictions, convert to one-hot
        pred_one_hot = torch.zeros(predictions.size(0), num_classes, 
                                   predictions.size(1), predictions.size(2), 
                                   predictions.size(3), device=predictions.device)
        for i in range(num_classes):
            pred_one_hot[:, i] = (predictions == i).float()
        predictions = pred_one_hot
    
    # Convert targets to one-hot
    targets_one_hot = torch.zeros_like(predictions)
    for i in range(num_classes):
        targets_one_hot[:, i] = (targets == i).float()
    
    per_class_dice = []
    foreground_dices = []  # Only classes 1-5, excluding background
    
    for class_id in range(num_classes):
        pred = predictions[:, class_id]
        target = targets_one_hot[:, class_id]
        
        intersection = torch.sum(pred * target)
        pred_sum = torch.sum(pred * pred)
        target_sum = torch.sum(target * target)
        
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        per_class_dice.append(dice.item())
        
        # Accumulate foreground classes (1-5)
        if class_id > 0:
            foreground_dices.append(dice.item())
    
    # Mean is computed only on foreground classes (hemorrhage types)
    mean_dice = np.mean(foreground_dices) if len(foreground_dices) > 0 else 0.0
    
    return per_class_dice, mean_dice


import numpy as np

def compute_per_class_dice_numpy(
    predictions,
    targets,
    num_classes=6,
    smooth=1e-5
):
    """
    Compute per-class Dice score from numpy arrays,
    considering ONLY classes present in the ground truth.

    Args:
        predictions: [D,H,W] class indices or [C,D,H,W] probabilities
        targets: [D,H,W] ground truth labels
        num_classes: total number of classes (including background)
        smooth: smoothing constant

    Returns:
        per_class_dice: list of dice (None if class absent in GT)
        mean_fg_dice: mean Dice over AVAILABLE foreground classes
    """

    if predictions.ndim == 4:
        predictions = np.argmax(predictions, axis=0)

    per_class_dice = [None] * num_classes
    foreground_dices = []

    for class_id in range(1, num_classes):  # exclude background
        pred = (predictions == class_id).astype(np.float32)
        target = (targets == class_id).astype(np.float32)

        target_sum = np.sum(target)

        # 🚨 Skip classes not present in GT
        if target_sum == 0:
            continue

        intersection = np.sum(pred * target)
        pred_sum = np.sum(pred)

        dice = (2.0 * intersection + smooth) / (
            pred_sum + target_sum + smooth
        )

        per_class_dice[class_id] = dice
        foreground_dices.append(dice)

    mean_fg_dice = (
        np.mean(foreground_dices)
        if len(foreground_dices) > 0
        else 0.0
    )

    return per_class_dice, mean_fg_dice



def create_segmentation_visualizations(image_slice, pred_slice, target_slice, 
                                       slice_idx=None, num_classes=6):
    """
    Create visualizations of predictions and targets on a single slice.
    
    Args:
        image_slice: Input image slice [H, W]
        pred_slice: Prediction slice [H, W] (class indices)
        target_slice: Target slice [H, W] (class indices)
        slice_idx: Slice index for display
        num_classes: Number of classes
    
    Returns:
        image_array: Numpy array suitable for wandb.Image()
        pred_array: Numpy array with colored predictions
        target_array: Numpy array with colored targets
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Normalize image to [0, 1]
    if isinstance(image_slice, torch.Tensor):
        image_slice = image_slice.cpu().numpy()
    if isinstance(pred_slice, torch.Tensor):
        pred_slice = pred_slice.cpu().numpy()
    if isinstance(target_slice, torch.Tensor):
        target_slice = target_slice.cpu().numpy()
    
    img_min, img_max = image_slice.min(), image_slice.max()
    if img_max > img_min:
        image_normalized = (image_slice - img_min) / (img_max - img_min)
    else:
        image_normalized = np.zeros_like(image_slice)
    
    # Create colored segmentation maps
    def create_colored_mask(mask, num_classes):
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in range(num_classes):
            class_mask = (mask == class_id)
            if class_id in CLASS_COLORS:
                colored[class_mask] = CLASS_COLORS[class_id]
        return colored
    
    pred_colored = create_colored_mask(pred_slice, num_classes)
    target_colored = create_colored_mask(target_slice, num_classes)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_normalized, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(image_normalized, cmap='gray', alpha=0.5)
    axes[1].imshow(pred_colored, alpha=0.5)
    axes[1].set_title(f'Prediction (Slice {slice_idx})')
    axes[1].axis('off')
    
    # Ground truth
    axes[2].imshow(image_normalized, cmap='gray', alpha=0.5)
    axes[2].imshow(target_colored, alpha=0.5)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=np.array(CLASS_COLORS[i])/255.0, 
                                      label=CLASS_NAMES[i]) 
                      for i in range(num_classes)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, -0.05))
    
    # Convert to image array using savefig
    import io
    from PIL import Image
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image_array = np.array(Image.open(buf))
    buf.close()
    
    plt.close(fig)
    
    return image_array


def log_validation_batch_to_wandb(wandb, batch_images, batch_predictions, batch_targets, 
                                   batch_dices, iteration, num_classes=6, max_samples=4):
    """
    Log validation batch samples to W&B with predictions and targets.
    
    Args:
        wandb: wandb module
        batch_images: [B, 1, D, H, W]
        batch_predictions: [B, D, H, W] or [B, C, D, H, W]
        batch_targets: [B, D, H, W]
        batch_dices: Per-class dice scores for the batch
        iteration: Current iteration
        num_classes: Number of classes
        max_samples: Maximum number of samples to log
    """
    if isinstance(batch_images, torch.Tensor):
        batch_images = batch_images.cpu().numpy()
    if isinstance(batch_predictions, torch.Tensor):
        if batch_predictions.dim() == 5:
            batch_predictions = torch.argmax(batch_predictions, dim=1).cpu().numpy()
        else:
            batch_predictions = batch_predictions.cpu().numpy()
    if isinstance(batch_targets, torch.Tensor):
        batch_targets = batch_targets.cpu().numpy()
    
    # Log a few samples
    num_samples = min(max_samples, batch_images.shape[0])
    images_to_log = []
    
    for sample_idx in range(num_samples):
        img = batch_images[sample_idx, 0]  # [D, H, W]
        pred = batch_predictions[sample_idx]  # [D, H, W]
        target = batch_targets[sample_idx]  # [D, H, W]
        
        # Select middle slice in z-direction
        mid_z = img.shape[0] // 2
        
        viz_array = create_segmentation_visualizations(
            img[mid_z], pred[mid_z], target[mid_z],
            slice_idx=mid_z, num_classes=num_classes
        )
        
        images_to_log.append(wandb.Image(
            viz_array,
            caption=f"Sample {sample_idx}, Iter {iteration}"
        ))
    
    wandb.log({f"val/sample_{i}": img for i, img in enumerate(images_to_log)})


def log_training_batch_to_wandb(wandb, batch_images, batch_predictions, batch_targets,
                                iteration, num_classes=6, max_samples=2):
    """
    Log training batch samples to W&B with predictions and targets.
    
    Args:
        wandb: wandb module
        batch_images: [B, 1, D, H, W]
        batch_predictions: [B, C, D, H, W]
        batch_targets: [B, D, H, W]
        iteration: Current iteration
        num_classes: Number of classes
        max_samples: Maximum number of samples to log
    """
    if isinstance(batch_images, torch.Tensor):
        batch_images = batch_images.cpu().detach().numpy()
    if isinstance(batch_predictions, torch.Tensor):
        batch_predictions = torch.argmax(batch_predictions, dim=1).cpu().detach().numpy()
    if isinstance(batch_targets, torch.Tensor):
        batch_targets = batch_targets.cpu().detach().numpy()
    
    # Log a few samples
    num_samples = min(max_samples, batch_images.shape[0])
    images_to_log = []
    
    for sample_idx in range(num_samples):
        img = batch_images[sample_idx, 0]  # [D, H, W]
        pred = batch_predictions[sample_idx]  # [D, H, W]
        target = batch_targets[sample_idx]  # [D, H, W]
        
        # Select middle slice in z-direction
        mid_z = img.shape[0] // 2
        
        viz_array = create_segmentation_visualizations(
            img[mid_z], pred[mid_z], target[mid_z],
            slice_idx=mid_z, num_classes=num_classes
        )
        
        images_to_log.append(wandb.Image(
            viz_array,
            caption=f"Train Sample {sample_idx}, Iter {iteration}"
        ))
    
    wandb.log({f"train/sample_{i}": img for i, img in enumerate(images_to_log)})


def print_per_class_metrics(per_class_dice, prefix="Validation"):
    """
    per_class_dice: dict {class_id: mean_dice}
    """
    print(f"\n{prefix} Per-Class Dice Scores:")
    print("-" * 50)
    print(f"{'Class':<15} {'Name':<15} {'Dice Score':<15}")
    print("-" * 50)

    dices = []

    for class_id, dice in sorted(per_class_dice.items()):
        class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
        print(f"{class_id:<15} {class_name:<15} {dice:<15.6f}")
        dices.append(dice)

    if len(dices) > 0:
        print("-" * 50)
        print(f"{'Mean':<15} {'FG Only':<15} {np.mean(dices):<15.6f}")

