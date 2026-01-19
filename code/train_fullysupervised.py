"""
Fully supervised training script for Vnet on MBHSeg24 dataset with W&B tracking, CSV logging, and comprehensive checkpointing.
"""
import argparse
import logging
import os
import random
import shutil
import sys
import csv
import math
from datetime import datetime

# Add code directory to Python path
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, code_dir)

import numpy as np
import torch
import nibabel as nib
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Run: pip install wandb")

try:
    import nibabel
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed. Run: pip install nibabel")

from dataloders.mbhseg import MBHSegDataset, RandomRotFlip, ROICrop, ToTensor
from utils import losses, metrics
from networks.net_factory_3d import net_factory_3d


def validate_with_sliding_window(
    model,
    data_root,
    dataset,
    fold,
    patch_size,
    stride_xy=64,
    stride_z=64,
    num_classes=6
):
    """Validation with sliding window inference (GT-aware Dice)."""

    db_val = MBHSegDataset(
        data_root=data_root,
        dataset=dataset,
        fold=fold,
        split='val',
        return_label=True,
        transform=None,
    )

    # One list PER CLASS, storing dice ONLY when class is present in GT
    all_per_class_dices = {c: [] for c in range(1, num_classes)}
    sample_data = []

    for idx in range(len(db_val)):
        sample = db_val[idx]
        image = sample['image'].squeeze().cpu().numpy()
        label = sample['label'].squeeze().cpu().numpy()

        image_original = image.copy()
        w, h, d = image.shape

        # ---------------- padding (unchanged) ----------------
        w_pad = max(patch_size[0] - w, 0)
        h_pad = max(patch_size[1] - h, 0)
        d_pad = max(patch_size[2] - d, 0)

        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
        dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

        if w_pad > 0 or h_pad > 0 or d_pad > 0:
            image = np.pad(
                image,
                [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                mode='constant',
                constant_values=0
            )

        ww, hh, dd = image.shape

        # ---------------- sliding window (unchanged) ----------------
        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

        score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
        cnt = np.zeros(image.shape, dtype=np.float32)

        for x in range(sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(sz):
                    zs = min(stride_z * z, dd - patch_size[2])

                    patch = image[
                        xs:xs+patch_size[0],
                        ys:ys+patch_size[1],
                        zs:zs+patch_size[2]
                    ]

                    patch = torch.from_numpy(
                        patch[None, None].astype(np.float32)
                    ).cuda()

                    with torch.no_grad():
                        prob = torch.softmax(model(patch), dim=1)

                    prob = prob.cpu().numpy()[0]
                    score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += prob
                    cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

        score_map /= cnt[None]
        pred_map = np.argmax(score_map, axis=0)

        # ---------------- remove padding ----------------
        if w_pad > 0 or h_pad > 0 or d_pad > 0:
            pred_map = pred_map[wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
            score_map = score_map[:, wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]

        # ---------------- GT-aware Dice computation ----------------
        per_class_dice, sample_mean_dice = metrics.compute_per_class_dice_numpy(
            score_map,
            label,
            num_classes=num_classes
        )

        # Append ONLY valid dice values
        for class_id in range(1, num_classes):
            dice = per_class_dice[class_id]
            if dice is not None:
                all_per_class_dices[class_id].append(dice)

        # Store visualization samples
        if len(sample_data) < 4:
            sample_data.append({
                'image': image_original,
                'prediction': pred_map,
                'target': label,
                'dice': sample_mean_dice  # mean over AVAILABLE FG classes
            })

    # ---------------- final aggregation ----------------
    mean_per_class_dice = {
        c: np.mean(all_per_class_dices[c])
        for c in all_per_class_dices
        if len(all_per_class_dices[c]) > 0
    }

    overall_mean_dice = np.mean(list(mean_per_class_dice.values()))

    return mean_per_class_dice, overall_mean_dice, sample_data



parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str,
                    default='../data', help='Root data path')
parser.add_argument('--dataset', type=str,
                    default='mbhseg24', help='Dataset: mbhseg24 or mbhseg25')
parser.add_argument('--fold', type=int, default=0, help='Cross-validation fold (0-4)')
parser.add_argument('--exp', type=str,
                    default='MBHSeg_Vnet_FullySup', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iterations')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

parser.add_argument('--num_classes', type=int, default=6, help='output channel of network')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

parser.add_argument('--use_wandb', type=int, default=1, help='Use Weights & Biases for tracking')
parser.add_argument('--wandb_project', type=str, default='mbhseg-vnet', help='W&B project name')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class MetricsLogger:
    """Log metrics to CSV file with per-class support."""
    def __init__(self, csv_path, num_classes=6):
        self.csv_path = csv_path
        self.num_classes = num_classes
        class_names = ['Background', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH']
        self.fieldnames = ['iter_num', 'mean_dice'] + [f'dice_class_{i}_{class_names[i]}' for i in range(num_classes)]
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_if_improved(self, iter_num, mean_dice, per_class_dice, prev_best):
        """Log only if mean dice improves over previous best."""
        if mean_dice > prev_best:
            row_data = {'iter_num': iter_num, 'mean_dice': f'{mean_dice:.6f}'}
            class_names = ['Background', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH']
            for class_id in range(1, self.num_classes):
                dice = per_class_dice.get(class_id)
                row_data[f'dice_class_{class_id}'] = (
                    f'{dice:.6f}' if dice is not None else 'NA')
            
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row_data)
            return True
        return False


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    
    # Setup W&B
    if HAS_WANDB and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp}_fold{args.fold}",
            config=vars(args),
            notes=f"Vnet fully supervised on MBHSeg24 fold {args.fold}"
        )
    
    # Initialize model
    net = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    model = net.cuda()
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    # Load MBHSeg24 dataset with ROICrop
    transform_train = transforms.Compose([
        RandomRotFlip(),
        ROICrop(mask_key='label', margin=10, max_crop=args.patch_size, prob=0.5, 
                mode='constant', constant_values=0),
        ToTensor(),
    ])
    
    db_train = MBHSegDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        fold=args.fold,
        split='train',
        return_label=True,
        transform=transform_train,
    )
    
    trainloader = DataLoader(
        db_train, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        worker_init_fn=worker_init_fn
    )
    
    model.train()
    
    # Optimizer and loss functions
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # Include background in training with lower weight to avoid dominating the loss
    class_weights = torch.tensor([0.1, 3.0, 3.0, 3.0, 3.0, 3.0], device='cuda')
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = losses.DiceLoss(num_classes) # we exlude background inside DiceLoss
    
    # Logging and checkpointing
    writer = SummaryWriter(snapshot_path + '/log')
    csv_logger = MetricsLogger(os.path.join(snapshot_path, 'validation_metrics.csv'))
    
    logging.info(f"{len(trainloader)} iterations per epoch")
    logging.info(f"Dataset: {args.dataset}, Fold: {args.fold}")
    
    iter_num = 0
    best_performance = 0.0
    lr_ = base_lr
    iterator = tqdm(range(max_iterations), ncols=70)
    
    for epoch in range(max_iterations // len(trainloader) + 1):
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            # Forward pass
            output = model(volume_batch)
            output_soft = F.softmax(output, dim=1)
            
            # Compute losses
            loss_ce = ce_loss(output, label_batch)
            loss_dice = dice_loss(output_soft, label_batch)
            loss = loss_ce + loss_dice
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_num = iter_num + 1
            
            # Learning rate decay
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            # TensorBoard logging
            writer.add_scalar('train/loss', loss, iter_num)
            writer.add_scalar('train/loss_ce', loss_ce, iter_num)
            writer.add_scalar('train/loss_dice', loss_dice, iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)
            
            # W&B logging
            if HAS_WANDB and args.use_wandb:
                wandb_log_dict = {
                    'train/loss': loss.item(),
                    'train/loss_ce': loss_ce.item(),
                    'train/loss_dice': loss_dice.item(),
                    'train/lr': lr_,
                    'iteration': iter_num,
                }
                
                # Log sample predictions every 100 iterations
                if iter_num % 100 == 0:
                    try:
                        with torch.no_grad():
                            pred_output = torch.softmax(output, dim=1)
                        metrics.log_training_batch_to_wandb(
                            wandb, volume_batch, pred_output, label_batch,
                            iteration=iter_num, num_classes=num_classes, max_samples=2
                        )
                    except Exception as e:
                        logging.warning(f"Failed to log training samples: {e}")
                
                wandb.log(wandb_log_dict)
            
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f' % (
                    iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))
            
            # Validation every 1000 iterations
            if iter_num > 0 and iter_num % 1000 == 0:
                model.eval()
                
                # Use sliding window validation
                per_class_dice, avg_dice, val_samples = validate_with_sliding_window(
                    model, 
                    args.data_root, 
                    args.dataset, 
                    args.fold, 
                    args.patch_size,
                    num_classes=num_classes
                )
                
                # Print per-class metrics
                metrics.print_per_class_metrics(per_class_dice, prefix="Validation")
                
                # TensorBoard and W&B logging
                for class_id, dice in per_class_dice.items():
                    writer.add_scalar(
                        f'val/dice_class_{class_id}_{metrics.CLASS_NAMES[class_id]}',
                        dice,
                        iter_num
                    )

                
                if HAS_WANDB and args.use_wandb:
                    wandb_log_dict[f'val/dice_class_{class_id}_...'] = dice
                    for class_id, dice in enumerate(per_class_dice):
                        wandb_log_dict[f'val/dice_class_{class_id}_{metrics.CLASS_NAMES.get(class_id, f"Class_{class_id}")}'] = dice
                    wandb.log(wandb_log_dict)
                
                # Save validation sample visualizations and NIfTI files
                try:
                    val_images_dir = os.path.join(snapshot_path, 'val_images')
                    os.makedirs(val_images_dir, exist_ok=True)
                    
                    for sample_idx, sample in enumerate(val_samples):
                        img = sample['image']  # [W, H, D]
                        pred = sample['prediction']  # [W, H, D]
                        target = sample['target']  # [W, H, D]
                        
                        # Save as NIfTI images
                        if HAS_NIBABEL:
                            # Save original image
                            img_nifti = nib.Nifti1Image(img.astype(np.float32), np.eye(4))
                            img_path = os.path.join(val_images_dir, f'sample_{sample_idx}_image.nii.gz')
                            nib.save(img_nifti, img_path)
                            
                            # Save ground truth segmentation
                            gt_nifti = nib.Nifti1Image(target.astype(np.uint8), np.eye(4))
                            gt_path = os.path.join(val_images_dir, f'sample_{sample_idx}_gt.nii.gz')
                            nib.save(gt_nifti, gt_path)
                            
                            # Save prediction
                            pred_nifti = nib.Nifti1Image(pred.astype(np.uint8), np.eye(4))
                            pred_path = os.path.join(val_images_dir, f'sample_{sample_idx}_pred.nii.gz')
                            nib.save(pred_nifti, pred_path)
                            
                            logging.info(f"Saved validation sample {sample_idx} NIfTI files to {val_images_dir}")
                        else:
                            logging.warning("nibabel not installed, skipping NIfTI saving. Install with: pip install nibabel")
                        
                        # Also log PNG slice to W&B if enabled
                        if HAS_WANDB and args.use_wandb:
                            try:
                                # Select middle slice for visualization
                                mid_z = img.shape[2] // 2
                                
                                viz_array = metrics.create_segmentation_visualizations(
                                    img[:, :, mid_z], pred[:, :, mid_z], target[:, :, mid_z],
                                    slice_idx=mid_z, num_classes=num_classes
                                )
                                
                                wandb.log({
                                    f'val/sample_{sample_idx}': wandb.Image(
                                        viz_array,
                                        caption=f"Val Sample {sample_idx}, Iter {iter_num}, Dice: {sample['dice']:.4f}"
                                    )
                                })
                            except Exception as e:
                                logging.warning(f"Failed to log visualization for sample {sample_idx}: {e}")
                except Exception as e:
                    logging.warning(f"Failed to save validation samples: {e}")
                
                logging.info('iteration %d : val_dice_score : %f' % (iter_num, avg_dice))
                
                # Save best model and log improvement
                if avg_dice > best_performance:
                    best_performance = avg_dice
                    save_mode_path = os.path.join(snapshot_path,
                                                  f'iter_{iter_num}_dice_{avg_dice:.4f}.pth')
                    save_best = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    logging.info(f"Best model saved with dice {avg_dice:.4f}")
                    
                    # Log to CSV
                    csv_logger.log_if_improved(iter_num, avg_dice, per_class_dice, best_performance)
                    if HAS_WANDB and args.use_wandb:
                        wandb.log({'best_dice': avg_dice})
                
                model.train()
            
            if iter_num >= max_iterations:
                break
        
        if iter_num >= max_iterations:
            iterator.close()
            break
    
    # Save last model
    save_last = os.path.join(snapshot_path, f'{args.model}_last_model.pth')
    torch.save(model.state_dict(), save_last)
    logging.info(f"Last model saved to {save_last}")
    
    writer.close()
    if HAS_WANDB and args.use_wandb:
        wandb.finish()
    
    logging.info(f"Training finished! Best dice: {best_performance:.4f}")
    logging.info(f"View training dashboard: https://wandb.ai/{wandb.run.entity}/{args.wandb_project}/runs/{wandb.run.id}" if (HAS_WANDB and args.use_wandb and wandb.run) else "Training finished!")
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # combine exp and labeled_num
    # args.exp = args.exp + '_' + str(args.labeled_num)
    snapshot_path = "../model/{}/{}".format(args.exp, args.model)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
        
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
