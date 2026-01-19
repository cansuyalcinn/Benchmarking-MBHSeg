"""
Training script for DCNet on MBHSeg24 dataset with W&B tracking, CSV logging, and comprehensive checkpointing.
"""
import argparse
import logging
import os
import random
import shutil
import sys
import csv
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
from torch.nn.modules.loss import CrossEntropyLoss, KLDivLoss, BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Run: pip install wandb")

from dataloders.mbhseg import MBHSegDataset, RandomRotFlip, ROICrop, ToTensor
from utils import losses, ramps, metrics
from networks.net_factory_3d import net_factory_3d
from val_3D import test_all_case
# Mixed precision disabled due to cuDNN issues
# from torch.cuda.amp import autocast, GradScaler

def get_current_consistency_weight(epoch):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def validate_with_sliding_window(model, data_root, dataset, fold, patch_size, model_name, 
                                 stride_xy=64, stride_z=64, num_classes=6):
    """Validation with sliding window inference similar to test_single_case."""
    import math

    db_val = MBHSegDataset(
        data_root=data_root,
        dataset=dataset,
        fold=fold,
        split='val',
        return_label=True,
        transform=None,
    )
    
    all_per_class_dices = [[] for _ in range(num_classes)]
    sample_data = []  # Store a few samples for visualization
    
    for idx in range(len(db_val)):
        sample = db_val[idx]
        image = sample['image'].squeeze().cpu().numpy()  # [W, H, D]
        label = sample['label'].squeeze().cpu().numpy()  # [W, H, D]
        
        # Store original image for later
        image_original = image.copy()
        
        w, h, d = image.shape
        
        # Padding if needed
        add_pad = False
        if w < patch_size[0]:
            w_pad = patch_size[0] - w
            add_pad = True
        else:
            w_pad = 0
        if h < patch_size[1]:
            h_pad = patch_size[1] - h
            add_pad = True
        else:
            h_pad = 0
        if d < patch_size[2]:
            d_pad = patch_size[2] - d
            add_pad = True
        else:
            d_pad = 0
        
        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
        dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
        
        if add_pad:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], 
                          mode='constant', constant_values=0)
        
        ww, hh, dd = image.shape
        
        # Sliding window
        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        
        score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
        cnt = np.zeros(image.shape).astype(np.float32)
        
        for x in range(0, sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    
                    test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).cuda()
                    
                    with torch.no_grad():
                        if model_name == 'mcnet_kd':
                            output1, output2, _, _, _ = model(test_patch)
                            y = torch.softmax(output1, dim=1)
                        else:
                            y = torch.softmax(model(test_patch), dim=1)
                    
                    y = y.cpu().data.numpy()
                    y = y[0, :, :, :, :]
                    score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                    cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
        
        score_map = score_map / np.expand_dims(cnt, axis=0)
        pred_map = np.argmax(score_map, axis=0)
        
        # Remove padding
        if add_pad:
            pred_map = pred_map[wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
            score_map = score_map[:, wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        
        # Compute per-class dice scores
        per_class_dice, _ = metrics.compute_per_class_dice_numpy(
            score_map,
            label, 
            num_classes=num_classes
        )
        
        for class_id, dice in enumerate(per_class_dice):
            all_per_class_dices[class_id].append(dice)
        
        # Store first few samples for visualization (use ORIGINAL unpadded image)
        if len(sample_data) < 4:
            sample_data.append({
                'image': image_original,  # Use original unpadded image
                'prediction': pred_map,
                'target': label,
                'dice': np.mean(per_class_dice)
            })
    
    # Compute mean dice per class
    mean_per_class_dice = [np.mean(dices) if len(dices) > 0 else 0.0 for dices in all_per_class_dices]
    overall_mean_dice = np.mean(mean_per_class_dice[1:]) # exclude background

    
    return mean_per_class_dice, overall_mean_dice, sample_data



parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str,
                    default='../data', help='Root data path')
parser.add_argument('--dataset', type=str,
                    default='mbhseg24', help='Dataset: mbhseg24 or mbhseg25')
parser.add_argument('--fold', type=int, default=0, help='Cross-validation fold (0-4)')
parser.add_argument('--exp', type=str,
                    default='MBHSeg_DCNet3D', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mcnet_kd', help='model_name')
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

parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')
parser.add_argument('--beta', type=float, default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--temp', default=1, type=float)

parser.add_argument('--use_wandb', type=int, default=1, help='Use Weights & Biases for tracking')
parser.add_argument('--wandb_project', type=str, default='mbhseg-dcnet', help='W&B project name')

args = parser.parse_args()

import torch.nn as nn

#### HERE WE WILL ADD THE TWOSTEAMBATCHSAMPLER ### TODO

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.p = 2

    def forward(self, g_s, g_t):
        loss = sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])
        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


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
            for i, dice in enumerate(per_class_dice):
                row_data[f'dice_class_{i}_{class_names[i]}'] = f'{dice:.6f}'
            
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
            notes=f"MBHSeg24 fold {args.fold} with ROICrop"
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
    class_weights = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], device='cuda') # we ignore background
    ce_loss = CrossEntropyLoss(weight=class_weights)

    mse_criterion = F.mse_loss
    criterion_att = Attention()
    dice_loss = losses.DiceLoss(num_classes)
    
    # Logging and checkpointing
    writer = SummaryWriter(snapshot_path + '/log')
    csv_logger = MetricsLogger(os.path.join(snapshot_path, 'validation_metrics.csv'))
    
    logging.info(f"{len(trainloader)} iterations per epoch")
    logging.info(f"Dataset: {args.dataset}, Fold: {args.fold}")
    
    iter_num = 0
    best_performance = 0.0
    lr_ = base_lr
    iterator = tqdm(range(max_iterations), ncols=70)
    cur_threshold = 1 / num_classes
    # Disable mixed precision for now due to cuDNN issues
    # scaler = GradScaler()
    
    # History for plotting
    history = {
        'iter_num': [],
        'loss': [],
        'loss_seg_dice': [],
        'loss_at_kd': [],
        'loss_dc0': [],
        'loss_cer': [],
        'consistency_weight': [],
        'val_iter_num': [],
        'val_dice': [],
    }
    
    for epoch in range(max_iterations // len(trainloader) + 1):
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            # Full precision training (disabled mixed precision due to cuDNN issues)
            output1, output2, encoder_features, decoder_features1, decoder_features2 = model(volume_batch)
            
            loss_seg_dice = 0
            
            output1_soft = F.softmax(output1, dim=1)
            output2_soft = F.softmax(output2, dim=1)
            output1_soft0 = F.softmax(output1 / 0.5, dim=1)
            output2_soft0 = F.softmax(output2 / 0.5, dim=1)
            
            # Threshold calculation
            with torch.no_grad():
                max_values1, _ = torch.max(output1_soft, dim=1)
                max_values2, _ = torch.max(output2_soft, dim=1)
                percent = (iter_num + 1) / max_iterations
                
                cur_threshold1 = (1 - percent) * cur_threshold + percent * max_values1.mean()
                cur_threshold2 = (1 - percent) * cur_threshold + percent * max_values2.mean()
                mean_max_values = min(max_values1.mean(), max_values2.mean())
                
                cur_threshold = min(cur_threshold1, cur_threshold2)
                cur_threshold = torch.clip(cur_threshold, 0.25, 0.95)
            
            mask_high = (output1_soft > cur_threshold) & (output2_soft > cur_threshold)
            mask_non_similarity = (mask_high == False)
            
            new_output1_soft = torch.mul(mask_non_similarity, output1_soft)
            new_output2_soft = torch.mul(mask_non_similarity, output2_soft)
            high_output1 = torch.mul(mask_high, output1)
            high_output2 = torch.mul(mask_high, output2)
            high_output1_soft = torch.mul(mask_high, output1_soft)
            high_output2_soft = torch.mul(mask_high, output2_soft)
            
            pseudo_output1 = torch.argmax(output1_soft, dim=1)
            pseudo_output2 = torch.argmax(output2_soft, dim=1)
            pseudo_high_output1 = torch.argmax(high_output1_soft, dim=1)
            pseudo_high_output2 = torch.argmax(high_output2_soft, dim=1)
            
            max_output1_indices = new_output1_soft > new_output2_soft
            max_output1_value0 = torch.mul(max_output1_indices, output1_soft0)
            min_output2_value0 = torch.mul(max_output1_indices, output2_soft0)
            
            max_output2_indices = new_output2_soft > new_output1_soft
            max_output2_value0 = torch.mul(max_output2_indices, output2_soft0)
            min_output1_value0 = torch.mul(max_output2_indices, output1_soft0)
            
            loss_dc0 = 0
            loss_cer = 0
            loss_at_kd = criterion_att(encoder_features, decoder_features2)
            
            loss_dc0 += mse_criterion(max_output1_value0.detach(), min_output2_value0)
            loss_dc0 += mse_criterion(max_output2_value0.detach(), min_output1_value0)
            
            # loss_seg_dice += dice_loss(output1_soft, label_batch.unsqueeze(1))
            # loss_seg_dice += dice_loss(output2_soft, label_batch.unsqueeze(1))

            loss_seg_dice += dice_loss(output1_soft, label_batch)
            loss_seg_dice += dice_loss(output2_soft, label_batch)

            
            if mean_max_values >= 0.95:
                loss_cer += ce_loss(output1, pseudo_output2.long().detach())
                loss_cer += ce_loss(output2, pseudo_output1.long().detach())
            else:
                loss_cer += ce_loss(high_output1, pseudo_high_output2.long().detach())
                loss_cer += ce_loss(high_output2, pseudo_high_output1.long().detach())
            
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            supervised_loss = loss_seg_dice
            loss = supervised_loss + (1 - consistency_weight) * (1000 * loss_at_kd) + consistency_weight * (
                1000 * loss_dc0) + 0.3 * loss_cer
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_num = iter_num + 1
            
            # Store history
            history['iter_num'].append(iter_num)
            history['loss'].append(loss.item())
            history['loss_seg_dice'].append(loss_seg_dice.item())
            history['loss_at_kd'].append(loss_at_kd.item())
            history['loss_dc0'].append(loss_dc0.item())
            history['loss_cer'].append(loss_cer.item())
            history['consistency_weight'].append(consistency_weight)
            
            # TensorBoard logging
            writer.add_scalar('train/loss', loss, iter_num)
            writer.add_scalar('train/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/loss_at_kd', loss_at_kd, iter_num)
            writer.add_scalar('train/loss_dc0', loss_dc0, iter_num)
            writer.add_scalar('train/loss_cer', loss_cer, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/cur_threshold', cur_threshold, iter_num)
            
            # W&B logging
            if HAS_WANDB and args.use_wandb:
                wandb_log_dict = {
                    'train/loss': loss.item(),
                    'train/loss_seg_dice': loss_seg_dice.item(),
                    'train/loss_at_kd': loss_at_kd.item(),
                    'train/loss_dc0': loss_dc0.item(),
                    'train/loss_cer': loss_cer.item(),
                    'train/consistency_weight': consistency_weight,
                    'train/cur_threshold': cur_threshold.item(),
                    'iteration': iter_num,
                }
                
                # Log sample predictions every 100 iterations
                if iter_num % 100 == 0:
                    try:
                        with torch.no_grad():
                            pred_output = torch.softmax(output1, dim=1)
                        metrics.log_training_batch_to_wandb(
                            wandb, volume_batch, pred_output, label_batch,
                            iteration=iter_num, num_classes=num_classes, max_samples=2
                        )
                    except Exception as e:
                        logging.warning(f"Failed to log training samples: {e}")
                
                wandb.log(wandb_log_dict)
            
            logging.info(
                'iteration %d : loss : %03f, loss_seg_dice: %03f, loss_at_kd: %03f, '
                'loss_dc0: %03f, loss_cer: %03f, consistency_weight: %03f, cur_threshold: %03f' % (
                    iter_num, loss, loss_seg_dice, loss_at_kd, loss_dc0, loss_cer,
                    consistency_weight, cur_threshold))
            
            # Validation every 200 iterations
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                
                # Use test_all_case-like validation with sliding window
                per_class_dice, avg_dice, val_samples = validate_with_sliding_window(
                    model, 
                    args.data_root, 
                    args.dataset, 
                    args.fold, 
                    args.patch_size,
                    args.model,
                    num_classes=num_classes
                )
                
                # Store validation history
                history['val_iter_num'].append(iter_num)
                history['val_dice'].append(avg_dice)
                
                # Print per-class metrics
                metrics.print_per_class_metrics(per_class_dice, prefix="Validation")
                
                # TensorBoard and W&B logging
                writer.add_scalar('val/dice_score', avg_dice, iter_num)
                for class_id, dice in enumerate(per_class_dice):
                    writer.add_scalar(f'val/dice_class_{class_id}_{metrics.CLASS_NAMES.get(class_id, f"Class_{class_id}")}', 
                                    dice, iter_num)
                
                if HAS_WANDB and args.use_wandb:
                    wandb_log_dict = {'val/dice_score': avg_dice, 'iteration': iter_num}
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
    
    # Setup snapshot path
    args.exp = f"{args.exp}_fold{args.fold}"
    snapshot_path = f"../model/{args.exp}/{args.model}"
    
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    train(args, snapshot_path)
