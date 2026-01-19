
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch import nn
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
import torch
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
import warnings
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.crossval_split import generate_crossval_split
from torch.optim import Adam
from torch import distributed as dist
import copy
from scipy.ndimage import binary_erosion
from collections import OrderedDict
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from time import time, sleep
import torch.nn.functional as F
import shutil
import multiprocessing
import os
import sys 
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from torch.utils.data import Sampler
import itertools
from nnunetv2.training.logging.nnunet_logger import nnUNetLoggerCBS
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from torch import autocast, nn
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

# IN THIS CODE wE JUST HAVE NNUNET WITH region weighting startegy.

# we will write our custom dice and ce loss here, that returns the matrix instead of the mean value.  

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from typing import Callable
from nnunetv2.utilities.ddp_allgather import AllGatherGrad

class MemoryEfficientSoftDiceLoss2(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss2, self).__init__()

        # If False, background class (channel 0) is ignored in the dice computation
        self.do_bg = do_bg
        # If True, Dice is computed over the whole batch instead of per sample
        self.batch_dice = batch_dice
        # Optional non-linearity (e.g., softmax or sigmoid) to be applied to predictions
        self.apply_nonlin = apply_nonlin
        # Small constant to prevent division by zero
        self.smooth = smooth
        # If True, and using DDP (distributed data parallel), sync stats across GPUs
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))  # ensure shape is [B, 1, H, W, D]

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)
            # Count how many ground truth voxels per class (excluding ignored voxels if loss_mask is given).

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:] # drop background class from prediction too

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes) # prediction ∩ GT
            sum_pred = x.sum(axes) # prediction area
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0) # sum across batch
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc

    
from torch import nn, Tensor
class RobustCrossEntropyLoss2(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

# # class DC_and_CE_loss2(nn.Module):
# #     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
# #                 dice_class=MemoryEfficientSoftDiceLoss2):
# #         """
# #         Weights for CE and Dice do not need to sum to one. You can set whatever you want.
# #         :param soft_dice_kwargs:
# #         :param ce_kwargs:
# #         :param aggregate:
# #         :param square_dice:
# #         :param weight_ce:
# #         :param weight_dice:
# #         """
# #         super(DC_and_CE_loss2, self).__init__()
# #         if ignore_label is not None:
# #             ce_kwargs['ignore_index'] = ignore_label

# #         self.weight_dice = weight_dice
# #         self.weight_ce = weight_ce
# #         self.ignore_label = ignore_label

# #         self.ce = RobustCrossEntropyLoss2(**ce_kwargs)
# #         self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

# #     def forward(self, net_output: torch.Tensor, target: torch.Tensor):
# #         """
# #         target must be b, c, x, y(, z) with c=1
# #         :param net_output:
# #         :param target:
# #         :return:
# #         """
# #         if self.ignore_label is not None:
# #             assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
# #                                         '(DC_and_CE_loss)'
# #             mask = target != self.ignore_label
# #             # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
# #             # ignore gradients in those areas anyway
# #             target_dice = torch.where(mask, target, 0)
# #             num_fg = mask.sum()
# #         else:
# #             target_dice = target
# #             mask = None

# #         dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
# #             if self.weight_dice != 0 else 0
        
# #         ce_loss = self.ce(net_output, target[:, 0]) \
# #             if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

# #         result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
# #         return result
    

class DC_and_CE_loss2(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                dice_class=MemoryEfficientSoftDiceLoss2):
        
        super(DC_and_CE_loss2, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss2(**ce_kwargs)  # must support reduction='none'
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, voxel_weights: torch.Tensor = None):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0

        # compute per-voxel CE loss
        ce_per_voxel = self.ce(net_output, target[:, 0])

        if voxel_weights is not None:
            # voxel_weights shape 2x96x96x96
            # ce_per_voxel shape 2x96x96x96
            assert voxel_weights.shape == ce_per_voxel.shape, "Shape mismatch between voxel weights and CE loss"
            ce_loss = (ce_per_voxel * voxel_weights).mean()
        else:
            ce_loss = ce_per_voxel.mean()

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class nnUNetTrainerRegionWeighted(nnUNetTrainerNoDeepSupervision):
    """
    Trainer for semi-supervised learning with mutual learning on segmentation heads.
    """
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device('cuda'),
            percantage: float = 0.05,
        ):
        super().__init__(plans, configuration, fold, dataset_json, device, percantage)
        self.grad_scaler = None

        # NNunet base settings
        self.initial_lr = 1e-2 # 0.01
        self.weight_decay = 3e-5 # 0.00003

        self.unpack_dataset = True
        self.percentage_labeled_data = percantage # TODO: How to set this?
        self.consistency_criterion_ml = nn.CrossEntropyLoss(reduction='none')
        print(f"Percentage of labeled data: {self.percentage_labeled_data}")

        # CANSU: we added the percentage_labeled_data to the output folder name
        self.output_folder_base = join(
            nnUNet_results,
            self.plans_manager.dataset_name,
            f"{self.__class__.__name__}__{self.plans_manager.plans_name}__{configuration}__perc{self.percentage_labeled_data}"
        ) if nnUNet_results is not None else None

        self.output_folder = join(self.output_folder_base, f'fold_{fold}')


    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            # INITIALIZE THE NETWORK
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _build_loss(self):
        loss = DC_and_CE_loss2({'batch_dice': self.configuration_manager.batch_dice,'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
                                   {'reduction': 'none'}, # CANSU: we use reduction='none' to get the matrix instead of the mean value 
                                   weight_ce=1, 
                                   weight_dice=1,
                                   ignore_label=self.label_manager.ignore_label, 
                                   dice_class=MemoryEfficientSoftDiceLoss2)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        # print(f"Data shape: {data.shape}, Target shape: {target.shape}")
        # 2,1,96,96,96 and # 2,1,96,96,96

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # ADDING REGION AWARE LOSS WEIGHTS
        #####
        # Step 1: Preprocess masks
        copy_target = target.clone()
        copy_target = copy_target[:, 0]  # squeeze channel → [B, H, W, D] 4x96x96x96
        flair = data[:, 0]     # [B, H, W, D] 2x96x96x96

        lesion_mask = copy_target > 0
        hard_mask = torch.zeros_like(copy_target, dtype=torch.bool)

        for b in range(data.shape[0]):
            lesion_voxels = flair[b][lesion_mask[b]]

            if lesion_voxels.numel() == 0:
                continue

            threshold = torch.quantile(lesion_voxels, 0.66)
            hard_mask[b] = (flair[b] < threshold) & lesion_mask[b] > 0

        easy_mask = lesion_mask & ~hard_mask # 2x96x96x96

        voxel_weights = torch.ones_like(copy_target, dtype=torch.float32)  # [B, H, W, D]
        voxel_weights[hard_mask] = 2.0  #2x96x96x96

        #####
        self.optimizer.zero_grad(set_to_none=True)

        # # with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
        # #     output = self.network(data)
        # #     # del data
        # #     l = self.loss(output, target)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target, voxel_weights)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

class nnUNetTrainerRegionWeighted_100epochs(nnUNetTrainerRegionWeighted):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
