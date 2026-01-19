import math
import os
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, model_name=None):
    w, h, d = image.shape
    
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    if model_name == 'mcnet_kd':
                        output1, output2, encoder_features, decoder_features1, decoder_features2 = net(test_patch)
                        y1 = output1

                    elif model_name == 'vnet_SSNet':
                        out_seg, embedding = net(test_patch)
                        y1 = out_seg

                    elif model_name == 'vnet_bcp':
                        out_seg, features = net(test_patch)
                        y1 = out_seg

                    elif model_name == 'vnet_MLRPL':
                        out_seg1, out_seg2 = net(test_patch)
                        y1 = out_seg1

                    elif model_name=="vnet_MLRPL_ours":
                        outputs = net(test_patch)
                        y1 = outputs['out_seg1'] # to use the first decoder output 
                        # TODO: HERE WE CAN ADD THE ENSEMBLE OF THE TWO DECODERS. 


                    elif model_name=="Vnet_MLRPL_weakstrong":
                        outputs = net(test_patch)
                        y1 = outputs['out_segnp'] # from non perturbated one. 


                    elif model_name=="Vnet_MLRPL_3decoder":
                        outputs = net(test_patch)
                        y1 = outputs['out_seg1'] # from non perturbated one. 

                    
                    elif model_name=="Vnet_MLRPL_border":
                        outputs = net(test_patch)
                        y1 = outputs['out_seg1']

                    elif model_name=="Vnet_base":
                        outputs = net(test_patch)
                        y1 = outputs['out_seg1']

                    elif model_name=="swinunet":
                        y1 = net(test_patch)

                    elif model_name=="CAML_vnet":
                        y1 = net(test_patch)

                    elif model_name=="VNet_caml":
                        y1 = net(test_patch)

                    elif model_name=="CAML3d_v1":
                        y1, outputs_a, embedding_v, embedding_a = net(test_patch)

                    else:   
                        y1 = net(test_patch)
                        
                    # ensemble
                    y = torch.softmax(y1, dim=1)


                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, data_root, dataset='mbhseg24', fold=0, test_list="test.txt", 
                  num_classes=6, patch_size=(96, 96, 96), 
                  stride_xy=64, stride_z=64, model_name=None):
    """
    Test model on all cases in test set using NIfTI files.
    
    Args:
        net: model network
        data_root: root data directory (e.g., '../data')
        dataset: 'mbhseg24' or 'mbhseg25'
        fold: fold index (0-4)
        test_list: split file name (e.g., 'test.txt')
        num_classes: number of output classes
        patch_size: patch size for sliding window
        stride_xy: stride in xy plane
        stride_z: stride in z direction
        model_name: model name for output selection
    
    Returns:
        total_metric: averaged metrics per class (shape: [num_classes-1, 2] for dice and hd95)
    """
    
    # Load patient IDs from split file
    splits_dir = os.path.join(data_root, 'splits', dataset, f'fold_{fold}')
    split_file = os.path.join(splits_dir, test_list)
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        patient_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    # Setup image and label directories based on dataset
    if dataset == 'mbhseg24':
        img_dir = os.path.join(data_root, 'MBHSeg24', 'images')
        gt_dir = os.path.join(data_root, 'MBHSeg24', 'ground_truths')
    elif dataset == 'mbhseg25':
        mv_base = os.path.join(data_root, 'MBHSeg25', 'Majority_voting')
        img_dir = os.path.join(mv_base, 'images')
        gt_dir = os.path.join(mv_base, 'ground_truths')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Compute metrics for foreground classes only (1 to num_classes-1)
    total_metric = np.zeros((num_classes - 1, 2))
    print(f"Validation begin - Testing {len(patient_ids)} cases from {dataset} fold {fold}")
    
    for pid in tqdm(patient_ids):
        # Load NIfTI image
        img_path = os.path.join(img_dir, f'{pid}.nii.gz')
        gt_path = os.path.join(gt_dir, f'{pid}.nii.gz')
        
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found: {gt_path}")
            continue
        
        # Load NIfTI files
        img_nii = nib.load(img_path)
        gt_nii = nib.load(gt_path)
        image = img_nii.get_fdata().astype(np.float32)
        label = gt_nii.get_fdata().astype(np.uint8)
        
        # Apply NCCT normalization and min-max scaling (matching dataset preprocessing)
        image = np.clip(image, -100.0, 300.0)  # NCCT brain windowing
        min_v, max_v = float(image.min()), float(image.max())
        if max_v > min_v:
            image = (image - min_v) / (max_v - min_v)
        else:
            image = np.zeros_like(image, dtype=np.float32)
        
        # Run inference with sliding window
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, model_name=model_name)
        
        # Compute metrics for foreground classes (1 to num_classes-1)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    
    print("Validation end")
    return total_metric / len(patient_ids)