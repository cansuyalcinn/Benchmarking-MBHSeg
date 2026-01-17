"""
Process voxel-level samples and create majority voting dataset for MBHSeg25.
"""

import os
import shutil
import nibabel as nib
import numpy as np
from scipy import stats
from tqdm import tqdm


def process_majority_voting(mbhseg25_path, majority_images_path, majority_gts_path):
    """
    Process each voxel-level sample and perform majority voting on labels.
    
    Args:
        mbhseg25_path: Path to MBHSeg25 directory
        majority_images_path: Path to output directory for images
        majority_gts_path: Path to output directory for ground truths
    """
    # Process each voxel-level sample
    voxel_label_path = os.path.join(mbhseg25_path, 'MBH_Train_2025_voxel-label')
    voxel_folders = sorted([f for f in os.listdir(voxel_label_path) if os.path.isdir(os.path.join(voxel_label_path, f))])

    print(f"\nProcessing {len(voxel_folders)} samples...")
    print("="*60)

    processed = 0
    errors = []

    for folder_name in tqdm(voxel_folders, desc="Processing samples", unit="sample"):
        try:
            folder_path = os.path.join(voxel_label_path, folder_name)
            
            # Find image file
            image_file = None
            label_files = []
            
            for file in os.listdir(folder_path):
                if file.startswith('image') and file.endswith('.nii.gz'):
                    image_file = file
                elif file.startswith('label') and file.endswith('.nii.gz'):
                    label_files.append(file)
            
            if not image_file:
                print(f"Warning: No image file found in {folder_name}")
                errors.append(f"{folder_name}: No image file")
                continue
            
            if not label_files:
                print(f"Warning: No label files found in {folder_name}")
                errors.append(f"{folder_name}: No label files")
                continue
            
            # Copy image to the new location
            src_image = os.path.join(folder_path, image_file)
            dst_image = os.path.join(majority_images_path, f"{folder_name}.nii.gz")
            shutil.copy2(src_image, dst_image)
            
            # Load all labels and perform majority voting
            label_arrays = []
            image_nii = nib.load(src_image)
            
            for label_file in sorted(label_files):
                label_path = os.path.join(folder_path, label_file)
                label_nii = nib.load(label_path)
                label_data = label_nii.get_fdata()
                label_arrays.append(label_data)
            
            # Stack all labels and compute majority vote pixel-wise
            # Using scipy.stats.mode along axis 0 (across annotators)
            # by default, if there are 2 annotators saying different labels, we select the smaller label. it is good because if there is 0 and 1 we select 0. so we might stop mistakes.
            stacked_labels = np.stack(label_arrays, axis=0)
            majority_label, _ = stats.mode(stacked_labels, axis=0, keepdims=False)
            
            # Save majority voting result
            majority_nii = nib.Nifti1Image(majority_label, image_nii.affine, image_nii.header)
            dst_label = os.path.join(majority_gts_path, f"{folder_name}.nii.gz")
            nib.save(majority_nii, dst_label)
            
            processed += 1
            
        except Exception as e:
            error_msg = f"{folder_name}: {str(e)}"
            errors.append(error_msg)
            print(f"Error processing {folder_name}: {str(e)}")

    print("="*60)
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed}/{len(voxel_folders)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nError details:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
    
    return processed, errors


if __name__ == "__main__":
    # Set up paths
    data_path = '/media/cansu/DiskSpace/Cansu/Benchmarking-MBHSeg/data'
    mbhseg25_path = os.path.join(data_path, 'MBHSeg25')
    
    # Create output directories
    majority_voting_path = os.path.join(mbhseg25_path, 'Majority_voting')
    majority_images_path = os.path.join(majority_voting_path, 'images')
    majority_gts_path = os.path.join(majority_voting_path, 'ground_truths')

    os.makedirs(majority_images_path, exist_ok=True)
    os.makedirs(majority_gts_path, exist_ok=True)

    print(f"Created directories:")
    print(f"  {majority_images_path}")
    print(f"  {majority_gts_path}")
    
    # Process majority voting
    process_majority_voting(mbhseg25_path, majority_images_path, majority_gts_path)
    
    # Verify the created dataset
    final_images = len([f for f in os.listdir(majority_images_path) if f.endswith('.nii.gz')])
    final_labels = len([f for f in os.listdir(majority_gts_path) if f.endswith('.nii.gz')])

    print("\n" + "="*60)
    print("FINAL VERIFICATION")
    print("="*60)
    print(f"\nMBHSeg25/Majority_voting/")
    print(f"  images/: {final_images} files")
    print(f"  ground_truths/: {final_labels} files")
    print("="*60)
