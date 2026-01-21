import os
import numpy as np
import nibabel as nib
from collections import Counter
from tqdm import tqdm

all_data_mbhseg25 = "/media/cansu/DiskSpace/Cansu/Benchmarking-MBHSeg/data/MBHSeg25/MBH_Train_2025_voxel-label"
output_path = "/media/cansu/DiskSpace/Cansu/Benchmarking-MBHSeg/data/MBHSeg25/Majority_voting/ground_truths"
os.makedirs(output_path, exist_ok=True)

for folder in tqdm(sorted(os.listdir(all_data_mbhseg25))):
    folder_path = os.path.join(all_data_mbhseg25, folder)

    if not os.path.isdir(folder_path):
        continue

    image_path = os.path.join(folder_path, "image.nii.gz")
    if not os.path.exists(image_path):
        continue

    # find all annotator labels
    label_files = sorted([
        f for f in os.listdir(folder_path)
        if f.startswith("label_annot_") and f.endswith(".nii.gz")
    ])

    if len(label_files) < 2:
        print(f"Skipping {folder}: less than 2 annotations")
        continue

    # load labels
    label_arrays = []
    for lf in label_files:
        lbl_nii = nib.load(os.path.join(folder_path, lf))
        label_arrays.append(lbl_nii.get_fdata().astype(np.int16))

    label_arrays = np.stack(label_arrays, axis=0)  # (N_annot, H, W, D)

    # output array
    majority_label = np.zeros(label_arrays.shape[1:], dtype=np.int16)

    # flatten for easier iteration
    flat_labels = label_arrays.reshape(label_arrays.shape[0], -1)

    for idx in range(flat_labels.shape[1]):
        votes = flat_labels[:, idx]
        vote_count = Counter(votes)

        most_common = vote_count.most_common()
        max_count = most_common[0][1]

        # all classes that have the maximum vote count
        tied_classes = [cls for cls, cnt in most_common if cnt == max_count]

        # pick minimum class among ties
        majority_label.flat[idx] = min(tied_classes)

    # save majority-voted label
    out_nii = nib.Nifti1Image(
        majority_label,
        affine=lbl_nii.affine,
        header=lbl_nii.header
    )

    out_path = os.path.join(output_path, f"{folder}.nii.gz")
    nib.save(out_nii, out_path)

    print(f"Saved majority label for {folder} ({len(label_files)} annotators)")
