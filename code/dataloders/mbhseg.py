import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import nibabel as nib

# ==================== MBHSeg Dataset (NIfTI) ====================
class MBHSegDataset(Dataset):
    """
    MBHSeg dataset loader that reads patient IDs from split .txt files, loads NIfTI images
    (and optional labels), applies NCCT normalization, and scales to [0, 1].

    Args:
        data_root: Root data directory (e.g., '/media/.../Benchmarking-MBHSeg/data')
        dataset: 'mbhseg24' or 'mbhseg25'
        fold: integer fold index (0-4)
        split: 'train', 'val', or 'test'
        use_majority_labels: for 'mbhseg25', whether to use Majority_voting labels/images
        return_label: if True, returns labels alongside images
        transform: optional transform callable operating on sample dict {'image', 'label'}
    """

    def __init__(
        self,
        data_root: str,
        dataset: str,
        fold: int,
        split: str = 'train',
        transform=None,
        task: str = 'multiclass'
    ):
        super().__init__()
        assert dataset in {'mbhseg24', 'mbhseg25'}, "dataset must be 'mbhseg24' or 'mbhseg25'"
        assert split in {'train', 'val', 'test'}, "split must be 'train', 'val', or 'test'"

        self.data_root = data_root
        self.dataset = dataset
        self.fold = int(fold)
        self.split = split
        self.transform = transform
        self.task = task

        # Locate split file
        splits_dir = os.path.join(self.data_root, 'splits', dataset, f'fold_{self.fold}')
        split_file = os.path.join(splits_dir, f'{self.split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.patient_ids = [line.strip() for line in f.readlines() if line.strip()]

        # Resolve image and label base paths
        if dataset == 'mbhseg24':
            self.img_dir = os.path.join(self.data_root, 'MBHSeg24', 'images')
            self.gt_dir = os.path.join(self.data_root, 'MBHSeg24', 'ground_truths')
        else:  # mbhseg25 - we use the majority voting labels
            mv_base = os.path.join(self.data_root, 'MBHSeg25', 'Majority_voting')
            self.img_dir = os.path.join(mv_base, 'images')
            self.gt_dir = os.path.join(mv_base, 'ground_truths')

        print(f"Loaded {len(self.patient_ids)} patients for {dataset} fold {self.fold} [{self.split}]")

    def __len__(self):
        return len(self.patient_ids)

    @staticmethod
    def _ncct_normalize(arr: np.ndarray, clip_min: float = 0.0, clip_max: float = 100.0) -> np.ndarray:
        """Apply NCCT brain windowing and return float32 array."""
        arr = arr.astype(np.float32)
        arr = np.clip(arr, clip_min, clip_max)
        return arr

    @staticmethod
    def _minmax_01(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0,1] per volume."""
        min_v = float(arr.min())
        max_v = float(arr.max())
        if max_v > min_v:
            arr = (arr - min_v) / (max_v - min_v)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        return arr.astype(np.float32)

    def _load_paths(self, pid: str):
        """Resolve image/label paths for a given patient id."""
        img_path = os.path.join(self.img_dir, f'{pid}.nii.gz')
        gt_path = os.path.join(self.gt_dir, f'{pid}.nii.gz')
        return img_path, gt_path

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        img_path, gt_path = self._load_paths(pid)

        img_nii = nib.load(img_path)
        img_np = img_nii.get_fdata().astype(np.float32)
        img_np = self._ncct_normalize(img_np)
        img_np = self._minmax_01(img_np)

        gt_np = nib.load(gt_path).get_fdata().astype(np.uint8)

        # To tensor with channel-first 3D: [1, W, H, D]
        image = torch.from_numpy(img_np).unsqueeze(0)  # [1, ...]
        label = torch.from_numpy(gt_np.astype(np.uint8))
        sample = {'image': image, 'label': label, 'pid': pid}

        if self.transform is not None and gt_np is not None:
            sample_np = {'image': img_np, 'label': gt_np}
            sample_np = self.transform(sample_np)

            # Convert back to torch tensors if transform returned numpy arrays
            if isinstance(sample_np['image'], np.ndarray):
                sample['image'] = torch.from_numpy(sample_np['image']).unsqueeze(0)
            else:
                sample['image'] = sample_np['image']

            sample['label'] = sample_np['label'] if isinstance(sample_np['label'], torch.Tensor) else torch.from_numpy(sample_np['label'].astype(np.uint8))

            # if task is binary, convert label to binary
            if 'task' in self.__dict__ and self.task == 'binary':
                sample['label'] = (sample['label'] > 0).long()

        return sample




### Crop variations ###
 ##############################################
class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}

class BalancedRandomCrop(object):
    """
    Balanced random crop that ensures, with probability `foreground_prob`,
    the crop intersects foreground (brain area) based on a simple intensity mask.

    This helps avoid empty (all-zero) patches. Otherwise performs a uniform
    random crop, matching the behavior of `RandomCrop`.

    Args:
        output_size (tuple[int,int,int]): Desired crop size (w,h,d).
        foreground_prob (float): Probability to sample a foreground-biased crop (default 0.5).
        threshold (float or None): Intensity threshold for foreground in [0,1]. If None, auto-set to 0.05.
        with_sdf (bool): Whether to crop `sdf` field alongside image/label.
    """

    def __init__(self, output_size, foreground_prob: float = 0.5, threshold: float | None = None, with_sdf: bool = False):
        self.output_size = output_size
        self.foreground_prob = float(foreground_prob)
        self.threshold = threshold
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # Pad the sample if necessary to ensure we can crop output_size
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        def clamp_start(start, size, max_dim):
            return int(np.clip(start, 0, max_dim - size))

        use_foreground = np.random.rand() < self.foreground_prob

        if use_foreground:
            # Foreground mask based on intensity; assumes images are normalized to [0,1]
            thr = self.threshold if self.threshold is not None else 0.05
            fg_mask = image > thr

            if np.any(fg_mask):
                # Pick a random foreground voxel as approximate center
                coords = np.argwhere(fg_mask)
                cx, cy, cz = coords[np.random.randint(0, len(coords))]

                # Random jitter around center to introduce diversity
                jw = np.random.randint(-self.output_size[0] // 4, self.output_size[0] // 4 + 1)
                jh = np.random.randint(-self.output_size[1] // 4, self.output_size[1] // 4 + 1)
                jd = np.random.randint(-self.output_size[2] // 4, self.output_size[2] // 4 + 1)

                w1 = clamp_start(cx - self.output_size[0] // 2 + jw, self.output_size[0], w)
                h1 = clamp_start(cy - self.output_size[1] // 2 + jh, self.output_size[1], h)
                d1 = clamp_start(cz - self.output_size[2] // 2 + jd, self.output_size[2], d)
            else:
                # Fallback to random crop if no foreground present
                w1 = np.random.randint(0, max(1, w - self.output_size[0]))
                h1 = np.random.randint(0, max(1, h - self.output_size[1]))
                d1 = np.random.randint(0, max(1, d - self.output_size[2]))
        else:
            # Uniform random crop
            w1 = np.random.randint(0, max(1, w - self.output_size[0]))
            h1 = np.random.randint(0, max(1, h - self.output_size[1]))
            d1 = np.random.randint(0, max(1, d - self.output_size[2]))

        label_c = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_c = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf_c = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image_c, 'label': label_c, 'sdf': sdf_c}
        else:
            return {'image': image_c, 'label': label_c}

class ROICrop(object):
    """
    Region-of-Interest (ROI) based crop transform. Crops around a lesion/ROI mask with:
    - Tight bounding box of the mask expanded by a margin.
    - Optional random sub-crop limited by max_crop size.
    - Optional random fallback with probability (1 - prob).
    - Symmetric padding to reach max_crop size or nearest multiple of multp.
    - Applied to all items in the sample dict using the same coordinates.

    Args:
        mask_key (str): key in sample dict containing the mask (e.g., 'label').
        v2r_key (str, optional): unused; kept for compatibility.
        margin (int or list): margin around ROI bounding box (default 10).
        max_crop (tuple[int,int,int], optional): target crop size. If None, no size limiting.
        multp (int, optional): if set, pad image to nearest multiple of multp instead of max_crop.
        prob (float): probability of ROI-based crop; (1-prob) triggers random uniform crop (default 1.0).
        **kwargs_pad: arguments passed to np.pad (e.g., mode='constant', constant_values=0).
    """

    def __init__(self, mask_key, v2r_key=None, margin=10, max_crop=None, multp=None, prob=1, **kwargs):
        self.mask_key = mask_key
        self.margin = margin
        self.max_crop = max_crop
        self.multp = multp
        self.prob = prob
        self.v2r_key = v2r_key
        self.kwargs_pad = kwargs
        if prob != 1:
            assert max_crop is not None, "max_crop must be set if prob != 1"

    def _compute_pad(self, image_shape):
        """Compute symmetric padding to reach max_crop size or nearest multiple of multp."""
        pad = [0, 0, 0]
        if self.max_crop is not None:
            diff_shape = [self.max_crop[it_d] - image_shape[it_d] for it_d in range(len(image_shape))]
            pad = [np.clip(diff_shape[it_d], 0, self.max_crop[it_d]) for it_d in range(len(diff_shape))]
        elif self.multp is not None:
            pad = [
                int(np.ceil(image_shape[0] / self.multp) * self.multp - image_shape[0]),
                int(np.ceil(image_shape[1] / self.multp) * self.multp - image_shape[1]),
                int(np.ceil(image_shape[2] / self.multp) * self.multp - image_shape[2])
            ]

        padw = [
            [pad[0] // 2, pad[0] - pad[0] // 2],
            [pad[1] // 2, pad[1] - pad[1] // 2],
            [pad[2] // 2, pad[2] - pad[2] // 2]
        ]
        return padw

    def _get_random_crop_coords(self, image_shape, max_crop):
        """Generate random crop coordinates of size max_crop within image_shape."""
        diff_shape = [image_shape[it_d] - max_crop[it_d] for it_d in range(len(image_shape))]
        x = np.random.choice(np.arange(0, diff_shape[0])) if diff_shape[0] > 0 else 0
        y = np.random.choice(np.arange(0, diff_shape[1])) if diff_shape[1] > 0 else 0
        z = np.random.choice(np.arange(0, diff_shape[2])) if diff_shape[2] > 0 else 0
        crop_coords = [
            [x, x + min(max_crop[0], image_shape[0])],
            [y, y + min(max_crop[1], image_shape[1])],
            [z, z + min(max_crop[2], image_shape[2])]
        ]
        return crop_coords

    def crop_label(self, mask, margin=10, threshold=0):
        """Extract tight bounding box of mask > threshold with margin expansion."""
        ndim = len(mask.shape)
        if isinstance(margin, int):
            margin = [margin] * ndim

        crop_coord = []
        idx = np.where(mask > threshold)
        if len(idx[0]) == 0:
            # Empty mask; return full volume
            return mask, [[0, mask.shape[0]], [0, mask.shape[1]], [0, mask.shape[2]]]

        for it_index, index in enumerate(idx):
            clow = max(0, int(np.min(idx[it_index])) - margin[it_index])
            chigh = min(mask.shape[it_index], int(np.max(idx[it_index])) + margin[it_index])
            crop_coord.append([clow, chigh])

        mask_cropped = mask[
            crop_coord[0][0]: crop_coord[0][1],
            crop_coord[1][0]: crop_coord[1][1],
            crop_coord[2][0]: crop_coord[2][1]
        ]

        return mask_cropped, crop_coord

    def _get_crop_coords(self, mask):
        """Compute ROI-centered crop coordinates with optional random sub-crop."""
        m = mask.astype('int')
        _, crop_coords_1 = self.crop_label(m, margin=self.margin)

        image_shape = (
            crop_coords_1[0][1] - crop_coords_1[0][0],
            crop_coords_1[1][1] - crop_coords_1[1][0],
            crop_coords_1[2][1] - crop_coords_1[2][0]
        )

        if self.max_crop is not None:
            max_crop = self.max_crop
        else:
            max_crop = image_shape

        crop_coords_2 = self._get_random_crop_coords(image_shape, max_crop)
        crop_coords = [
            [crop_coords_1[0][0] + crop_coords_2[0][0], crop_coords_1[0][0] + crop_coords_2[0][1]],
            [crop_coords_1[1][0] + crop_coords_2[1][0], crop_coords_1[1][0] + crop_coords_2[1][1]],
            [crop_coords_1[2][0] + crop_coords_2[2][0], crop_coords_1[2][0] + crop_coords_2[2][1]]
        ]

        return crop_coords

    def __call__(self, sample):
        image = sample['image']
        mask = sample[self.mask_key] > 0
        init_image_shape = image.shape

        # Decide: ROI crop or random uniform crop
        if np.random.rand() > self.prob:
            # Random uniform crop
            if self.max_crop is not None:
                max_crop = self.max_crop
            else:
                max_crop = init_image_shape

            crop_coords = self._get_random_crop_coords(init_image_shape, max_crop)

        else:
            # ROI-centered crop
            crop_coords = self._get_crop_coords(mask)

            # Compute padding to reach target size or multiple
            image_shape = (
                crop_coords[0][1] - crop_coords[0][0],
                crop_coords[1][1] - crop_coords[1][0],
                crop_coords[2][1] - crop_coords[2][0]
            )

            padw = self._compute_pad(image_shape)

            # Expand crop region to account for padding
            crop_coords[0][0] = np.clip(crop_coords[0][0] - padw[0][0], 0, init_image_shape[0])
            crop_coords[1][0] = np.clip(crop_coords[1][0] - padw[1][0], 0, init_image_shape[1])
            crop_coords[2][0] = np.clip(crop_coords[2][0] - padw[2][0], 0, init_image_shape[2])

            crop_coords[0][1] = np.clip(crop_coords[0][1] + padw[0][1], 0, init_image_shape[0])
            crop_coords[1][1] = np.clip(crop_coords[1][1] + padw[1][1], 0, init_image_shape[1])
            crop_coords[2][1] = np.clip(crop_coords[2][1] + padw[2][1], 0, init_image_shape[2])

        # Compute final padding
        image_shape = (
            crop_coords[0][1] - crop_coords[0][0],
            crop_coords[1][1] - crop_coords[1][0],
            crop_coords[2][1] - crop_coords[2][0]
        )

        padw = self._compute_pad(image_shape)

        # Apply crop and pad to all items in sample
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            item = item[
                crop_coords[0][0]: crop_coords[0][1],
                crop_coords[1][0]: crop_coords[1][1],
                crop_coords[2][0]: crop_coords[2][1],
            ]
            item = np.pad(item, np.array(padw).astype('int'), **self.kwargs_pad)
            ret_dict[key] = item

        return ret_dict

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}



### Transformation variations ###
 ##############################################
class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}

class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}



### Sampler for labeled - unlabeled data ###
 ##############################################
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



