import os
import argparse
import json
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import sys
from scipy import ndimage
import torch.nn.functional as F
from torchvision import transforms

from inv.segmentation.utils import (
    resize_segments_np,
    compute_segment_centroids,
    offset_single_centroid,
    get_baseline_metrics,
    plot_segments
)

# ProMERGE imports
from promerge.crf import densecrf
from promerge import dino
from promerge.utils import plot_masks, resize_pil, IoU, check_num_fg_sides, dino_similarity

class ProMERGE_Wrapper:
    """
    ProMERGE Wrapper Class
    """

    def __init__(self, device='cuda'):
        # Initialize ProMERGE model
        self.device = device
        self.fixed_size = 480  # Default size used in ProMERGE
        self.patch_size = 8

        # DINO transformer settings
        self.vit_arch = 'base'
        self.vit_feat = 'k'
        self.feat_dim = 768

        # ProMERGE hyperparameters
        self.stride = 4
        self.bipartition_tau = 0.2
        self.cc_maskarea_tau = 0.5
        self.iou_filter_tau = 0.8
        self.foreground_include_tau = 0.1
        self.intersection_merge_tau = 0.5
        self.merge_ioa_tau = 0.1
        self.dino_merge_tau = 0.1

        # Load DINO backbone
        url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        self.backbone_dino = dino.ViTFeat(url, self.feat_dim, self.vit_arch, self.vit_feat, 8)
        self.backbone_dino.eval()
        self.backbone_dino.to(self.device)

        # Image transformation applied to all images for DINO transformer
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def segment_at_points(self, image: np.ndarray, points: np.ndarray):
        """
        image: H×W×3 uint8
        points: N×2 float, (x,y) pixel coordinates of positives
        returns: HxW mask and the central point
        """
        # Convert points to integer coordinates
        pt = [int(points[1]), int(points[0])]

        # Ensure image is in PIL format for ProMERGE processing
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        # Store original dimensions
        image_width, image_height = image_pil.width, image_pil.height

        # Resize image to fixed size for DINO
        resized_image = image_pil.resize((self.fixed_size, self.fixed_size), Image.LANCZOS)
        resized_image, _, _, feat_w, feat_h = resize_pil(resized_image, self.patch_size)

        # Convert to tensor for DINO
        image_tensor = self.to_tensor(resized_image).unsqueeze(0).to(self.device)

        # Get DINO features
        dino_features_raw = self.backbone_dino(image_tensor)[0]
        dino_features = dino_features_raw.reshape((dino_features_raw.shape[0], feat_w, feat_h)).detach()

        # Generate seed features from the point
        # Scale point to feature map coordinates
        seed_x = int(pt[0] * feat_w / image_width)
        seed_y = int(pt[1] * feat_h / image_height)

        # Make sure coordinates are in bounds
        seed_x = max(0, min(seed_x, feat_w - 1))
        seed_y = max(0, min(seed_y, feat_h - 1))

        # Generate affinity from seed
        seed_feature = dino_features[:, seed_x, seed_y]
        normalized_dino_features = dino_features / (torch.linalg.norm(dino_features, axis=0) + 1e-5)
        normalized_affinity = self.calculate_affinity_matrix(dino_features, seed_feature)

        # Create mask from affinity
        masked_affinity = (normalized_affinity > self.bipartition_tau).float()

        # Clean up the mask with connected components
        objects_ccs, n_ccs = ndimage.label(masked_affinity.cpu().numpy())

        # Get the component containing the seed point
        seed_component = objects_ccs[seed_x, seed_y]
        if seed_component == 0:  # If seed point is in background
            largest_cc = 1
            for i in range(1, n_ccs + 1):
                if np.sum(objects_ccs == i) > np.sum(objects_ccs == largest_cc):
                    largest_cc = i
            seed_component = largest_cc

        # Extract the mask
        mask = np.zeros_like(objects_ccs)
        mask[objects_ccs == seed_component] = 1

        # Convert to tensor
        mask = torch.from_numpy(mask).to(self.device)

        # Upsample to original size
        mask_upsampled = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(self.fixed_size, self.fixed_size),
            mode='nearest'
        ).squeeze()

        # Apply CRF refinement
        mask_np = np.float32(mask_upsampled.cpu() >= 1)
        mask_crf = densecrf(np.array(resized_image), mask_np)
        mask_crf = ndimage.binary_fill_holes(mask_crf >= 0.5)

        # Resize back to original dimensions
        mask_final = np.uint8(mask_crf * 255)
        mask_final = Image.fromarray(mask_final)
        mask_final = np.asarray(mask_final.resize((image_width, image_height)))

        # Binarize the final mask
        final_mask = (mask_final > 127).astype(np.uint8)

        return final_mask, pt

    def calculate_affinity_matrix(self, dino_features: torch.tensor, seed: torch.tensor, eps=1e-5):
        """Inner product between seed feature and every feature in DINO"""
        seed = seed.unsqueeze(-1).unsqueeze(-1)
        dino_features = dino_features / (torch.linalg.norm(dino_features, axis=0) + eps)
        seed = seed / (torch.linalg.norm(seed, axis=0) + eps)
        affinity = (dino_features * seed).sum(dim=0)
        assert affinity.shape == dino_features.shape[1:]
        return affinity


def main(args, h5_save_file):
    # Build ProMERGE wrapper
    promerge_wrapper = ProMERGE_Wrapper(device=args.device)

    os.makedirs(os.path.dirname(h5_save_file), exist_ok=True)

    # Collect all .png images from directory
    dataset = h5py.File(args.annotations_h5, 'r')

    with h5py.File(h5_save_file, "w") as h5f:
        for img_idx, img_name in enumerate(dataset.keys()):
            print(f"Processing {img_name}...")
            # Load image
            image = dataset[img_name]
            I = image['rgb'][:]

            # Load GT segments and centroids
            gt_segs = image['segment'][:]
            centroids = compute_segment_centroids(torch.tensor(gt_segs))

            # Create group for this image
            img_grp = h5f.create_group(f"img{img_idx}")
            img_grp.create_dataset("image_rgb", data=I, compression="gzip")
            img_grp.create_dataset("segments_gt", data=gt_segs, compression="gzip")

            for si, cent in enumerate(centroids):
                offsets = offset_single_centroid(
                    cent, N=args.num_offset_points,
                    min_mag=args.min_mag, max_mag=args.max_mag
                )
                seg_grp = img_grp.create_group(f"seg{si}")
                for pi, pt in enumerate(offsets):
                    mask, pt_mv = promerge_wrapper.segment_at_points(I, pt.detach().cpu().numpy())
                    pt_grp = seg_grp.create_group(f"pt{pi}")
                    pt_grp.create_dataset("segment", data=mask, compression="gzip")
                    pt_grp.create_dataset("centroid", data=pt_mv, compression="gzip")

    print(f"Done writing all segments to '{h5_save_file}'.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--save_dir", default='./test_vis_pointseg/')

    p.add_argument("--annotations_h5", default='/ccn2/u/lilianch/data/entityseg_100.h5')

    # test
    p.add_argument("--test", type=str,  nargs='+',
                        choices=['h5', 'vis', 'metrics'], default =['h5', 'vis', 'metrics'],
                        help='h5 for saving segments, vis for visualizing, metrics to save/print metrics')

    # vis test and metrics test: /ccn2/u/lilianch/data/segments.h5
    p.add_argument("--h5_file", default='./segments.h5',
                   help='location to retrieve segments h5 file (assumes existing)')

    # misc args
    p.add_argument("--num_offset_points", type=int, default=1)
    p.add_argument("--min_mag", type=float, default=10.0)
    p.add_argument("--max_mag", type=float, default=25.0)
    p.add_argument("--device", default="cuda:0")

    args = p.parse_args()

    save_dir = args.save_dir
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    h5_save_file = os.path.join(save_dir, 'segments.h5')
    if 'h5' in args.test:
        if os.path.exists(h5_save_file):
            response = input(f"File '{h5_save_file}' already exists. Overwrite? (y/[n]): ").strip().lower()
            if response != 'y':
                print("Aborting.")
                sys.exit(0)
        main(args, h5_save_file)

    if 'vis' in args.test:
        assert os.path.exists(h5_save_file), f"Expected file '{h5_save_file}' to exist, but it was not found. Make sure the file is generated or the correct path is provided."
        img_save_dir = os.path.join(save_dir, 'vis/')
        os.makedirs(os.path.dirname(img_save_dir), exist_ok=True)
        plot_segments(h5_save_file, img_save_dir)

    if 'metrics' in args.test:
        assert os.path.exists(h5_save_file), f"Expected file '{h5_save_file}' to exist, but it was not found. Make sure the file is generated or the correct path is provided."
        metrics_save_json = os.path.join(save_dir, 'metrics.json')
        os.makedirs(os.path.dirname(metrics_save_json), exist_ok=True)
        if not os.path.exists(metrics_save_json):
            with open(metrics_save_json, 'w') as f:
                pass
        get_baseline_metrics(h5_save_file, metrics_save_json)
