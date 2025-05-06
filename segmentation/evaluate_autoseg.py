
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
    offset_single_centroid,
    get_baseline_metrics_autoseg,
    plot_segments_autoseg
)

# ProMERGE imports
from promerge.crf import densecrf
from promerge import dino
from promerge.utils import plot_masks, resize_pil, IoU, check_num_fg_sides, dino_similarity

class ProMERGE_Wrapper:
    """
    ProMERGE Wrapper Class that returns all masks in N×H×W format
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

    def calculate_affinity_matrix(self, dino_features: torch.tensor, seed: torch.tensor, eps=1e-5):
        """Inner product between seed feature and every feature in DINO"""
        seed = seed.unsqueeze(-1).unsqueeze(-1)
        dino_features = dino_features / (torch.linalg.norm(dino_features, axis=0) + eps)
        seed = seed / (torch.linalg.norm(seed, axis=0) + eps)
        affinity = (dino_features * seed).sum(dim=0)
        assert affinity.shape == dino_features.shape[1:]
        return affinity

    def generate_dino_affinities(self, dino_features, feat_w, feat_h):
        """Generate foreground and background affinities across the feature map"""
        masked_foreground_affinities = []
        masked_background_affinities = []

        for i in range(0, feat_w, self.stride):
            for j in range(0, feat_h, self.stride):
                seed_feature = dino_features[:, i, j]
                normalized_affinity = self.calculate_affinity_matrix(dino_features, seed_feature)

                masked_affinity = (normalized_affinity > self.bipartition_tau).float()

                fg_sides = check_num_fg_sides(masked_affinity)
                if fg_sides > 1:
                    # background candidate
                    masked_background_affinities.append(masked_affinity)
                    continue

                # mask splitting
                objects_ccs, n_ccs = ndimage.label(masked_affinity.cpu().numpy())
                object_sizes = []
                for obj_idx in range(1, n_ccs+1):
                    object_sizes.append(
                        (obj_idx, np.sum(objects_ccs[objects_ccs == obj_idx]) / obj_idx))

                # find the object corresponding to the center, then find its size
                if objects_ccs[i, j] > 0:  # Ensure the seed point is in a labeled component
                    base_object_size = object_sizes[objects_ccs[i, j] - 1][1]
                    object_sizes.sort(key=lambda x: x[1], reverse=True)

                    tmp = []
                    for object in object_sizes:
                        idx, size = object[0], object[1]
                        if size < self.cc_maskarea_tau * base_object_size:
                            break
                        masked_cc = np.zeros_like(objects_ccs)
                        masked_cc[objects_ccs == idx] = 1
                        tmp.append(masked_cc)
                        masked_cc_torch = torch.from_numpy(masked_cc).to(self.device)
                        masked_foreground_affinities.append((masked_cc_torch, torch.sum(masked_cc_torch)))

        return masked_foreground_affinities, masked_background_affinities

    def cascade_filter(self, masked_foreground_affinities, normalized_dino_features, background):
        """Filter foreground masks"""
        cascade_filtered_masks = []

        # sort the foreground masks by their size
        masked_foreground_affinities.sort(key=lambda x: x[1])
        combined_foreground = torch.zeros_like(background)

        background_mean_ft = torch.mean(normalized_dino_features[:, background.bool()], dim=-1)

        # process the foreground masks, possibly use mask splitting
        for mask, _ in masked_foreground_affinities:
            new_mask_points = torch.clone(mask)
            new_mask_points[combined_foreground > 0] = 0

            mask_mean_ft = torch.mean(
                normalized_dino_features[:, mask.bool()], dim=-1)

            if (torch.sum(new_mask_points * background) / torch.sum(new_mask_points)) < self.iou_filter_tau \
                    and mask_mean_ft @ background_mean_ft < self.foreground_include_tau:
                combined_foreground += mask
                cascade_filtered_masks.append((mask > 0).to(torch.uint8))

        return cascade_filtered_masks

    def segment_image(self, image):
        """
        Segment an image into multiple objects
        image: H×W×3 uint8 or PIL Image
        returns: N×H×W binary masks
        """
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

        # Generate affinities
        masked_foreground_affinities, masked_background_affinities = self.generate_dino_affinities(
            dino_features, feat_w, feat_h
        )

        if len(masked_foreground_affinities) == 0:
            print("No foreground masks detected")
            return np.zeros((0, image_height, image_width), dtype=np.uint8)

        normalized_dino_features = dino_features / (torch.linalg.norm(dino_features, axis=0) + 1e-5)

        # Calculate background
        if len(masked_background_affinities):
            # pixel-wise voting for background
            masked_background_affinities_agg = torch.stack(masked_background_affinities)
            background_summed = torch.mean(masked_background_affinities_agg, axis=0)
            background = (background_summed > 0.5).to(torch.int)
        else:
            print("No background masks detected")
            return np.zeros((0, image_height, image_width), dtype=np.uint8)

        # Skip if background is too small
        if torch.sum(background) / (background.shape[0] * background.shape[1]) < 0.1:
            print("Background insufficient size")
            return np.zeros((0, image_height, image_width), dtype=np.uint8)

        # Filter foreground masks
        cascade_filtered_masks = self.cascade_filter(
            masked_foreground_affinities, normalized_dino_features, background
        )

        # Cluster masks
        clustered_masks = set()
        cascade_filtered_masks.sort(key=lambda x: torch.sum(x), reverse=True)

        # Merging similar masks
        for mask in cascade_filtered_masks:
            if torch.sum(mask) == 0:
                continue
            if len(clustered_masks) == 0:
                clustered_masks.add(mask)
                continue
            masks_to_combine = []
            for cluster_mask in clustered_masks:
                # skip over empty masks
                intersection = cluster_mask & mask
                intersect_area = torch.sum(intersection)
                if intersect_area == 0:
                    continue
                # if section IOU > threshold, combine
                intersection_over_mask_area = intersect_area / torch.sum(mask)
                if intersection_over_mask_area > self.intersection_merge_tau:
                    masks_to_combine.append(cluster_mask)
                elif intersection_over_mask_area > self.merge_ioa_tau and dino_similarity(
                    mask,
                    cluster_mask,
                    normalized_dino_features,
                    self.dino_merge_tau
                ):
                    masks_to_combine.append(cluster_mask)

            # if not added to an existing cluster of masks, add to a new cluster of masks
            if len(masks_to_combine) == 0:
                clustered_masks.add(mask)
            else:
                combined_mask = torch.zeros_like(mask)
                for mask_to_combine in masks_to_combine:
                    clustered_masks.remove(mask_to_combine)
                    combined_mask += mask_to_combine
                combined_mask += mask
                combined_mask = (combined_mask > 0).to(torch.uint8)
                clustered_masks.add(combined_mask)

        # Upsample masks to fixed size
        clustered_masks_upsampled = set()
        for clustered_mask in clustered_masks:
            clustered_mask_upsampled = F.interpolate(
                clustered_mask.unsqueeze(0).unsqueeze(0),
                size=(self.fixed_size, self.fixed_size),
                mode='nearest'
            ).squeeze()
            clustered_masks_upsampled.add(clustered_mask_upsampled.to(self.device))

        clustered_masks_upsampled_list = list(clustered_masks_upsampled)

        if len(clustered_masks_upsampled_list) == 0:
            return np.zeros((0, image_height, image_width), dtype=np.uint8)

        # Apply CRF refinement and resize to original size
        masks_final = []
        for pseudo_mask in clustered_masks_upsampled_list:
            pseudo_mask_np = np.float32(pseudo_mask.cpu() >= 1)
            pseudo_mask_crf = densecrf(np.array(resized_image), pseudo_mask_np)
            pseudo_mask_crf = ndimage.binary_fill_holes(pseudo_mask_crf >= 0.5)

            mask1 = torch.from_numpy(pseudo_mask_crf).to(self.device)
            mask2 = torch.from_numpy(pseudo_mask_np).to(self.device)

            # Skip if CRF drastically changed the mask
            if IoU(mask1, mask2) < 0.5:
                continue

            # Convert to uint8, resize to original dimensions
            pseudo_mask_crf = np.uint8(pseudo_mask_crf * 255)
            pseudo_mask_crf = Image.fromarray(pseudo_mask_crf)
            pseudo_mask_crf = np.asarray(pseudo_mask_crf.resize((image_width, image_height)))

            masks_final.append(pseudo_mask_crf)

        if not masks_final:
            return np.zeros((0, image_height, image_width), dtype=np.uint8)

        # Stack all masks into N×H×W array and normalize to binary
        masks_array = np.stack(masks_final)
        masks_array = (masks_array > 127).astype(np.uint8)

        return masks_array


def main(args, h5_save_file):
    # Build ProMERGE wrapper
    promerge_wrapper = ProMERGE_Wrapper(device=args.device)

    os.makedirs(os.path.dirname(h5_save_file), exist_ok=True)

    # Load dataset
    dataset = h5py.File(args.annotations_h5, 'r')

    with h5py.File(h5_save_file, "w") as h5f:
        for img_idx, img_name in enumerate(dataset.keys()):
            print(f"Processing {img_name}...")
            # Load image
            image = dataset[img_name]
            I = image['rgb'][:]

            # Load GT segments
            gt_segs = image['segment'][:]

            # Create group for this image
            img_grp = h5f.create_group(f"img{img_idx}")
            img_grp.create_dataset("image_rgb", data=I, compression="gzip")
            img_grp.create_dataset("segments_gt", data=gt_segs, compression="gzip")

            # Get all masks for this image
            masks = promerge_wrapper.segment_image(I)

            # Save all masks
            img_grp.create_dataset("segments_pred", data=masks, compression="gzip")

    print(f"Done writing all segments to '{h5_save_file}'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--save_dir", default='./test_vis_autoseg_v0/')

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

        plot_segments_autoseg(h5_save_file, img_save_dir)

    if 'metrics' in args.test:
        assert os.path.exists(h5_save_file), f"Expected file '{h5_save_file}' to exist, but it was not found. Make sure the file is generated or the correct path is provided."
        metrics_save_json = os.path.join(save_dir, 'metrics.json')
        os.makedirs(os.path.dirname(metrics_save_json), exist_ok=True)
        if not os.path.exists(metrics_save_json):
            with open(metrics_save_json, 'w') as f:
                pass
        metrics = get_baseline_metrics_autoseg(h5_save_file, metrics_save_json)