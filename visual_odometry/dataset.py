#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset module for the Visual Odometry system.

This module handles the scene-based splitting strategy for the RGB-D Scenes Dataset v2.
It provides functionality for loading and processing the dataset, including:
- Scene-based train/validation/test splits
- Loading RGB and depth images
- Parsing pose files
- Generating consecutive frame pairs with relative poses
- Creating PyTorch Dataset classes

The dataset consists of 14 scenes containing furniture and objects, with:
- RGB and depth images in the imgs/ directory
- Camera poses in the pc/ directory
- 3D point clouds and labels in the pc/ directory
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for dataset paths
DATASET_ROOT = "/ssd_4TB/divake/LSF_regression/dataset_rgbd_scenes_v2"
IMGS_DIR = os.path.join(DATASET_ROOT, "imgs")
PC_DIR = os.path.join(DATASET_ROOT, "pc")

# Default scene-based splits
DEFAULT_SPLITS = {
    "train": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
    "val": ["11"],
    "test": ["12", "13", "14"]
}


class DatasetConfig:
    """Configuration class for dataset parameters."""
    
    def __init__(self, 
                 splits: Optional[Dict[str, List[str]]] = None,
                 img_size: Tuple[int, int] = (640, 480),
                 max_depth: float = 10.0,
                 transform: Optional[Callable] = None):
        """
        Initialize dataset configuration.
        
        Args:
            splits: Dictionary with train/val/test scene splits
            img_size: Target image size (width, height)
            max_depth: Maximum depth value in meters
            transform: Optional transform to apply to images
        """
        self.splits = splits if splits is not None else DEFAULT_SPLITS
        self.img_size = img_size
        self.max_depth = max_depth
        self.transform = transform


def list_scene_frames(scene_id: str) -> List[Tuple[str, str]]:
    """
    List all RGB and depth image pairs for a given scene.
    
    Args:
        scene_id: Scene identifier (e.g., "01", "02", etc.)
        
    Returns:
        List of tuples containing (color_image_path, depth_image_path)
    """
    scene_dir = os.path.join(IMGS_DIR, f"scene_{scene_id}")
    
    if not os.path.exists(scene_dir):
        logger.error(f"Scene directory not found: {scene_dir}")
        return []
    
    # Get all color images
    color_images = sorted(glob.glob(os.path.join(scene_dir, "*-color.png")))
    
    # Create pairs of color and depth images
    frame_pairs = []
    for color_path in color_images:
        # Extract frame number
        frame_num = os.path.basename(color_path).split('-')[0]
        depth_path = os.path.join(scene_dir, f"{frame_num}-depth.png")
        
        if os.path.exists(depth_path):
            frame_pairs.append((color_path, depth_path))
        else:
            logger.warning(f"Missing depth image for frame {frame_num} in scene {scene_id}")
    
    return frame_pairs


def parse_pose_file(scene_id: str) -> np.ndarray:
    """
    Parse pose file for a given scene.
    
    Args:
        scene_id: Scene identifier (e.g., "01", "02", etc.)
        
    Returns:
        Numpy array of shape (N, 7) containing camera poses
        Each row is [qw, qx, qy, qz, tx, ty, tz] where:
        - qw, qx, qy, qz: quaternion representing orientation
        - tx, ty, tz: translation vector
    """
    pose_file = os.path.join(PC_DIR, f"{scene_id}.pose")
    
    if not os.path.exists(pose_file):
        logger.error(f"Pose file not found: {pose_file}")
        return np.array([])
    
    # Read pose file
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            # Parse space-separated values
            values = line.strip().split()
            if len(values) == 7:
                # Convert to float
                pose = [float(v) for v in values]
                poses.append(pose)
            else:
                logger.warning(f"Invalid pose line in {pose_file}: {line}")
    
    return np.array(poses)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]
        
    Returns:
        Result quaternion [qw, qx, qy, qz]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a quaternion.
    
    Args:
        q: Quaternion [qw, qx, qy, qz]
        
    Returns:
        Inverse quaternion [qw, -qx, -qy, -qz]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def calculate_relative_pose(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """
    Calculate relative pose from pose1 to pose2.
    
    Args:
        pose1: First pose [qw, qx, qy, qz, tx, ty, tz]
        pose2: Second pose [qw, qx, qy, qz, tx, ty, tz]
        
    Returns:
        Relative pose [qw, qx, qy, qz, tx, ty, tz]
    """
    # Extract quaternions and translations
    q1 = pose1[:4]
    t1 = pose1[4:]
    
    q2 = pose2[:4]
    t2 = pose2[4:]
    
    # Calculate relative rotation: q_rel = q2 * q1^(-1)
    q1_inv = quaternion_inverse(q1)
    q_rel = quaternion_multiply(q2, q1_inv)
    
    # Calculate relative translation: t_rel = t2 - R(q_rel) * t1
    # For simplicity, we'll use the quaternion to rotate the translation
    # This is a simplified version and might need refinement for production use
    R = quaternion_to_rotation_matrix(q_rel)
    t_rel = t2 - np.dot(R, t1)
    
    return np.concatenate([q_rel, t_rel])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        q: Quaternion [qw, qx, qy, qz]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def generate_frame_pairs(scene_id: str, max_gap: int = 1) -> List[Tuple[str, str, np.ndarray]]:
    """
    Generate pairs of consecutive frames with corresponding relative poses.
    
    Args:
        scene_id: Scene identifier (e.g., "01", "02", etc.)
        max_gap: Maximum frame gap to consider as consecutive
        
    Returns:
        List of tuples containing (frame1, frame2, relative_pose)
        where frame1 and frame2 are (color_path, depth_path) tuples
    """
    # Get all frames for the scene
    frames = list_scene_frames(scene_id)
    
    # Parse pose file
    poses = parse_pose_file(scene_id)
    
    if len(frames) == 0 or len(poses) == 0:
        logger.error(f"No frames or poses found for scene {scene_id}")
        return []
    
    # Extract frame numbers from filenames
    frame_nums = []
    for color_path, _ in frames:
        frame_num = int(os.path.basename(color_path).split('-')[0])
        frame_nums.append(frame_num)
    
    # Create mapping from frame number to index
    frame_to_idx = {num: i for i, num in enumerate(frame_nums)}
    
    # Generate pairs
    pairs = []
    for i in range(len(frames) - max_gap):
        for gap in range(1, max_gap + 1):
            if i + gap < len(frames):
                frame1 = frames[i]
                frame2 = frames[i + gap]
                
                # Get frame numbers
                frame1_num = int(os.path.basename(frame1[0]).split('-')[0])
                frame2_num = int(os.path.basename(frame2[0]).split('-')[0])
                
                # Check if we have poses for these frames
                if frame1_num < len(poses) and frame2_num < len(poses):
                    pose1 = poses[frame1_num]
                    pose2 = poses[frame2_num]
                    
                    # Calculate relative pose
                    rel_pose = calculate_relative_pose(pose1, pose2)
                    
                    pairs.append((frame1, frame2, rel_pose))
                else:
                    logger.warning(f"Missing pose for frame {frame1_num} or {frame2_num} in scene {scene_id}")
    
    return pairs


def load_rgb_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load RGB image from path.
    
    Args:
        path: Path to RGB image
        target_size: Optional target size (width, height)
        
    Returns:
        RGB image as numpy array with shape (H, W, 3)
    """
    img = Image.open(path)
    
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    
    return np.array(img)


def load_depth_image(path: str, target_size: Optional[Tuple[int, int]] = None, 
                    max_depth: float = 10.0) -> np.ndarray:
    """
    Load depth image from path and convert to meters.
    
    Args:
        path: Path to depth image
        target_size: Optional target size (width, height)
        max_depth: Maximum depth value in meters
        
    Returns:
        Depth image as numpy array with shape (H, W)
    """
    depth = Image.open(path)
    
    if target_size is not None:
        depth = depth.resize(target_size, Image.NEAREST)
    
    depth_array = np.array(depth).astype(np.float32)
    
    # Convert to meters (based on our dataset analysis)
    # For RGB-D Scenes v2, depth values need to be divided by 1000
    depth_array = depth_array / 1000.0
    
    # Create a mask for invalid depth values (0)
    valid_mask = (depth_array > 0).astype(np.float32)
    
    # Clip to maximum depth
    depth_array = np.clip(depth_array, 0, max_depth)
    
    return depth_array


class RGBDScenesDataset(Dataset):
    """PyTorch Dataset for RGB-D Scenes v2."""
    
    def __init__(self, 
                 split: str,
                 config: DatasetConfig,
                 max_frame_gap: int = 1):
        """
        Initialize the dataset.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            config: Dataset configuration
            max_frame_gap: Maximum frame gap for consecutive pairs
        """
        self.split = split
        self.config = config
        self.max_frame_gap = max_frame_gap
        
        # Get scenes for this split
        self.scenes = config.splits.get(split, [])
        
        # Generate all frame pairs
        self.frame_pairs = []
        for scene_id in self.scenes:
            pairs = generate_frame_pairs(scene_id, max_frame_gap)
            self.frame_pairs.extend(pairs)
        
        logger.info(f"Loaded {len(self.frame_pairs)} frame pairs for {split} split")
    
    def __len__(self) -> int:
        """Return the number of frame pairs."""
        return len(self.frame_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing:
            - rgb1: RGB image from first frame
            - depth1: Depth image from first frame
            - rgb2: RGB image from second frame
            - depth2: Depth image from second frame
            - rel_pose: Relative pose from first to second frame
        """
        frame1, frame2, rel_pose = self.frame_pairs[idx]
        
        # Unpack frame paths
        rgb1_path, depth1_path = frame1
        rgb2_path, depth2_path = frame2
        
        # Load images
        rgb1 = load_rgb_image(rgb1_path, self.config.img_size)
        depth1 = load_depth_image(depth1_path, self.config.img_size, self.config.max_depth)
        rgb2 = load_rgb_image(rgb2_path, self.config.img_size)
        depth2 = load_depth_image(depth2_path, self.config.img_size, self.config.max_depth)
        
        # Convert to torch tensors
        rgb1 = torch.from_numpy(rgb1).float().permute(2, 0, 1) / 255.0  # (3, H, W)
        depth1 = torch.from_numpy(depth1).float().unsqueeze(0)  # (1, H, W)
        rgb2 = torch.from_numpy(rgb2).float().permute(2, 0, 1) / 255.0  # (3, H, W)
        depth2 = torch.from_numpy(depth2).float().unsqueeze(0)  # (1, H, W)
        rel_pose = torch.from_numpy(rel_pose).float()
        
        # Apply transforms if available
        if self.config.transform is not None:
            rgb1 = self.config.transform(rgb1)
            rgb2 = self.config.transform(rgb2)
        
        return {
            'rgb1': rgb1,
            'depth1': depth1,
            'rgb2': rgb2,
            'depth2': depth2,
            'rel_pose': rel_pose
        }


def verify_dataset_structure() -> Dict[str, Dict[str, int]]:
    """
    Verify the dataset structure and report statistics.
    
    Returns:
        Dictionary with statistics for each split
    """
    stats = {
        'train': {'scenes': 0, 'frames': 0, 'pairs': 0},
        'val': {'scenes': 0, 'frames': 0, 'pairs': 0},
        'test': {'scenes': 0, 'frames': 0, 'pairs': 0}
    }
    
    # Check if dataset directories exist
    if not os.path.exists(DATASET_ROOT):
        logger.error(f"Dataset root directory not found: {DATASET_ROOT}")
        return stats
    
    if not os.path.exists(IMGS_DIR):
        logger.error(f"Images directory not found: {IMGS_DIR}")
        return stats
    
    if not os.path.exists(PC_DIR):
        logger.error(f"Point cloud directory not found: {PC_DIR}")
        return stats
    
    # Check each split
    for split, scene_ids in DEFAULT_SPLITS.items():
        stats[split]['scenes'] = len(scene_ids)
        
        for scene_id in scene_ids:
            # Count frames
            frames = list_scene_frames(scene_id)
            stats[split]['frames'] += len(frames)
            
            # Count pairs
            pairs = generate_frame_pairs(scene_id)
            stats[split]['pairs'] += len(pairs)
    
    # Print statistics
    logger.info("Dataset Statistics:")
    for split, split_stats in stats.items():
        logger.info(f"  {split.capitalize()} Split:")
        logger.info(f"    Scenes: {split_stats['scenes']}")
        logger.info(f"    Frames: {split_stats['frames']}")
        logger.info(f"    Frame Pairs: {split_stats['pairs']}")
    
    return stats


def create_dataloaders(config: DatasetConfig, 
                      batch_size: int = 8,
                      num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for all splits.
    
    Args:
        config: Dataset configuration
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        
    Returns:
        Dictionary with DataLoaders for each split
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = RGBDScenesDataset(split, config)
        
        shuffle = (split == 'train')
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


def visualize_sample(sample: Dict[str, torch.Tensor]) -> None:
    """
    Visualize a sample from the dataset.
    
    Args:
        sample: Sample dictionary from the dataset
    """
    rgb1 = sample['rgb1'].permute(1, 2, 0).numpy()
    depth1 = sample['depth1'][0].numpy()
    rgb2 = sample['rgb2'].permute(1, 2, 0).numpy()
    depth2 = sample['depth2'][0].numpy()
    rel_pose = sample['rel_pose'].numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].imshow(rgb1)
    axes[0, 0].set_title('RGB Frame 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(depth1, cmap='plasma')
    axes[0, 1].set_title('Depth Frame 1')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(rgb2)
    axes[1, 0].set_title('RGB Frame 2')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(depth2, cmap='plasma')
    axes[1, 1].set_title('Depth Frame 2')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Print relative pose
    qw, qx, qy, qz, tx, ty, tz = rel_pose
    print(f"Relative Pose:")
    print(f"  Rotation (quaternion): [{qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f}]")
    print(f"  Translation: [{tx:.4f}, {ty:.4f}, {tz:.4f}]")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Verifying dataset structure...")
    stats = verify_dataset_structure()
    
    print("\nCreating dataset with default configuration...")
    config = DatasetConfig()
    
    # Create datasets
    train_dataset = RGBDScenesDataset('train', config)
    val_dataset = RGBDScenesDataset('val', config)
    test_dataset = RGBDScenesDataset('test', config)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Example of custom split configuration
    print("\nCreating dataset with custom split configuration...")
    custom_config = DatasetConfig(
        splits={
            "train": ["01", "02", "03", "04", "05"],
            "val": ["06", "07"],
            "test": ["08", "09", "10", "11", "12", "13", "14"]
        },
        img_size=(320, 240),  # Smaller images for faster processing
        max_depth=5.0  # Limit depth range
    )
    
    custom_train_dataset = RGBDScenesDataset('train', custom_config)
    print(f"Custom train dataset size: {len(custom_train_dataset)}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(config, batch_size=4)
    
    print("Done!") 