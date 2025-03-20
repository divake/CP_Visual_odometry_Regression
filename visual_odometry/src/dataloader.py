#!/usr/bin/env python3
"""
DataLoader for RGB-D Scenes Dataset v2.

This module handles loading and preprocessing data from the RGB-D Scenes Dataset v2.
It includes:
- Custom PyTorch Dataset class for RGB-D data
- Data preprocessing functions
- Functions to create train/val/test data splits
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import cv2
import glob
from pathlib import Path
import random
from torchvision import transforms
from PIL import Image

from utils import compute_relative_pose, normalize_quaternion
import config


class RGBDDataset(Dataset):
    """
    Dataset class for RGB-D Scenes Dataset v2 visual odometry.
    
    Loads pairs of consecutive RGB frames and corresponding ground truth relative poses.
    
    Attributes:
        dataset_path (str): Path to the RGB-D Scenes Dataset
        scene_ids (list): List of scene IDs to use
        img_pairs (list): List of consecutive image pairs
        pose_pairs (list): List of pose pairs corresponding to image pairs
        transform (callable): Optional transform to apply to the images
    """
    
    def __init__(self, dataset_path, scene_ids, transform=None):
        """
        Initialize the dataset.
        
        Args:
            dataset_path (str): Path to the RGB-D Scenes Dataset
            scene_ids (list): List of scene IDs to use
            transform (callable, optional): Transform to apply to the images
        """
        self.dataset_path = dataset_path
        self.scene_ids = scene_ids
        self.transform = transform
        
        # Lists to store image pairs and pose pairs
        self.img_pairs = []
        self.pose_pairs = []
        
        # Load data for each scene
        for scene_id in scene_ids:
            self._load_scene_data(scene_id)
    
    def _load_scene_data(self, scene_id):
        """
        Load data for a specific scene.
        
        Args:
            scene_id (str): Scene ID
        """
        # Get path to the scene images and poses
        scene_img_dir = os.path.join(self.dataset_path, "imgs", f"scene_{scene_id}")
        pose_file = os.path.join(self.dataset_path, "pc", f"{scene_id}.pose")
        
        # Check if directories and files exist
        if not os.path.exists(scene_img_dir):
            print(f"Warning: Scene image directory {scene_img_dir} not found.")
            return
        
        if not os.path.exists(pose_file):
            print(f"Warning: Pose file {pose_file} not found.")
            return
        
        # Load pose data
        pose_data = self._load_poses(pose_file)
        
        # Get color images sorted by frame number
        color_files = sorted(glob.glob(os.path.join(scene_img_dir, "*-color.png")))
        
        # Extract frame numbers from color files
        frame_numbers = [int(os.path.basename(f).split("-")[0]) for f in color_files]
        
        # Create mapping from frame number to pose index (assuming pose file has one pose per line)
        frame_to_pose = {frame: i for i, frame in enumerate(frame_numbers)}
        
        # Create pairs of consecutive frames
        for i in range(len(color_files) - 1):
            frame1 = frame_numbers[i]
            frame2 = frame_numbers[i + 1]
            
            # Skip if frames are not consecutive in the pose file
            if frame2 - frame1 > 10:  # Allow some gap, but not too large
                continue
            
            # Get corresponding poses
            pose_idx1 = frame_to_pose.get(frame1)
            pose_idx2 = frame_to_pose.get(frame2)
            
            if pose_idx1 is None or pose_idx2 is None:
                continue
            
            if pose_idx1 >= len(pose_data) or pose_idx2 >= len(pose_data):
                continue
            
            pose1 = pose_data[pose_idx1]
            pose2 = pose_data[pose_idx2]
            
            # Compute relative pose from pose1 to pose2
            relative_pose = compute_relative_pose(pose1, pose2)
            
            # Add to our lists
            self.img_pairs.append((color_files[i], color_files[i + 1]))
            self.pose_pairs.append(relative_pose)
    
    def _load_poses(self, pose_file):
        """
        Load pose data from a pose file.
        
        Args:
            pose_file (str): Path to the pose file
            
        Returns:
            numpy.ndarray: Array of poses, each as [quaternion(w,x,y,z), translation(x,y,z)]
        """
        poses = []
        
        with open(pose_file, 'r') as f:
            for line in f:
                # Each line contains: a b c d x y z (quaternion + translation)
                values = [float(v) for v in line.strip().split()]
                if len(values) == 7:
                    poses.append(values)
        
        return np.array(poses)
    
    def __len__(self):
        """Return the number of image pairs in the dataset."""
        return len(self.img_pairs)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (img1, img2, relative_pose)
                - img1: First image tensor
                - img2: Second image tensor
                - relative_pose: Relative pose as [quaternion(w,x,y,z), translation(x,y,z)]
        """
        img1_path, img2_path = self.img_pairs[idx]
        relative_pose = self.pose_pairs[idx]
        
        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        # Apply transformation if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Convert relative pose to tensor
        relative_pose = torch.tensor(relative_pose, dtype=torch.float32)
        
        # Normalize the quaternion part
        relative_pose[:4] = normalize_quaternion(relative_pose[:4])
        
        return img1, img2, relative_pose
    
    def _load_image(self, img_path):
        """
        Load an image from the given path.
        
        Args:
            img_path (str): Path to the image
            
        Returns:
            PIL.Image: Loaded image
        """
        # Use PIL to open the image
        return Image.open(img_path).convert('RGB')


def get_data_transforms(img_height, img_width, use_augmentation):
    """
    Get data transformations for training and validation.
    
    Args:
        img_height (int): Height to resize images to
        img_width (int): Width to resize images to
        use_augmentation (bool): Whether to use data augmentation
        
    Returns:
        dict: Dictionary with train and validation transforms
    """
    # Define transformations
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
    }


def create_data_loaders(
    dataset_path=config.DATASET_PATH,
    scene_id=config.SCENE_ID,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    img_height=config.IMG_HEIGHT,
    img_width=config.IMG_WIDTH,
    use_augmentation=config.USE_AUGMENTATION
):
    """
    Create data loaders for training, validation, and testing.
    
    This function implements a frame-based split for scene_02:
    - 80% of frames used for training (further split into train/val with 80/20 ratio)
    - 100% of frames used for testing to evaluate the full trajectory
    
    Args:
        dataset_path (str): Path to the dataset
        scene_id (str): Scene ID to use
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of workers for data loading
        img_height (int): Height to resize images to
        img_width (int): Width to resize images to
        use_augmentation (bool): Whether to use data augmentation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get transforms with custom image dimensions
    transforms_dict = get_data_transforms(img_height, img_width, use_augmentation)
    
    # Load the entire dataset for the specified scene
    full_dataset = RGBDDataset(
        dataset_path=dataset_path,
        scene_ids=[scene_id],  # Convert to list
        transform=None  # No transform initially
    )
    
    # Total number of frame pairs
    total_pairs = len(full_dataset)
    print(f"Total frame pairs in scene_{scene_id}: {total_pairs}")
    
    # Create indices for train/test split
    indices = list(range(total_pairs))
    
    # Set random seed for reproducibility
    random.seed(42)
    # Shuffle indices
    random.shuffle(indices)
    
    # Calculate split points
    train_size = int(config.TRAIN_FRAME_RATIO * total_pairs)
    
    # Split indices
    train_val_indices = indices[:train_size]
    
    # For testing, use ALL frames to reconstruct the full trajectory
    test_indices = list(range(total_pairs)) if config.TEST_FULL_TRAJECTORY else indices[train_size:]
    
    # Create train/val datasets using the train indices
    train_val_dataset = Subset(full_dataset, train_val_indices)
    
    # Apply transforms to the train_val dataset
    train_val_dataset.dataset.transform = transforms_dict['train']
    
    # Further split train into train and validation
    train_size = int(config.TRAIN_VAL_SPLIT * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Override the transform for validation dataset
    val_dataset.dataset.dataset.transform = transforms_dict['val']
    
    # Create test dataset with all frames
    test_dataset = Subset(full_dataset, test_indices)
    test_dataset.dataset.transform = transforms_dict['val']  # Use validation transform for testing
    
    print(f"Train set: {len(train_dataset)} pairs")
    print(f"Validation set: {len(val_dataset)} pairs")
    print(f"Test set: {len(test_dataset)} pairs")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = RGBDDataset(
        dataset_path=config.DATASET_PATH,
        scene_ids=[config.SCENE_ID],
        transform=get_data_transforms(config.IMG_HEIGHT, config.IMG_WIDTH, config.USE_AUGMENTATION)['train']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img1, img2, rel_pose = dataset[0]
        print(f"Image 1 shape: {img1.shape}")
        print(f"Image 2 shape: {img2.shape}")
        print(f"Relative pose: {rel_pose}")
    
    # Test the data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=config.DATASET_PATH,
        scene_id=config.SCENE_ID,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        use_augmentation=config.USE_AUGMENTATION
    )
    
    print(f"Train loader size: {len(train_loader)}")
    print(f"Val loader size: {len(val_loader)}")
    print(f"Test loader size: {len(test_loader)}") 