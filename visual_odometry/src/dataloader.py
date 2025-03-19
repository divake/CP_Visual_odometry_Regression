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
from torch.utils.data import Dataset, DataLoader, random_split
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


def get_data_transforms():
    """
    Get data transformations for training and validation.
    
    Returns:
        dict: Dictionary with 'train' and 'val' transformations
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
    }


def create_data_loaders():
    """
    Create data loaders for training, validation, and testing.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get transforms
    transforms_dict = get_data_transforms()
    
    # Create the full training/validation dataset
    train_val_dataset = RGBDDataset(
        dataset_path=config.DATASET_PATH,
        scene_ids=config.TRAIN_SCENES,
        transform=transforms_dict['train']
    )
    
    # Split into train and validation
    dataset_size = len(train_val_dataset)
    train_size = int(config.TRAIN_VAL_SPLIT * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        train_val_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Override the transform for validation dataset
    val_dataset.dataset.transform = transforms_dict['val']
    
    # Create test dataset
    test_dataset = RGBDDataset(
        dataset_path=config.DATASET_PATH,
        scene_ids=config.TEST_SCENES,
        transform=transforms_dict['val']  # Use validation transform for testing
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = RGBDDataset(
        dataset_path=config.DATASET_PATH,
        scene_ids=["01"],
        transform=get_data_transforms()['train']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img1, img2, rel_pose = dataset[0]
        print(f"Image 1 shape: {img1.shape}")
        print(f"Image 2 shape: {img2.shape}")
        print(f"Relative pose: {rel_pose}")
    
    # Test the data loaders
    train_loader, val_loader, test_loader = create_data_loaders()
    
    print(f"Train loader size: {len(train_loader)}")
    print(f"Val loader size: {len(val_loader)}")
    print(f"Test loader size: {len(test_loader)}") 