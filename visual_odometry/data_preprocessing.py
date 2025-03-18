#!/usr/bin/env python3

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class RGBDOdometryDataset(Dataset):
    def __init__(self, dataset_path, scene_id, transform=None, split='train', 
                 train_ratio=0.7, val_ratio=0.15, use_depth=False):
        """
        PyTorch Dataset for RGBD Odometry.
        
        Args:
            dataset_path (str): Path to the RGB-D dataset
            scene_id (str): Scene ID to use
            transform (callable, optional): Transform to apply to images
            split (str): Dataset split ('train', 'val', or 'test')
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            use_depth (bool): Whether to use depth images
        """
        self.dataset_path = dataset_path
        self.scene_id = scene_id
        self.transform = transform
        self.split = split
        self.use_depth = use_depth
        
        # Image paths
        self.scene_img_path = os.path.join(dataset_path, "imgs", f"scene_{scene_id}")
        
        # Get all RGB images and depth images
        self.color_images = sorted([f for f in os.listdir(self.scene_img_path) 
                                    if f.endswith("-color.png")])
        
        self.depth_images = sorted([f for f in os.listdir(self.scene_img_path) 
                                    if f.endswith("-depth.png")])
        
        # Ensure we have matching pairs of color and depth images
        self._verify_image_pairs()
        
        # Load all poses
        self.poses = self._load_poses()
        
        # Create frame pairs (current and next frame)
        self.frame_pairs = []
        for i in range(len(self.color_images) - 1):
            self.frame_pairs.append((i, i+1))
        
        # Split into train/val/test
        total_pairs = len(self.frame_pairs)
        train_end = int(total_pairs * train_ratio)
        val_end = train_end + int(total_pairs * val_ratio)
        
        if split == 'train':
            self.frame_pairs = self.frame_pairs[:train_end]
        elif split == 'val':
            self.frame_pairs = self.frame_pairs[train_end:val_end]
        else:  # test
            self.frame_pairs = self.frame_pairs[val_end:]
            
        print(f"Created {split} dataset with {len(self.frame_pairs)} image pairs")
    
    def _verify_image_pairs(self):
        """
        Verify that we have matching pairs of color and depth images.
        """
        color_frames = set([f.split('-')[0] for f in self.color_images])
        depth_frames = set([f.split('-')[0] for f in self.depth_images])
        
        # Find common frame numbers
        common_frames = color_frames.intersection(depth_frames)
        
        # Filter images to only include those with matching pairs
        self.color_images = sorted([f for f in self.color_images if f.split('-')[0] in common_frames])
        self.depth_images = sorted([f for f in self.depth_images if f.split('-')[0] in common_frames])
        
        # Map frame number to index for quick lookup
        self.color_frame_to_idx = {f.split('-')[0]: i for i, f in enumerate(self.color_images)}
        self.depth_frame_to_idx = {f.split('-')[0]: i for i, f in enumerate(self.depth_images)}
    
    def _load_poses(self):
        """
        Load camera poses from the pose file.
        
        Returns:
            numpy.ndarray: Array of camera poses (quaternions and translations)
        """
        pose_file_path = os.path.join(self.dataset_path, "pc", f"{self.scene_id}.pose")
        poses = []
        
        with open(pose_file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            values = [float(v) for v in line.strip().split()]
            if len(values) == 7:  # Ensure valid pose line
                poses.append(values)
        
        # Ensure we have the same number of poses as images
        assert len(poses) >= len(self.color_images), f"Not enough poses ({len(poses)}) for images ({len(self.color_images)})"
        
        # Trim poses to match image count if necessary
        if len(poses) > len(self.color_images):
            poses = poses[:len(self.color_images)]
            
        return np.array(poses)
    
    def _get_relative_pose(self, pose1, pose2):
        """
        Calculate the relative pose between two absolute poses.
        
        Args:
            pose1 (numpy.ndarray): First pose (quaternion and translation)
            pose2 (numpy.ndarray): Second pose (quaternion and translation)
            
        Returns:
            numpy.ndarray: Relative pose (quaternion and translation)
        """
        # Extract quaternions and translations
        q1 = pose1[:4]  # w, x, y, z
        t1 = pose1[4:7]
        
        q2 = pose2[:4]
        t2 = pose2[4:7]
        
        # Convert quaternions to rotation matrices (scipy uses x,y,z,w)
        R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()
        R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]]).as_matrix()
        
        # Calculate relative rotation: R_rel = R2 * R1^T
        R_rel = np.dot(R2, R1.T)
        
        # Calculate relative translation: t_rel = t2 - R_rel * t1
        t_rel = t2 - np.dot(R_rel, t1)
        
        # Convert back to quaternion (scipy returns x,y,z,w)
        q_rel_scipy = Rotation.from_matrix(R_rel).as_quat()
        
        # Convert to w,x,y,z format
        q_rel = np.array([q_rel_scipy[3], q_rel_scipy[0], q_rel_scipy[1], q_rel_scipy[2]])
        
        # Normalize quaternion
        q_rel = q_rel / np.linalg.norm(q_rel)
        
        return np.concatenate([q_rel, t_rel])
    
    def __len__(self):
        """
        Return the number of frame pairs.
        """
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        """
        Get a pair of frames with relative pose.
        
        Args:
            idx (int): Index into the dataset
            
        Returns:
            dict: Dictionary containing image pairs and relative pose
        """
        frame1_idx, frame2_idx = self.frame_pairs[idx]
        
        # Load RGB images
        img1_path = os.path.join(self.scene_img_path, self.color_images[frame1_idx])
        img2_path = os.path.join(self.scene_img_path, self.color_images[frame2_idx])
        
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Create output dictionary
        output = {}
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        output['img1'] = img1
        output['img2'] = img2
        
        # Load depth images if required
        if self.use_depth:
            depth1_path = os.path.join(self.scene_img_path, self.depth_images[frame1_idx])
            depth2_path = os.path.join(self.scene_img_path, self.depth_images[frame2_idx])
            
            depth1 = cv2.imread(depth1_path, cv2.IMREAD_ANYDEPTH)
            depth2 = cv2.imread(depth2_path, cv2.IMREAD_ANYDEPTH)
            
            # Convert to tensor and normalize (depth values are in millimeters)
            depth1 = torch.tensor(depth1.astype(np.float32) / 5000.0, dtype=torch.float32).unsqueeze(0)
            depth2 = torch.tensor(depth2.astype(np.float32) / 5000.0, dtype=torch.float32).unsqueeze(0)
            
            output['depth1'] = depth1
            output['depth2'] = depth2
        
        # Get relative pose
        pose1 = self.poses[frame1_idx]
        pose2 = self.poses[frame2_idx]
        rel_pose = self._get_relative_pose(pose1, pose2)
        
        output['rel_pose'] = torch.tensor(rel_pose, dtype=torch.float32)
        output['frame1_idx'] = frame1_idx
        output['frame2_idx'] = frame2_idx
        
        return output

def create_data_loaders(dataset_path, scene_id, batch_size=32, use_depth=False):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        dataset_path (str): Path to the RGB-D dataset
        scene_id (str): Scene ID to use
        batch_size (int): Batch size for data loaders
        use_depth (bool): Whether to use depth images
        
    Returns:
        tuple: Tuple of (train_loader, val_loader, test_loader)
    """
    # Define transforms for training and evaluation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = RGBDOdometryDataset(
        dataset_path=dataset_path,
        scene_id=scene_id,
        transform=train_transform,
        split='train',
        use_depth=use_depth
    )
    
    val_dataset = RGBDOdometryDataset(
        dataset_path=dataset_path,
        scene_id=scene_id,
        transform=eval_transform,
        split='val',
        use_depth=use_depth
    )
    
    test_dataset = RGBDOdometryDataset(
        dataset_path=dataset_path,
        scene_id=scene_id,
        transform=eval_transform,
        split='test',
        use_depth=use_depth
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def verify_data_pipeline(data_loader, save_path=None):
    """
    Verify the data pipeline by displaying sample image pairs and poses.
    
    Args:
        data_loader (DataLoader): DataLoader to sample from
        save_path (str, optional): Path to save visualization
    """
    # Get a batch of data
    batch = next(iter(data_loader))
    
    # Extract images and poses
    img1_batch = batch['img1']
    img2_batch = batch['img2']
    rel_pose_batch = batch['rel_pose']
    frame1_idx_batch = batch['frame1_idx']
    frame2_idx_batch = batch['frame2_idx']
    
    print(f"Batch information:")
    print(f"Image 1 shape: {img1_batch.shape}")
    print(f"Image 2 shape: {img2_batch.shape}")
    print(f"Relative pose shape: {rel_pose_batch.shape}")
    
    # Check if depth images are included
    if 'depth1' in batch:
        depth1_batch = batch['depth1']
        depth2_batch = batch['depth2']
        print(f"Depth 1 shape: {depth1_batch.shape}")
        print(f"Depth 2 shape: {depth2_batch.shape}")
    
    # Display sample pairs
    num_samples = min(4, img1_batch.shape[0])
    fig_rows = 2 if 'depth1' not in batch else 4
    
    plt.figure(figsize=(15, 5 * fig_rows // 2))
    
    for i in range(num_samples):
        # Convert tensor to numpy and denormalize RGB images
        img1 = img1_batch[i].permute(1, 2, 0).numpy()
        img1 = img1 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img1 = np.clip(img1, 0, 1)
        
        img2 = img2_batch[i].permute(1, 2, 0).numpy()
        img2 = img2 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img2 = np.clip(img2, 0, 1)
        
        # Display RGB images
        plt.subplot(fig_rows, num_samples, i+1)
        plt.imshow(img1)
        plt.title(f"Frame 1 (idx: {frame1_idx_batch[i].item()})")
        plt.axis('off')
        
        plt.subplot(fig_rows, num_samples, i+1+num_samples)
        plt.imshow(img2)
        plt.title(f"Frame 2 (idx: {frame2_idx_batch[i].item()})")
        plt.axis('off')
        
        # Display depth images if available
        if 'depth1' in batch:
            depth1 = depth1_batch[i].squeeze().numpy()
            depth2 = depth2_batch[i].squeeze().numpy()
            
            plt.subplot(fig_rows, num_samples, i+1+2*num_samples)
            plt.imshow(depth1, cmap='viridis')
            plt.title(f"Depth 1 (idx: {frame1_idx_batch[i].item()})")
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.subplot(fig_rows, num_samples, i+1+3*num_samples)
            plt.imshow(depth2, cmap='viridis')
            plt.title(f"Depth 2 (idx: {frame2_idx_batch[i].item()})")
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
        
        # Print pose information
        quat = rel_pose_batch[i][:4].numpy()
        trans = rel_pose_batch[i][4:7].numpy()
        
        # Calculate rotation angle from quaternion (in degrees)
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        angle = np.linalg.norm(rot.as_rotvec()) * 180 / np.pi
        
        # Calculate translation magnitude
        trans_mag = np.linalg.norm(trans)
        
        print(f"Sample {i}:")
        print(f"  Frame pair: {frame1_idx_batch[i].item()} -> {frame2_idx_batch[i].item()}")
        print(f"  Quaternion (wxyz): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        print(f"  Translation (xyz): [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")
        print(f"  Rotation angle: {angle:.2f} degrees")
        print(f"  Translation magnitude: {trans_mag:.4f} meters")
        print()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Data pipeline verification saved to {save_path}")
    
    plt.show()

def main():
    """
    Main function to test the data preprocessing pipeline.
    """
    # Get the absolute path of the workspace
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(workspace_dir, "dataset_rgbd_scenes_v2")
    
    # Verify that the dataset path exists
    if not os.path.exists(dataset_path):
        potential_paths = [
            dataset_path,
            "/ssd_4TB/divake/LSF_regression/dataset_rgbd_scenes_v2",
            "../dataset_rgbd_scenes_v2",
            "../../dataset_rgbd_scenes_v2"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                dataset_path = path
                print(f"Found dataset at: {dataset_path}")
                break
        else:
            print(f"ERROR: Dataset not found at {dataset_path}")
            print("Please specify the correct path in the script")
            return
    
    # Set parameters
    scene_id = "02"
    batch_size = 4
    use_depth = True  # Set to True to include depth information
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization/outputs")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        scene_id=scene_id,
        batch_size=batch_size,
        use_depth=use_depth
    )
    
    print(f"Created data loaders:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # Verify the data pipeline
    verify_data_pipeline(
        train_loader,
        save_path=os.path.join(output_dir, "data_pipeline_samples.png")
    )
    
    # Also verify the validation and test dataloaders
    print("\nVerifying validation loader:")
    verify_data_pipeline(
        val_loader,
        save_path=os.path.join(output_dir, "validation_samples.png")
    )
    
    print("\nVerifying test loader:")
    verify_data_pipeline(
        test_loader,
        save_path=os.path.join(output_dir, "test_samples.png")
    )

if __name__ == "__main__":
    main() 