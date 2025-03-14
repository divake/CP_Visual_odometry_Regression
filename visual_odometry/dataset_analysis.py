#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset analysis script for the RGB-D Scenes Dataset v2.

This script analyzes:
- RGB and depth image statistics
- Camera pose formats and distributions
- Relative pose calculations
- Dataset statistics and visualization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import re
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# Add parent directory to path to import modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set paths - update these to match your dataset location
DATASET_ROOT = "dataset_rgbd_scenes_v2"  # Update this path
IMGS_DIR = os.path.join(DATASET_ROOT, "imgs")
PC_DIR = os.path.join(DATASET_ROOT, "pc")

# Create output directory for analysis results
OUTPUT_DIR = "dataset_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def analyze_scene_structure(scene_id="01"):
    """
    Analyze the structure of a scene directory.
    
    Args:
        scene_id: Scene ID (e.g., "01")
    """
    print(f"\n=== Analyzing Scene {scene_id} Structure ===")
    
    # Check image directory
    scene_img_dir = os.path.join(IMGS_DIR, f"scene_{scene_id}")
    if not os.path.exists(scene_img_dir):
        print(f"Error: Scene image directory not found: {scene_img_dir}")
        return
    
    # Count RGB and depth images
    rgb_files = sorted(glob.glob(os.path.join(scene_img_dir, "*-color.png")))
    depth_files = sorted(glob.glob(os.path.join(scene_img_dir, "*-depth.png")))
    
    print(f"Scene {scene_id} contains:")
    print(f"  - {len(rgb_files)} RGB images")
    print(f"  - {len(depth_files)} depth images")
    
    # Check if RGB and depth images match
    if len(rgb_files) != len(depth_files):
        print(f"Warning: Number of RGB and depth images don't match!")
    
    # Check point cloud files
    pose_file = os.path.join(PC_DIR, f"{scene_id}.pose")
    ply_file = os.path.join(PC_DIR, f"{scene_id}.ply")
    label_file = os.path.join(PC_DIR, f"{scene_id}.label")
    
    print(f"Point cloud files:")
    print(f"  - Pose file exists: {os.path.exists(pose_file)}")
    print(f"  - PLY file exists: {os.path.exists(ply_file)}")
    print(f"  - Label file exists: {os.path.exists(label_file)}")
    
    # Analyze file naming pattern
    if rgb_files:
        first_rgb = os.path.basename(rgb_files[0])
        print(f"RGB image naming pattern: {first_rgb}")
    
    if depth_files:
        first_depth = os.path.basename(depth_files[0])
        print(f"Depth image naming pattern: {first_depth}")
    
    # Extract frame numbers
    frame_numbers = []
    for rgb_file in rgb_files:
        match = re.search(r'(\d+)-color\.png', os.path.basename(rgb_file))
        if match:
            frame_numbers.append(int(match.group(1)))
    
    print(f"Frame number range: {min(frame_numbers)} to {max(frame_numbers)}")
    print(f"Total frames: {len(frame_numbers)}")
    
    # Check for missing frames
    expected_frames = set(range(min(frame_numbers), max(frame_numbers) + 1))
    missing_frames = expected_frames - set(frame_numbers)
    if missing_frames:
        print(f"Warning: {len(missing_frames)} missing frames: {sorted(missing_frames)[:10]}...")
    else:
        print("No missing frames detected.")


def analyze_rgb_depth_images(scene_id="01", num_samples=5):
    """
    Analyze RGB and depth images from a scene.
    
    Args:
        scene_id: Scene ID (e.g., "01")
        num_samples: Number of sample images to analyze
    """
    print(f"\n=== Analyzing RGB and Depth Images for Scene {scene_id} ===")
    
    scene_img_dir = os.path.join(IMGS_DIR, f"scene_{scene_id}")
    rgb_files = sorted(glob.glob(os.path.join(scene_img_dir, "*-color.png")))
    depth_files = sorted(glob.glob(os.path.join(scene_img_dir, "*-depth.png")))
    
    if not rgb_files or not depth_files:
        print("No images found!")
        return
    
    # Select sample indices
    total_frames = min(len(rgb_files), len(depth_files))
    sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    # Analyze samples
    rgb_stats = []
    depth_stats = []
    
    for idx in sample_indices:
        rgb_path = rgb_files[idx]
        depth_path = depth_files[idx]
        
        # Load RGB image
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Load depth image
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        # Compute RGB statistics
        rgb_mean = np.mean(rgb_img, axis=(0, 1))
        rgb_std = np.std(rgb_img, axis=(0, 1))
        rgb_min = np.min(rgb_img, axis=(0, 1))
        rgb_max = np.max(rgb_img, axis=(0, 1))
        
        # Compute depth statistics
        depth_mean = np.mean(depth_img)
        depth_std = np.std(depth_img)
        depth_min = np.min(depth_img)
        depth_max = np.max(depth_img)
        
        # Count valid depth pixels (non-zero)
        valid_depth_pixels = np.count_nonzero(depth_img)
        total_pixels = depth_img.shape[0] * depth_img.shape[1]
        valid_depth_percentage = (valid_depth_pixels / total_pixels) * 100
        
        rgb_stats.append({
            'frame': os.path.basename(rgb_path),
            'shape': rgb_img.shape,
            'mean': rgb_mean,
            'std': rgb_std,
            'min': rgb_min,
            'max': rgb_max
        })
        
        depth_stats.append({
            'frame': os.path.basename(depth_path),
            'shape': depth_img.shape,
            'mean': depth_mean,
            'std': depth_std,
            'min': depth_min,
            'max': depth_max,
            'valid_percentage': valid_depth_percentage
        })
        
        # Visualize one sample
        if idx == sample_indices[0]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # RGB image
            axes[0].imshow(rgb_img)
            axes[0].set_title(f"RGB Image: {os.path.basename(rgb_path)}")
            axes[0].axis('off')
            
            # Depth image (normalized for visualization)
            depth_viz = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            depth_viz = cv2.applyColorMap(depth_viz.astype(np.uint8), cv2.COLORMAP_JET)
            depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_BGR2RGB)
            axes[1].imshow(depth_viz)
            axes[1].set_title(f"Depth Image: {os.path.basename(depth_path)}")
            axes[1].axis('off')
            
            # Depth histogram
            valid_depths = depth_img[depth_img > 0].flatten()
            axes[2].hist(valid_depths, bins=50)
            axes[2].set_title("Depth Histogram")
            axes[2].set_xlabel("Depth Value")
            axes[2].set_ylabel("Frequency")
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"scene_{scene_id}_sample_images.png"))
            plt.close()
    
    # Print statistics
    print("\nRGB Image Statistics:")
    print(f"  - Resolution: {rgb_stats[0]['shape']}")
    print(f"  - Mean RGB: {np.mean([s['mean'] for s in rgb_stats], axis=0)}")
    print(f"  - Std RGB: {np.mean([s['std'] for s in rgb_stats], axis=0)}")
    print(f"  - Min RGB: {np.min([s['min'] for s in rgb_stats], axis=0)}")
    print(f"  - Max RGB: {np.max([s['max'] for s in rgb_stats], axis=0)}")
    
    print("\nDepth Image Statistics:")
    print(f"  - Resolution: {depth_stats[0]['shape']}")
    print(f"  - Mean Depth: {np.mean([s['mean'] for s in depth_stats])}")
    print(f"  - Std Depth: {np.mean([s['std'] for s in depth_stats])}")
    print(f"  - Min Depth: {np.min([s['min'] for s in depth_stats])}")
    print(f"  - Max Depth: {np.max([s['max'] for s in depth_stats])}")
    print(f"  - Valid Depth Percentage: {np.mean([s['valid_percentage'] for s in depth_stats]):.2f}%")
    
    # Save statistics to JSON
    with open(os.path.join(OUTPUT_DIR, f"scene_{scene_id}_image_stats.json"), 'w') as f:
        json.dump({
            'rgb_stats': rgb_stats,
            'depth_stats': depth_stats
        }, f, indent=4, cls=NumpyEncoder)


def analyze_pose_file(scene_id="01"):
    """
    Analyze the pose file for a scene.
    
    Args:
        scene_id: Scene ID (e.g., "01")
    """
    print(f"\n=== Analyzing Pose File for Scene {scene_id} ===")
    
    pose_file = os.path.join(PC_DIR, f"{scene_id}.pose")
    if not os.path.exists(pose_file):
        print(f"Error: Pose file not found: {pose_file}")
        return
    
    # Read pose file
    poses = []
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        
        print(f"Pose file contains {len(lines)} lines")
        
        for i, line in enumerate(lines):
            if i < 3:  # Print first few lines to understand format
                print(f"Line {i+1}: {line.strip()}")
            
            # Parse pose data (format: a b c d x y z)
            # Where a b c d is quaternion and x y z is position
            values = line.strip().split()
            if len(values) == 7:  # Quaternion + position format
                qw, qx, qy, qz = float(values[0]), float(values[1]), float(values[2]), float(values[3])
                tx, ty, tz = float(values[4]), float(values[5]), float(values[6])
                
                # Create 4x4 transformation matrix
                quat = np.array([qw, qx, qy, qz])
                quat = quat / np.linalg.norm(quat)  # Normalize quaternion
                
                # Convert quaternion to rotation matrix
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy uses [x,y,z,w] order
                
                # Create transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                
                poses.append({
                    'quaternion': quat,
                    'position': np.array([tx, ty, tz]),
                    'matrix': T
                })
    
    if not poses:
        print("No valid poses found!")
        return
    
    print(f"Successfully parsed {len(poses)} poses")
    
    # Analyze pose data
    positions = np.array([pose['position'] for pose in poses])
    quaternions = np.array([pose['quaternion'] for pose in poses])
    
    # Convert to Euler angles for analysis
    euler_angles = []
    for pose in poses:
        R = pose['matrix'][:3, :3]
        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        euler_angles.append(euler)
    
    euler_angles = np.array(euler_angles)
    
    # Compute statistics
    pos_mean = np.mean(positions, axis=0)
    pos_std = np.std(positions, axis=0)
    pos_min = np.min(positions, axis=0)
    pos_max = np.max(positions, axis=0)
    
    quat_mean = np.mean(quaternions, axis=0)
    quat_std = np.std(quaternions, axis=0)
    
    euler_mean = np.mean(euler_angles, axis=0)
    euler_std = np.std(euler_angles, axis=0)
    euler_min = np.min(euler_angles, axis=0)
    euler_max = np.max(euler_angles, axis=0)
    
    print("\nPosition Statistics (x, y, z):")
    print(f"  - Mean: {pos_mean}")
    print(f"  - Std: {pos_std}")
    print(f"  - Min: {pos_min}")
    print(f"  - Max: {pos_max}")
    print(f"  - Range: {pos_max - pos_min}")
    
    print("\nQuaternion Statistics (w, x, y, z):")
    print(f"  - Mean: {quat_mean}")
    print(f"  - Std: {quat_std}")
    
    print("\nRotation Statistics (Euler angles in degrees):")
    print(f"  - Mean: {euler_mean}")
    print(f"  - Std: {euler_std}")
    print(f"  - Min: {euler_min}")
    print(f"  - Max: {euler_max}")
    print(f"  - Range: {euler_max - euler_min}")
    
    # Compute relative poses between consecutive frames
    relative_poses = []
    for i in range(1, len(poses)):
        # T_rel = inv(T_prev) * T_curr
        T_prev_inv = np.linalg.inv(poses[i-1]['matrix'])
        T_rel = np.matmul(T_prev_inv, poses[i]['matrix'])
        
        # Extract relative position
        rel_pos = T_rel[:3, 3]
        
        # Extract relative rotation as quaternion
        rel_rot_matrix = T_rel[:3, :3]
        rel_quat = Rotation.from_matrix(rel_rot_matrix).as_quat()  # [x, y, z, w] order
        rel_quat = np.array([rel_quat[3], rel_quat[0], rel_quat[1], rel_quat[2]])  # Convert to [w, x, y, z]
        
        relative_poses.append({
            'quaternion': rel_quat,
            'position': rel_pos,
            'matrix': T_rel
        })
    
    # Analyze relative translations
    rel_positions = np.array([pose['position'] for pose in relative_poses])
    rel_pos_norms = np.linalg.norm(rel_positions, axis=1)
    
    # Analyze relative rotations
    rel_quaternions = np.array([pose['quaternion'] for pose in relative_poses])
    
    # Convert to Euler angles for analysis
    rel_euler_angles = []
    for pose in relative_poses:
        R = pose['matrix'][:3, :3]
        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        rel_euler_angles.append(euler)
    
    rel_euler_angles = np.array(rel_euler_angles)
    rel_rot_norms = np.linalg.norm(rel_euler_angles, axis=1)
    
    print("\nRelative Translation Statistics:")
    print(f"  - Mean norm: {np.mean(rel_pos_norms)}")
    print(f"  - Std norm: {np.std(rel_pos_norms)}")
    print(f"  - Min norm: {np.min(rel_pos_norms)}")
    print(f"  - Max norm: {np.max(rel_pos_norms)}")
    
    print("\nRelative Rotation Statistics (Euler angles norm in degrees):")
    print(f"  - Mean norm: {np.mean(rel_rot_norms)}")
    print(f"  - Std norm: {np.std(rel_rot_norms)}")
    print(f"  - Min norm: {np.min(rel_rot_norms)}")
    print(f"  - Max norm: {np.max(rel_rot_norms)}")
    
    # Visualize trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Camera Trajectory')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Camera Trajectory for Scene {scene_id}')
    ax.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"scene_{scene_id}_trajectory.png"))
    plt.close()
    
    # Save relative pose statistics for model training reference
    rel_pose_stats = {
        'translation': {
            'mean': np.mean(rel_positions, axis=0),
            'std': np.std(rel_positions, axis=0),
            'min': np.min(rel_positions, axis=0),
            'max': np.max(rel_positions, axis=0),
            'norm_mean': np.mean(rel_pos_norms),
            'norm_std': np.std(rel_pos_norms)
        },
        'quaternion': {
            'mean': np.mean(rel_quaternions, axis=0),
            'std': np.std(rel_quaternions, axis=0)
        },
        'euler_angles': {
            'mean': np.mean(rel_euler_angles, axis=0),
            'std': np.std(rel_euler_angles, axis=0),
            'min': np.min(rel_euler_angles, axis=0),
            'max': np.max(rel_euler_angles, axis=0),
            'norm_mean': np.mean(rel_rot_norms),
            'norm_std': np.std(rel_rot_norms)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, f"scene_{scene_id}_relative_pose_stats.json"), 'w') as f:
        json.dump(rel_pose_stats, f, indent=4, cls=NumpyEncoder)


def analyze_consecutive_frames(scene_id="01", num_samples=5):
    """
    Analyze consecutive frames to understand frame-to-frame changes.
    
    Args:
        scene_id: Scene ID (e.g., "01")
        num_samples: Number of consecutive frame pairs to analyze
    """
    print(f"\n=== Analyzing Consecutive Frames for Scene {scene_id} ===")
    
    scene_img_dir = os.path.join(IMGS_DIR, f"scene_{scene_id}")
    rgb_files = sorted(glob.glob(os.path.join(scene_img_dir, "*-color.png")))
    depth_files = sorted(glob.glob(os.path.join(scene_img_dir, "*-depth.png")))
    
    if len(rgb_files) < 2 or len(depth_files) < 2:
        print("Not enough frames for consecutive analysis!")
        return
    
    # Read pose file to get ground truth relative poses
    pose_file = os.path.join(PC_DIR, f"{scene_id}.pose")
    if not os.path.exists(pose_file):
        print(f"Error: Pose file not found: {pose_file}")
        return
    
    # Parse pose file
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 7:  # Quaternion + position format
                qw, qx, qy, qz = float(values[0]), float(values[1]), float(values[2]), float(values[3])
                tx, ty, tz = float(values[4]), float(values[5]), float(values[6])
                
                # Create 4x4 transformation matrix
                quat = np.array([qw, qx, qy, qz])
                quat = quat / np.linalg.norm(quat)  # Normalize quaternion
                
                # Convert quaternion to rotation matrix
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy uses [x,y,z,w] order
                
                # Create transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                
                poses.append(T)
    
    # Select sample indices
    total_frames = min(len(rgb_files), len(depth_files), len(poses))
    if total_frames < 2:
        print("Not enough frames with poses!")
        return
    
    # Choose evenly spaced samples
    sample_indices = np.linspace(0, total_frames-2, num_samples, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        # Get consecutive frames
        rgb1_path = rgb_files[idx]
        rgb2_path = rgb_files[idx+1]
        depth1_path = depth_files[idx]
        depth2_path = depth_files[idx+1]
        
        # Load images
        rgb1 = cv2.imread(rgb1_path)
        rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
        
        rgb2 = cv2.imread(rgb2_path)
        rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB)
        
        depth1 = cv2.imread(depth1_path, cv2.IMREAD_ANYDEPTH)
        depth2 = cv2.imread(depth2_path, cv2.IMREAD_ANYDEPTH)
        
        # Get relative pose between frames
        T1 = poses[idx]
        T2 = poses[idx+1]
        T_rel = np.linalg.inv(T1) @ T2
        
        # Extract relative translation and rotation
        rel_trans = T_rel[:3, 3]
        rel_rot_matrix = T_rel[:3, :3]
        rel_rot_euler = Rotation.from_matrix(rel_rot_matrix).as_euler('xyz', degrees=True)
        rel_rot_quat = Rotation.from_matrix(rel_rot_matrix).as_quat()  # [x, y, z, w] order
        rel_rot_quat = np.array([rel_rot_quat[3], rel_rot_quat[0], rel_rot_quat[1], rel_rot_quat[2]])  # Convert to [w, x, y, z]
        
        # Print information
        print(f"\nConsecutive Frames {idx} and {idx+1}:")
        print(f"  - RGB Files: {os.path.basename(rgb1_path)} -> {os.path.basename(rgb2_path)}")
        print(f"  - Relative Translation: {rel_trans}, Norm: {np.linalg.norm(rel_trans)}")
        print(f"  - Relative Rotation (Euler): {rel_rot_euler}, Norm: {np.linalg.norm(rel_rot_euler)}")
        print(f"  - Relative Rotation (Quaternion): {rel_rot_quat}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # RGB images
        axes[0, 0].imshow(rgb1)
        axes[0, 0].set_title(f"RGB Frame {idx}")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(rgb2)
        axes[0, 1].set_title(f"RGB Frame {idx+1}")
        axes[0, 1].axis('off')
        
        # Depth images
        depth1_viz = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX)
        depth1_viz = cv2.applyColorMap(depth1_viz.astype(np.uint8), cv2.COLORMAP_JET)
        depth1_viz = cv2.cvtColor(depth1_viz, cv2.COLOR_BGR2RGB)
        
        depth2_viz = cv2.normalize(depth2, None, 0, 255, cv2.NORM_MINMAX)
        depth2_viz = cv2.applyColorMap(depth2_viz.astype(np.uint8), cv2.COLORMAP_JET)
        depth2_viz = cv2.cvtColor(depth2_viz, cv2.COLOR_BGR2RGB)
        
        axes[1, 0].imshow(depth1_viz)
        axes[1, 0].set_title(f"Depth Frame {idx}")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(depth2_viz)
        axes[1, 1].set_title(f"Depth Frame {idx+1}")
        axes[1, 1].axis('off')
        
        # Add relative pose information
        plt.figtext(0.5, 0.01, 
                   f"Relative Translation: [{rel_trans[0]:.4f}, {rel_trans[1]:.4f}, {rel_trans[2]:.4f}], Norm: {np.linalg.norm(rel_trans):.4f}\n"
                   f"Relative Rotation (Euler): [{rel_rot_euler[0]:.2f}°, {rel_rot_euler[1]:.2f}°, {rel_rot_euler[2]:.2f}°], Norm: {np.linalg.norm(rel_rot_euler):.2f}°",
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(OUTPUT_DIR, f"scene_{scene_id}_consecutive_frames_{idx}_{idx+1}.png"))
        plt.close()


def main():
    """Main function to run all analyses."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Select a scene to analyze
    scene_id = "01"  # Can be changed to analyze different scenes
    
    # Run analyses
    analyze_scene_structure(scene_id)
    analyze_rgb_depth_images(scene_id)
    analyze_pose_file(scene_id)
    analyze_consecutive_frames(scene_id)
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main() 