#!/usr/bin/env python3
"""
explore_dataset.py - Exploratory analysis of the RGB-D Scenes Dataset v2

This script loads and analyzes data from the RGB-D Scenes Dataset v2,
with a focus on scene_02. It visualizes RGB and depth images, analyzes
camera poses, visualizes trajectories, and examines frame-to-frame motion.

Results are saved in the 'Results/dataset_exploration/' directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
from scipy.spatial.transform import Rotation as R
import pandas as pd
import seaborn as sns
from pathlib import Path

# Define paths
DATASET_PATH = "/ssd_4TB/divake/LSF_regression/dataset_rgbd_scenes_v2"
SCENE_ID = "02"
RESULTS_DIR = "../Results/dataset_exploration"
IMG_RESULTS_DIR = os.path.join(RESULTS_DIR, "images")
TRAJ_RESULTS_DIR = os.path.join(RESULTS_DIR, "trajectory_plots")
ANALYSIS_RESULTS_DIR = os.path.join(RESULTS_DIR, "analysis")

def ensure_dir(directory):
    """Ensure that a directory exists."""
    os.makedirs(directory, exist_ok=True)

def load_and_visualize_images():
    """Load and visualize sample RGB and depth images from the dataset."""
    print("\n1. Loading and visualizing RGB and depth images...")
    
    # Get all image files in scene_02
    scene_dir = os.path.join(DATASET_PATH, "imgs", f"scene_{SCENE_ID}")
    all_files = os.listdir(scene_dir)
    
    # Filter color and depth files
    color_files = sorted([f for f in all_files if f.endswith('-color.png')])
    depth_files = sorted([f for f in all_files if f.endswith('-depth.png')])
    
    # Match pairs of color and depth images
    color_frames = [int(f.split('-')[0]) for f in color_files]
    depth_frames = [int(f.split('-')[0]) for f in depth_files]
    common_frames = sorted(list(set(color_frames) & set(depth_frames)))
    
    # Select 5 sample frames evenly distributed
    if len(common_frames) > 5:
        indices = np.linspace(0, len(common_frames)-1, 5, dtype=int)
        sample_frames = [common_frames[i] for i in indices]
    else:
        sample_frames = common_frames
    
    print(f"Found {len(color_files)} color images and {len(depth_files)} depth images")
    print(f"Selected sample frames: {sample_frames}")
    
    # Visualize sample frames
    fig, axes = plt.subplots(len(sample_frames), 2, figsize=(12, 3*len(sample_frames)))
    
    for i, frame_num in enumerate(sample_frames):
        # Load color image
        color_img_path = os.path.join(scene_dir, f"{frame_num:05d}-color.png")
        color_img = cv2.imread(color_img_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        # Load depth image
        depth_img_path = os.path.join(scene_dir, f"{frame_num:05d}-depth.png")
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
        
        # Normalize depth for better visualization
        depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        # Display image dimensions and format
        color_h, color_w = color_img.shape[:2]
        depth_h, depth_w = depth_img.shape[:2]
        
        # Plot the images
        axes[i, 0].imshow(color_img)
        axes[i, 0].set_title(f"RGB Frame {frame_num} ({color_w}x{color_h})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(depth_colored)
        axes[i, 1].set_title(f"Depth Frame {frame_num} ({depth_w}x{depth_h})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_RESULTS_DIR, f"scene_{SCENE_ID}_sample_frames.png"))
    plt.close()
    
    # Save detailed statistics
    with open(os.path.join(ANALYSIS_RESULTS_DIR, "image_stats.txt"), 'w') as f:
        f.write(f"Scene {SCENE_ID} Image Statistics\n")
        f.write(f"================================\n\n")
        f.write(f"Total color images: {len(color_files)}\n")
        f.write(f"Total depth images: {len(depth_files)}\n")
        f.write(f"Total matched frame pairs: {len(common_frames)}\n\n")
        
        # Add first frame info
        color_img_path = os.path.join(scene_dir, color_files[0])
        color_img = cv2.imread(color_img_path)
        depth_img_path = os.path.join(scene_dir, depth_files[0])
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
        
        f.write(f"Color image format: {color_files[0]}, shape: {color_img.shape}, dtype: {color_img.dtype}\n")
        f.write(f"Depth image format: {depth_files[0]}, shape: {depth_img.shape}, dtype: {depth_img.dtype}\n")
        f.write(f"Depth range: min={np.min(depth_img)}, max={np.max(depth_img)}\n")
    
    print("Image analysis complete. Results saved.")

def parse_pose_file():
    """Parse the pose file for scene_02 and extract quaternions and translations."""
    print("\n2. Parsing and analyzing ground truth camera poses...")
    
    pose_file_path = os.path.join(DATASET_PATH, "pc", f"{SCENE_ID}.pose")
    poses = []
    
    with open(pose_file_path, 'r') as f:
        for line in f:
            # Each line contains: a b c d x y z (quaternion + translation)
            values = [float(v) for v in line.strip().split()]
            if len(values) == 7:
                poses.append(values)
    
    poses = np.array(poses)
    quaternions = poses[:, 0:4]  # a b c d
    translations = poses[:, 4:7]  # x y z
    
    print(f"Loaded {len(poses)} poses from {pose_file_path}")
    print(f"First pose (quaternion + translation):")
    print(poses[0])
    
    # Convert quaternions to Euler angles and rotation matrices
    euler_angles = []
    rotation_matrices = []
    
    for quat in quaternions:
        # Note: scipy's Rotation takes quaternions in scalar-last format (x,y,z,w)
        # Our data is in scalar-first format (w,x,y,z)
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler_angles.append(rot.as_euler('xyz', degrees=True))
        rotation_matrices.append(rot.as_matrix())
    
    euler_angles = np.array(euler_angles)
    rotation_matrices = np.array(rotation_matrices)
    
    # Create a DataFrame for easier analysis
    poses_df = pd.DataFrame({
        'quat_w': quaternions[:, 0],
        'quat_x': quaternions[:, 1],
        'quat_y': quaternions[:, 2],
        'quat_z': quaternions[:, 3],
        'trans_x': translations[:, 0],
        'trans_y': translations[:, 1],
        'trans_z': translations[:, 2],
        'euler_x': euler_angles[:, 0],
        'euler_y': euler_angles[:, 1],
        'euler_z': euler_angles[:, 2]
    })
    
    # Print examples and save to file
    print("\nExample quaternion, Euler angle, and translation representations:")
    for i in range(min(5, len(poses))):
        print(f"\nFrame {i}:")
        print(f"  Quaternion (w,x,y,z): {quaternions[i]}")
        print(f"  Euler angles (x,y,z): {euler_angles[i]}")
        print(f"  Translation (x,y,z): {translations[i]}")
    
    # Save the DataFrame to CSV
    poses_df.to_csv(os.path.join(ANALYSIS_RESULTS_DIR, f"scene_{SCENE_ID}_poses.csv"), index=False)
    
    # Save rotation matrices examples to a text file
    with open(os.path.join(ANALYSIS_RESULTS_DIR, f"scene_{SCENE_ID}_rotation_matrices.txt"), 'w') as f:
        f.write(f"Scene {SCENE_ID} Rotation Matrices\n")
        f.write(f"===============================\n\n")
        for i in range(min(5, len(poses))):
            f.write(f"Frame {i} Rotation Matrix:\n")
            f.write(f"{rotation_matrices[i]}\n\n")
    
    return quaternions, translations, euler_angles, rotation_matrices, poses_df

def visualize_trajectory(translations, quaternions, rotation_matrices):
    """Visualize the camera trajectory in 3D and top-down view."""
    print("\n3. Visualizing camera trajectory...")
    
    # 3D plot of the full camera path
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], 'b-', label='Camera Path')
    ax.scatter(translations[0, 0], translations[0, 1], translations[0, 2], c='g', s=100, label='Start')
    ax.scatter(translations[-1, 0], translations[-1, 1], translations[-1, 2], c='r', s=100, label='End')
    
    # Add orientation vectors at sample points along the path
    # We'll add arrows to show camera orientation
    sample_indices = np.linspace(0, len(translations)-1, 10, dtype=int)
    
    for idx in sample_indices:
        # Camera position
        pos = translations[idx]
        
        # Get the rotation matrix
        rot_mat = rotation_matrices[idx]
        
        # Camera axes (we'll use a scale factor for better visualization)
        scale = 0.1
        x_axis = rot_mat[:, 0] * scale
        y_axis = rot_mat[:, 1] * scale
        z_axis = rot_mat[:, 2] * scale
        
        # Plot the axis arrows
        ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], color='r', label='X-axis' if idx == sample_indices[0] else "")
        ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], color='g', label='Y-axis' if idx == sample_indices[0] else "")
        ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], color='b', label='Z-axis' if idx == sample_indices[0] else "")
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Camera Trajectory for Scene {SCENE_ID}')
    
    # Add a legend
    ax.legend()
    
    # Make the plot more square/cubic
    max_range = np.array([
        translations[:, 0].max() - translations[:, 0].min(),
        translations[:, 1].max() - translations[:, 1].min(),
        translations[:, 2].max() - translations[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (translations[:, 0].max() + translations[:, 0].min()) * 0.5
    mid_y = (translations[:, 1].max() + translations[:, 1].min()) * 0.5
    mid_z = (translations[:, 2].max() + translations[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(os.path.join(TRAJ_RESULTS_DIR, f"scene_{SCENE_ID}_3d_trajectory.png"))
    plt.close()
    
    # Create a top-down (X-Z plane) view of the trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(translations[:, 0], translations[:, 2], 'b-', label='Camera Path')
    plt.scatter(translations[0, 0], translations[0, 2], c='g', s=100, label='Start')
    plt.scatter(translations[-1, 0], translations[-1, 2], c='r', s=100, label='End')
    
    # Add sample orientations (arrows showing X and Z axes in the top-down view)
    for idx in sample_indices:
        pos = translations[idx]
        rot_mat = rotation_matrices[idx]
        
        # Get x and z components of the x-axis and z-axis
        scale = 0.1
        x_axis = rot_mat[:, 0] * scale
        z_axis = rot_mat[:, 2] * scale
        
        # Plot only on X-Z plane
        plt.arrow(pos[0], pos[2], x_axis[0], x_axis[2], head_width=0.02, head_length=0.03, fc='r', ec='r')
        plt.arrow(pos[0], pos[2], z_axis[0], z_axis[2], head_width=0.02, head_length=0.03, fc='b', ec='b')
    
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f'Top-Down View (X-Z Plane) of Camera Trajectory for Scene {SCENE_ID}')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(TRAJ_RESULTS_DIR, f"scene_{SCENE_ID}_top_down_trajectory.png"))
    plt.close()
    
    print("Trajectory visualization complete. Results saved.")

def analyze_consecutive_frames(quaternions, translations, euler_angles):
    """Analyze pose differences between consecutive frames."""
    print("\n4. Analyzing consecutive frame pairs...")
    
    # Calculate relative poses between consecutive frames
    rel_translations = []
    rel_rotation_angles = []
    
    for i in range(1, len(quaternions)):
        # Calculate relative translation (Euclidean distance)
        trans_diff = translations[i] - translations[i-1]
        trans_dist = np.linalg.norm(trans_diff)
        rel_translations.append(trans_dist)
        
        # Calculate relative rotation
        # Method 1: Using Euler angles
        euler_diff = np.abs(euler_angles[i] - euler_angles[i-1])
        # Normalize the differences for angles (e.g., 359° - 1° = 358°, but should be 2°)
        euler_diff = np.minimum(euler_diff, 360 - euler_diff)
        # Total angular change (simple approximation)
        total_angular_change = np.linalg.norm(euler_diff)
        rel_rotation_angles.append(total_angular_change)
    
    # Create a DataFrame for the relative motion
    rel_motion_df = pd.DataFrame({
        'frame_index': range(1, len(quaternions)),
        'translation_magnitude': rel_translations,
        'rotation_magnitude': rel_rotation_angles
    })
    
    # Save the DataFrame to CSV
    rel_motion_df.to_csv(os.path.join(ANALYSIS_RESULTS_DIR, f"scene_{SCENE_ID}_relative_motion.csv"), index=False)
    
    # Plot the distribution of translation magnitudes
    plt.figure(figsize=(10, 6))
    sns.histplot(rel_translations, kde=True)
    plt.xlabel('Translation Magnitude (distance units)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Translation Magnitudes Between Consecutive Frames in Scene {SCENE_ID}')
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_RESULTS_DIR, f"scene_{SCENE_ID}_translation_distribution.png"))
    plt.close()
    
    # Plot the distribution of rotation angles
    plt.figure(figsize=(10, 6))
    sns.histplot(rel_rotation_angles, kde=True)
    plt.xlabel('Rotation Magnitude (degrees)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Rotation Magnitudes Between Consecutive Frames in Scene {SCENE_ID}')
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_RESULTS_DIR, f"scene_{SCENE_ID}_rotation_distribution.png"))
    plt.close()
    
    # Plot how these changes throughout the sequence
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(quaternions)), rel_translations)
    plt.xlabel('Frame Index')
    plt.ylabel('Translation Magnitude')
    plt.title(f'Translation Magnitude vs. Frame Index in Scene {SCENE_ID}')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(quaternions)), rel_rotation_angles)
    plt.xlabel('Frame Index')
    plt.ylabel('Rotation Magnitude (degrees)')
    plt.title(f'Rotation Magnitude vs. Frame Index in Scene {SCENE_ID}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_RESULTS_DIR, f"scene_{SCENE_ID}_motion_over_time.png"))
    plt.close()
    
    # Basic statistics
    translation_stats = {
        'min': np.min(rel_translations),
        'max': np.max(rel_translations),
        'mean': np.mean(rel_translations),
        'median': np.median(rel_translations),
        'std': np.std(rel_translations)
    }
    
    rotation_stats = {
        'min': np.min(rel_rotation_angles),
        'max': np.max(rel_rotation_angles),
        'mean': np.mean(rel_rotation_angles),
        'median': np.median(rel_rotation_angles),
        'std': np.std(rel_rotation_angles)
    }
    
    # Save statistics to a text file
    with open(os.path.join(ANALYSIS_RESULTS_DIR, f"scene_{SCENE_ID}_motion_statistics.txt"), 'w') as f:
        f.write(f"Scene {SCENE_ID} Motion Statistics\n")
        f.write(f"=============================\n\n")
        
        f.write("Translation Statistics (units):\n")
        f.write(f"  Minimum: {translation_stats['min']:.6f}\n")
        f.write(f"  Maximum: {translation_stats['max']:.6f}\n")
        f.write(f"  Mean: {translation_stats['mean']:.6f}\n")
        f.write(f"  Median: {translation_stats['median']:.6f}\n")
        f.write(f"  Standard Deviation: {translation_stats['std']:.6f}\n\n")
        
        f.write("Rotation Statistics (degrees):\n")
        f.write(f"  Minimum: {rotation_stats['min']:.6f}\n")
        f.write(f"  Maximum: {rotation_stats['max']:.6f}\n")
        f.write(f"  Mean: {rotation_stats['mean']:.6f}\n")
        f.write(f"  Median: {rotation_stats['median']:.6f}\n")
        f.write(f"  Standard Deviation: {rotation_stats['std']:.6f}\n")
    
    print("Consecutive frame analysis complete. Results saved.")

def generate_summary_report():
    """Generate a comprehensive report with key statistics and findings."""
    print("\n5. Generating summary report...")
    
    # Create a summary report
    with open(os.path.join(RESULTS_DIR, f"scene_{SCENE_ID}_summary_report.md"), 'w') as f:
        f.write(f"# RGB-D Scenes Dataset v2 - Scene {SCENE_ID} Analysis\n\n")
        
        # Introduction
        f.write("## Introduction\n\n")
        f.write("This report presents an analysis of Scene 02 from the RGB-D Scenes Dataset v2. ")
        f.write("It includes image statistics, camera trajectory analysis, and frame-to-frame motion characteristics.\n\n")
        
        # Image Analysis
        f.write("## Image Analysis\n\n")
        f.write("The dataset includes RGB color images and depth images for each frame. ")
        f.write("Sample frames have been visualized and analyzed to understand their format and dimensions.\n\n")
        f.write("![Sample Frames](images/scene_02_sample_frames.png)\n\n")
        
        # Trajectory Analysis
        f.write("## Camera Trajectory Analysis\n\n")
        f.write("The camera trajectory has been visualized in 3D space and as a top-down view. ")
        f.write("The trajectory shows the camera's path through the scene along with orientation information at sampled points.\n\n")
        f.write("![3D Trajectory](trajectory_plots/scene_02_3d_trajectory.png)\n\n")
        f.write("![Top-Down View](trajectory_plots/scene_02_top_down_trajectory.png)\n\n")
        
        # Frame-to-Frame Motion
        f.write("## Frame-to-Frame Motion Analysis\n\n")
        f.write("Analysis of the relative motion between consecutive frames provides insights into the camera movement patterns. ")
        f.write("This includes translation and rotation magnitudes which are important for understanding the challenges in visual odometry.\n\n")
        f.write("![Motion Over Time](analysis/scene_02_motion_over_time.png)\n\n")
        f.write("![Translation Distribution](analysis/scene_02_translation_distribution.png)\n\n")
        f.write("![Rotation Distribution](analysis/scene_02_rotation_distribution.png)\n\n")
        
        # Implications for Visual Odometry
        f.write("## Implications for Visual Odometry\n\n")
        f.write("Based on the analysis, the following observations are relevant for developing a visual odometry system:\n\n")
        f.write("1. **Motion Characteristics**: The distribution of translation and rotation magnitudes informs us about the typical frame-to-frame motion.\n")
        f.write("2. **Trajectory Complexity**: The 3D trajectory visualization reveals the complexity of the camera path, which affects odometry difficulty.\n")
        f.write("3. **Image Quality**: RGB and depth image quality, including resolution and format, impacts feature matching and depth estimation.\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("This analysis provides a comprehensive understanding of Scene 02 from the RGB-D Scenes Dataset v2. ")
        f.write("The insights gained will inform the development of our visual odometry pipeline, ")
        f.write("particularly in terms of algorithm selection and parameter tuning.\n")
    
    print("Summary report generated successfully.")

def main():
    """Main function to execute all analysis tasks."""
    print("Starting RGB-D Scenes Dataset v2 exploration...")
    
    # Ensure all directories exist
    for directory in [IMG_RESULTS_DIR, TRAJ_RESULTS_DIR, ANALYSIS_RESULTS_DIR]:
        ensure_dir(directory)
    
    # Load and visualize images
    load_and_visualize_images()
    
    # Parse pose file and analyze
    quaternions, translations, euler_angles, rotation_matrices, poses_df = parse_pose_file()
    
    # Visualize trajectory
    visualize_trajectory(translations, quaternions, rotation_matrices)
    
    # Analyze consecutive frames
    analyze_consecutive_frames(quaternions, translations, euler_angles)
    
    # Generate summary report
    generate_summary_report()
    
    print("\nExploration complete! All results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main() 