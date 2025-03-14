#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depth scaling analysis for the RGB-D Scenes Dataset v2.

This script analyzes the depth values to determine the appropriate scaling factor
to convert raw depth values to metric units (meters).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from tqdm import tqdm
import json

# Add parent directory to path to import modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set paths - update these to match your dataset location
DATASET_ROOT = "dataset_rgbd_scenes_v2"  # Update this path
IMGS_DIR = os.path.join(DATASET_ROOT, "imgs")
PC_DIR = os.path.join(DATASET_ROOT, "pc")

# Create output directory for analysis results
OUTPUT_DIR = "depth_scaling_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_depth_scaling(scene_id="01"):
    """
    Analyze depth scaling by comparing depth values with camera positions.
    
    Args:
        scene_id: Scene ID (e.g., "01")
    """
    print(f"\n=== Analyzing Depth Scaling for Scene {scene_id} ===")
    
    # Check image directory
    scene_img_dir = os.path.join(IMGS_DIR, f"scene_{scene_id}")
    if not os.path.exists(scene_img_dir):
        print(f"Error: Scene image directory not found: {scene_img_dir}")
        return
    
    # Get depth images
    depth_files = sorted(glob.glob(os.path.join(scene_img_dir, "*-depth.png")))
    
    if not depth_files:
        print("No depth images found!")
        return
    
    # Read pose file
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
                tx, ty, tz = float(values[4]), float(values[5]), float(values[6])
                poses.append(np.array([tx, ty, tz]))
    
    if len(poses) != len(depth_files):
        print(f"Warning: Number of poses ({len(poses)}) doesn't match number of depth images ({len(depth_files)})!")
        return
    
    # Compute distances between consecutive camera positions
    distances = []
    for i in range(1, len(poses)):
        dist = np.linalg.norm(poses[i] - poses[i-1])
        distances.append(dist)
    
    # Analyze depth values
    depth_stats = []
    
    for i, depth_file in enumerate(tqdm(depth_files, desc="Analyzing depth images")):
        # Load depth image
        depth_img = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        
        # Compute depth statistics
        valid_depths = depth_img[depth_img > 0]
        if len(valid_depths) > 0:
            depth_mean = np.mean(valid_depths)
            depth_median = np.median(valid_depths)
            depth_min = np.min(valid_depths)
            depth_max = np.max(valid_depths)
        else:
            depth_mean = depth_median = depth_min = depth_max = 0
        
        depth_stats.append({
            'frame': os.path.basename(depth_file),
            'mean': float(depth_mean),
            'median': float(depth_median),
            'min': float(depth_min),
            'max': float(depth_max),
            'position': poses[i].tolist()
        })
    
    # Analyze relationship between depth values and camera positions
    depth_means = np.array([s['mean'] for s in depth_stats])
    depth_medians = np.array([s['median'] for s in depth_stats])
    
    # Compute distances from first camera position
    origin_distances = []
    for pose in poses:
        dist = np.linalg.norm(pose - poses[0])
        origin_distances.append(dist)
    
    # Plot depth values vs. camera distances
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(origin_distances, depth_means, 'b-', label='Mean Depth')
    plt.plot(origin_distances, depth_medians, 'r-', label='Median Depth')
    plt.xlabel('Distance from Origin (m)')
    plt.ylabel('Depth Value')
    plt.title('Depth Values vs. Camera Distance from Origin')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(distances, 'g-', label='Camera Movement')
    plt.xlabel('Frame')
    plt.ylabel('Distance (m)')
    plt.title('Distance Between Consecutive Camera Positions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"scene_{scene_id}_depth_vs_distance.png"))
    plt.close()
    
    # Estimate scaling factor
    # We'll use the relationship between depth values and real-world distances
    # This is a simplified approach and might need refinement
    
    # Method 1: Use median depth and median distance
    median_depth = np.median(depth_medians)
    median_distance = np.median(origin_distances[1:])  # Skip first frame (origin)
    
    if median_distance > 0 and median_depth > 0:
        scaling_factor_1 = median_distance / median_depth
    else:
        scaling_factor_1 = 0
    
    # Method 2: Linear regression
    # We'll fit a line to the relationship between depth values and distances
    valid_indices = np.where((np.array(origin_distances) > 0) & (depth_medians > 0))[0]
    if len(valid_indices) > 1:
        x = depth_medians[valid_indices]
        y = np.array(origin_distances)[valid_indices]
        
        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
        scaling_factor_2 = slope
    else:
        scaling_factor_2 = 0
    
    print("\nDepth Scaling Analysis Results:")
    print(f"  - Method 1 (Median Ratio): {scaling_factor_1:.8f}")
    print(f"  - Method 2 (Linear Regression): {scaling_factor_2:.8f}")
    print(f"  - Recommended scaling factor: {(scaling_factor_1 + scaling_factor_2) / 2:.8f}")
    
    # Plot depth values with estimated scaling
    plt.figure(figsize=(10, 6))
    
    plt.plot(origin_distances, 'b-', label='Actual Distance (m)')
    
    scaled_depth_1 = depth_medians * scaling_factor_1
    plt.plot(scaled_depth_1, 'r-', label=f'Scaled Depth (Method 1, factor={scaling_factor_1:.8f})')
    
    scaled_depth_2 = depth_medians * scaling_factor_2
    plt.plot(scaled_depth_2, 'g-', label=f'Scaled Depth (Method 2, factor={scaling_factor_2:.8f})')
    
    plt.xlabel('Frame')
    plt.ylabel('Distance (m)')
    plt.title('Comparison of Actual Distances and Scaled Depth Values')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"scene_{scene_id}_depth_scaling.png"))
    plt.close()
    
    # Save results
    results = {
        'scaling_factor_method1': float(scaling_factor_1),
        'scaling_factor_method2': float(scaling_factor_2),
        'recommended_scaling_factor': float((scaling_factor_1 + scaling_factor_2) / 2),
        'depth_stats': depth_stats
    }
    
    with open(os.path.join(OUTPUT_DIR, f"scene_{scene_id}_depth_scaling_results.json"), 'w') as f:
        json.dump(results, f, indent=4)


def main():
    """Main function to run depth scaling analysis."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Select a scene to analyze
    scene_id = "01"  # Can be changed to analyze different scenes
    
    # Run analysis
    analyze_depth_scaling(scene_id)
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main() 