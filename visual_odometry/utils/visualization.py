#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for visual odometry.

This module implements various visualization functions for the visual odometry system:
- Plotting predicted vs. ground truth trajectories in 3D
- Visualizing per-frame pose errors
- Creating error histograms
- Generating videos of the visual odometry in action
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_trajectory_3d(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    view_angles: Tuple[float, float] = (30, 45)
) -> plt.Figure:
    """
    Plot predicted and ground truth trajectories in 3D.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        view_angles: View angles (elevation, azimuth) in degrees
        
    Returns:
        Matplotlib figure
    """
    # Extract translation components
    pred_translations = pred_poses[:, 4:]
    gt_translations = gt_poses[:, 4:]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(
        gt_translations[:, 0], 
        gt_translations[:, 1], 
        gt_translations[:, 2], 
        'g-', 
        linewidth=2, 
        label='Ground Truth'
    )
    ax.plot(
        pred_translations[:, 0], 
        pred_translations[:, 1], 
        pred_translations[:, 2], 
        'r--', 
        linewidth=2, 
        label='Predicted'
    )
    
    # Plot start and end points
    ax.scatter(
        gt_translations[0, 0], 
        gt_translations[0, 1], 
        gt_translations[0, 2], 
        c='g', 
        marker='o', 
        s=100, 
        label='Start'
    )
    ax.scatter(
        gt_translations[-1, 0], 
        gt_translations[-1, 1], 
        gt_translations[-1, 2], 
        c='b', 
        marker='x', 
        s=100, 
        label='End'
    )
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Set view angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    # This ensures the plot is not distorted
    max_range = np.max([
        np.ptp(gt_translations[:, 0]),
        np.ptp(gt_translations[:, 1]),
        np.ptp(gt_translations[:, 2])
    ])
    mid_x = np.mean([np.min(gt_translations[:, 0]), np.max(gt_translations[:, 0])])
    mid_y = np.mean([np.min(gt_translations[:, 1]), np.max(gt_translations[:, 1])])
    mid_z = np.mean([np.min(gt_translations[:, 2]), np.max(gt_translations[:, 2])])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved trajectory plot to {save_path}")
    
    return fig


def plot_pose_errors(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    title: str = "Pose Errors",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot per-frame pose errors.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    # Compute translation and rotation errors
    n_poses = len(pred_poses)
    translation_errors = np.zeros(n_poses)
    rotation_errors = np.zeros(n_poses)
    
    for i in range(n_poses):
        # Translation error
        translation_errors[i] = np.linalg.norm(pred_poses[i, 4:] - gt_poses[i, 4:])
        
        # Rotation error
        q1 = pred_poses[i, :4]
        q2 = gt_poses[i, :4]
        
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute dot product
        dot_product = np.clip(np.abs(np.sum(q1 * q2)), -1.0, 1.0)
        
        # Convert to angle in degrees
        rotation_errors[i] = 2 * np.arccos(dot_product) * 180.0 / np.pi
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot translation errors
    ax1.plot(translation_errors, 'r-', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Translation Error (m)')
    ax1.set_title('Translation Errors')
    ax1.grid(True)
    
    # Plot rotation errors
    ax2.plot(rotation_errors, 'b-', linewidth=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Rotation Error (deg)')
    ax2.set_title('Rotation Errors')
    ax2.grid(True)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved pose errors plot to {save_path}")
    
    return fig


def plot_error_histograms(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    title: str = "Error Histograms",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    bins: int = 30
) -> plt.Figure:
    """
    Plot histograms of translation and rotation errors.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure
    """
    # Compute translation and rotation errors
    n_poses = len(pred_poses)
    translation_errors = np.zeros(n_poses)
    rotation_errors = np.zeros(n_poses)
    
    for i in range(n_poses):
        # Translation error
        translation_errors[i] = np.linalg.norm(pred_poses[i, 4:] - gt_poses[i, 4:])
        
        # Rotation error
        q1 = pred_poses[i, :4]
        q2 = gt_poses[i, :4]
        
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute dot product
        dot_product = np.clip(np.abs(np.sum(q1 * q2)), -1.0, 1.0)
        
        # Convert to angle in degrees
        rotation_errors[i] = 2 * np.arccos(dot_product) * 180.0 / np.pi
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot translation error histogram
    ax1.hist(translation_errors, bins=bins, color='r', alpha=0.7)
    ax1.set_xlabel('Translation Error (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Translation Error Histogram')
    ax1.grid(True)
    
    # Plot rotation error histogram
    ax2.hist(rotation_errors, bins=bins, color='b', alpha=0.7)
    ax2.set_xlabel('Rotation Error (deg)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Rotation Error Histogram')
    ax2.grid(True)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error histograms to {save_path}")
    
    return fig


def create_trajectory_animation(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    title: str = "Trajectory Animation",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    view_angles: Tuple[float, float] = (30, 45),
    fps: int = 30,
    dpi: int = 100
) -> plt.Figure:
    """
    Create an animation of the trajectory.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        title: Animation title
        save_path: Optional path to save the animation
        figsize: Figure size (width, height) in inches
        view_angles: View angles (elevation, azimuth) in degrees
        fps: Frames per second
        dpi: Dots per inch
        
    Returns:
        Matplotlib figure
    """
    # Extract translation components
    pred_translations = pred_poses[:, 4:]
    gt_translations = gt_poses[:, 4:]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Set view angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Set equal aspect ratio
    max_range = np.max([
        np.ptp(gt_translations[:, 0]),
        np.ptp(gt_translations[:, 1]),
        np.ptp(gt_translations[:, 2])
    ])
    mid_x = np.mean([np.min(gt_translations[:, 0]), np.max(gt_translations[:, 0])])
    mid_y = np.mean([np.min(gt_translations[:, 1]), np.max(gt_translations[:, 1])])
    mid_z = np.mean([np.min(gt_translations[:, 2]), np.max(gt_translations[:, 2])])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Initialize empty plots
    gt_line, = ax.plot([], [], [], 'g-', linewidth=2, label='Ground Truth')
    pred_line, = ax.plot([], [], [], 'r--', linewidth=2, label='Predicted')
    current_gt_point = ax.scatter([], [], [], c='g', marker='o', s=100)
    current_pred_point = ax.scatter([], [], [], c='r', marker='x', s=100)
    
    # Add legend
    ax.legend()
    
    # Animation update function
    def update(frame):
        # Update ground truth trajectory
        gt_line.set_data(gt_translations[:frame+1, 0], gt_translations[:frame+1, 1])
        gt_line.set_3d_properties(gt_translations[:frame+1, 2])
        
        # Update predicted trajectory
        pred_line.set_data(pred_translations[:frame+1, 0], pred_translations[:frame+1, 1])
        pred_line.set_3d_properties(pred_translations[:frame+1, 2])
        
        # Update current points
        current_gt_point._offsets3d = (
            [gt_translations[frame, 0]], 
            [gt_translations[frame, 1]], 
            [gt_translations[frame, 2]]
        )
        current_pred_point._offsets3d = (
            [pred_translations[frame, 0]], 
            [pred_translations[frame, 1]], 
            [pred_translations[frame, 2]]
        )
        
        return gt_line, pred_line, current_gt_point, current_pred_point
    
    # Create animation
    n_frames = len(pred_poses)
    anim = FuncAnimation(
        fig, 
        update, 
        frames=n_frames, 
        interval=1000/fps, 
        blit=True
    )
    
    # Save animation if path is provided
    if save_path is not None:
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
        logger.info(f"Saved trajectory animation to {save_path}")
    
    return fig


def generate_vo_video(
    rgb_images: List[np.ndarray],
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    save_path: str,
    fps: int = 30,
    figsize: Tuple[int, int] = (12, 8),
    view_angles: Tuple[float, float] = (30, 45)
) -> None:
    """
    Generate a video of the visual odometry in action.
    
    Args:
        rgb_images: List of RGB images
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        save_path: Path to save the video
        fps: Frames per second
        figsize: Figure size (width, height) in inches
        view_angles: View angles (elevation, azimuth) in degrees
    """
    # Extract translation components
    pred_translations = pred_poses[:, 4:]
    gt_translations = gt_poses[:, 4:]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = rgb_images[0].shape[:2]
    
    # Calculate figure size in pixels
    dpi = 100
    fig_width_px = int(figsize[0] * dpi)
    fig_height_px = int(figsize[1] * dpi)
    
    # Create video writer
    video_writer = cv2.VideoWriter(
        save_path, 
        fourcc, 
        fps, 
        (width + fig_width_px, max(height, fig_height_px))
    )
    
    # Process each frame
    for i in range(len(rgb_images)):
        # Create figure for trajectory
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories up to current frame
        ax.plot(
            gt_translations[:i+1, 0], 
            gt_translations[:i+1, 1], 
            gt_translations[:i+1, 2], 
            'g-', 
            linewidth=2, 
            label='Ground Truth'
        )
        ax.plot(
            pred_translations[:i+1, 0], 
            pred_translations[:i+1, 1], 
            pred_translations[:i+1, 2], 
            'r--', 
            linewidth=2, 
            label='Predicted'
        )
        
        # Plot current position
        ax.scatter(
            gt_translations[i, 0], 
            gt_translations[i, 1], 
            gt_translations[i, 2], 
            c='g', 
            marker='o', 
            s=100
        )
        ax.scatter(
            pred_translations[i, 0], 
            pred_translations[i, 1], 
            pred_translations[i, 2], 
            c='r', 
            marker='x', 
            s=100
        )
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"Frame {i}")
        
        # Set view angle
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        
        # Add legend
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.max([
            np.ptp(gt_translations[:, 0]),
            np.ptp(gt_translations[:, 1]),
            np.ptp(gt_translations[:, 2])
        ])
        mid_x = np.mean([np.min(gt_translations[:, 0]), np.max(gt_translations[:, 0])])
        mid_y = np.mean([np.min(gt_translations[:, 1]), np.max(gt_translations[:, 1])])
        mid_z = np.mean([np.min(gt_translations[:, 2]), np.max(gt_translations[:, 2])])
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert figure to image
        fig.canvas.draw()
        traj_img = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        # Convert RGBA to BGR
        traj_img = cv2.cvtColor(traj_img, cv2.COLOR_RGBA2BGR)
        
        # Resize trajectory image to match height
        if traj_img.shape[0] != height:
            scale = height / traj_img.shape[0]
            traj_img = cv2.resize(traj_img, (int(traj_img.shape[1] * scale), height))
        
        # Convert RGB image to BGR (OpenCV format)
        rgb_img = cv2.cvtColor(rgb_images[i], cv2.COLOR_RGB2BGR)
        
        # Combine images
        combined_img = np.hstack([rgb_img, traj_img])
        
        # Write frame to video
        video_writer.write(combined_img)
    
    # Release video writer
    video_writer.release()
    logger.info(f"Saved visual odometry video to {save_path}")


def save_evaluation_results(
    results: Dict[str, Dict[str, float]],
    save_path: str
) -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Dictionary with evaluation results
        save_path: Path to save the results
    """
    # Convert numpy values to Python types
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    # Convert results to Python types
    results_py = convert_to_python_types(results)
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(results_py, f, indent=4)
    
    logger.info(f"Saved evaluation results to {save_path}")


def visualize_results(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    results: Dict[str, Dict[str, float]],
    output_dir: str,
    rgb_images: Optional[List[np.ndarray]] = None
) -> None:
    """
    Generate and save all visualizations.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        results: Dictionary with evaluation results
        output_dir: Directory to save visualizations
        rgb_images: Optional list of RGB images for video generation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot trajectory
    plot_trajectory_3d(
        pred_poses, 
        gt_poses, 
        title="Trajectory Comparison", 
        save_path=os.path.join(output_dir, "trajectory.png")
    )
    
    # Plot pose errors
    plot_pose_errors(
        pred_poses, 
        gt_poses, 
        title="Pose Errors", 
        save_path=os.path.join(output_dir, "pose_errors.png")
    )
    
    # Plot error histograms
    plot_error_histograms(
        pred_poses, 
        gt_poses, 
        title="Error Histograms", 
        save_path=os.path.join(output_dir, "error_histograms.png")
    )
    
    # Create trajectory animation
    create_trajectory_animation(
        pred_poses, 
        gt_poses, 
        title="Trajectory Animation", 
        save_path=os.path.join(output_dir, "trajectory_animation.mp4")
    )
    
    # Generate video if RGB images are provided
    if rgb_images is not None:
        generate_vo_video(
            rgb_images, 
            pred_poses, 
            gt_poses, 
            save_path=os.path.join(output_dir, "vo_video.mp4")
        )
    
    # Save evaluation results
    save_evaluation_results(
        results, 
        save_path=os.path.join(output_dir, "evaluation_results.json")
    )
    
    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    # Create random poses for testing
    n_poses = 100
    
    # Ground truth poses
    gt_poses = np.zeros((n_poses, 7))
    gt_poses[:, 0] = 1.0  # qw = 1 (identity rotation)
    
    # Add some trajectory (a circle in the xy-plane)
    for i in range(n_poses):
        angle = 2 * np.pi * i / n_poses
        gt_poses[i, 4] = np.cos(angle)  # x
        gt_poses[i, 5] = np.sin(angle)  # y
        gt_poses[i, 6] = 0.01 * i       # z (small upward motion)
    
    # Predicted poses with some noise
    pred_poses = gt_poses.copy()
    
    # Add noise to translation
    translation_noise = np.random.normal(0, 0.05, (n_poses, 3))
    pred_poses[:, 4:] += translation_noise
    
    # Add noise to rotation
    rotation_noise = np.random.normal(0, 0.02, (n_poses, 4))
    pred_poses[:, :4] += rotation_noise
    
    # Normalize quaternions
    for i in range(n_poses):
        pred_poses[i, :4] = pred_poses[i, :4] / np.linalg.norm(pred_poses[i, :4])
    
    # Create dummy RGB images
    rgb_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(n_poses)]
    
    # Create output directory
    output_dir = "visualization_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot trajectory
    plot_trajectory_3d(
        pred_poses, 
        gt_poses, 
        title="Example Trajectory", 
        save_path=os.path.join(output_dir, "example_trajectory.png")
    )
    
    # Plot pose errors
    plot_pose_errors(
        pred_poses, 
        gt_poses, 
        title="Example Pose Errors", 
        save_path=os.path.join(output_dir, "example_pose_errors.png")
    )
    
    print("Example visualizations created in the 'visualization_examples' directory") 