#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.cm import get_cmap
import sys
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.pose_utils import quaternion_to_matrix

def plot_trajectory(poses, ax=None, color='b', label=None, plot_points=True, plot_path=True, 
                   marker_size=30, add_axis=False, axis_length=0.1, linewidth=1):
    """
    Plot a 3D trajectory.
    
    Args:
        poses (numpy.ndarray): Array of poses (each pose is a 7-element array: quaternion and translation)
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        color (str, optional): Color of the trajectory
        label (str, optional): Label for the trajectory
        plot_points (bool, optional): Whether to plot points
        plot_path (bool, optional): Whether to plot the path
        marker_size (int, optional): Size of markers
        add_axis (bool, optional): Whether to add coordinate axes at the start point
        axis_length (float, optional): Length of coordinate axes
        linewidth (int, optional): Width of the trajectory line
        
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    # Create axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Extract translations
    translations = np.array([pose[4:7] for pose in poses])
    
    # Plot the path
    if plot_path:
        ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], 
                color=color, linewidth=linewidth, label=label)
    
    # Plot points
    if plot_points:
        ax.scatter(translations[:, 0], translations[:, 1], translations[:, 2], 
                  c=color, marker='o', s=marker_size)
    
    # Add start and end markers
    ax.scatter(translations[0, 0], translations[0, 1], translations[0, 2], 
              c='g', marker='o', s=marker_size*2, label='Start')
    ax.scatter(translations[-1, 0], translations[-1, 1], translations[-1, 2], 
              c='r', marker='o', s=marker_size*2, label='End')
    
    # Add coordinate axes at the start point
    if add_axis:
        origin = translations[0]
        # Get the orientation of the first pose
        R = quaternion_to_matrix(poses[0][:4])
        
        # X axis (red)
        ax.quiver(origin[0], origin[1], origin[2], 
                 R[0, 0] * axis_length, R[1, 0] * axis_length, R[2, 0] * axis_length, 
                 color='r', arrow_length_ratio=0.1)
        
        # Y axis (green)
        ax.quiver(origin[0], origin[1], origin[2], 
                 R[0, 1] * axis_length, R[1, 1] * axis_length, R[2, 1] * axis_length, 
                 color='g', arrow_length_ratio=0.1)
        
        # Z axis (blue)
        ax.quiver(origin[0], origin[1], origin[2], 
                 R[0, 2] * axis_length, R[1, 2] * axis_length, R[2, 2] * axis_length, 
                 color='b', arrow_length_ratio=0.1)
    
    return ax

def plot_trajectories(gt_poses, est_poses=None, labels=None, title="Camera Trajectories", 
                     save_path=None, figsize=(12, 10), equal_aspect=True):
    """
    Plot ground truth and estimated trajectories for comparison.
    
    Args:
        gt_poses (numpy.ndarray): Array of ground truth poses
        est_poses (list, optional): List of arrays of estimated poses
        labels (list, optional): List of labels for estimated poses
        title (str, optional): Title of the plot
        save_path (str, optional): Path to save the figure
        figsize (tuple, optional): Figure size
        equal_aspect (bool, optional): Whether to use equal aspect ratio for all axes
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth trajectory
    plot_trajectory(gt_poses, ax=ax, color='b', label='Ground Truth', 
                   plot_points=False, linewidth=2)
    
    # Plot estimated trajectories
    if est_poses is not None:
        if not isinstance(est_poses[0], list) and not isinstance(est_poses[0], np.ndarray):
            est_poses = [est_poses]
        
        if labels is None:
            labels = [f"Estimated {i+1}" for i in range(len(est_poses))]
        
        colors = ['r', 'g', 'c', 'm', 'y', 'k']
        for i, poses in enumerate(est_poses):
            plot_trajectory(poses, ax=ax, color=colors[i % len(colors)], 
                           label=labels[i], plot_points=False, linewidth=2)
    
    # Setup axes
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    ax.legend()
    
    if equal_aspect:
        # Set equal aspect ratio
        translations = np.array([pose[4:7] for pose in gt_poses])
        max_range = np.max([
            np.max(translations[:, 0]) - np.min(translations[:, 0]),
            np.max(translations[:, 1]) - np.min(translations[:, 1]),
            np.max(translations[:, 2]) - np.min(translations[:, 2])
        ])
        
        mid_x = (np.max(translations[:, 0]) + np.min(translations[:, 0])) * 0.5
        mid_y = (np.max(translations[:, 1]) + np.min(translations[:, 1])) * 0.5
        mid_z = (np.max(translations[:, 2]) + np.min(translations[:, 2])) * 0.5
        
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory visualization saved to {save_path}")
    
    return fig

def animate_trajectory(poses, interval=50, figsize=(10, 8), save_path=None, fps=10, dpi=100):
    """
    Create an animation of a camera moving along a trajectory.
    
    Args:
        poses (numpy.ndarray): Array of poses (each pose is a 7-element array: quaternion and translation)
        interval (int, optional): Interval between frames in milliseconds
        figsize (tuple, optional): Figure size
        save_path (str, optional): Path to save the animation
        fps (int, optional): Frames per second for the saved animation
        dpi (int, optional): DPI for the saved animation
        
    Returns:
        matplotlib.animation.Animation: The animation object
    """
    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract translations
    translations = np.array([pose[4:7] for pose in poses])
    
    # Set up the plot
    line, = ax.plot([], [], [], 'b-', linewidth=2)
    point, = ax.plot([], [], [], 'ro', markersize=8)
    
    # Add start and end markers
    ax.scatter(translations[0, 0], translations[0, 1], translations[0, 2], 
              c='g', marker='o', s=50, label='Start')
    ax.scatter(translations[-1, 0], translations[-1, 1], translations[-1, 2], 
              c='r', marker='o', s=50, label='End')
    
    # Setup axes
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Camera Trajectory Animation')
    
    # Set equal aspect ratio
    max_range = np.max([
        np.max(translations[:, 0]) - np.min(translations[:, 0]),
        np.max(translations[:, 1]) - np.min(translations[:, 1]),
        np.max(translations[:, 2]) - np.min(translations[:, 2])
    ])
    
    mid_x = (np.max(translations[:, 0]) + np.min(translations[:, 0])) * 0.5
    mid_y = (np.max(translations[:, 1]) + np.min(translations[:, 1])) * 0.5
    mid_z = (np.max(translations[:, 2]) + np.min(translations[:, 2])) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    # Add coordinate axes
    axis_length = max_range * 0.05
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point
    
    def update(frame):
        # Update trajectory line
        line.set_data(translations[:frame+1, 0], translations[:frame+1, 1])
        line.set_3d_properties(translations[:frame+1, 2])
        
        # Update current position point
        point.set_data([translations[frame, 0]], [translations[frame, 1]])
        point.set_3d_properties([translations[frame, 2]])
        
        return line, point
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(poses),
                                  init_func=init, blit=True, interval=interval)
    
    # Save animation if requested
    if save_path:
        anim.save(save_path, fps=fps, dpi=dpi)
        print(f"Animation saved to {save_path}")
    
    plt.tight_layout()
    
    return anim

def plot_error_metrics(errors, labels=None, title="Error Metrics", save_path=None, figsize=(10, 6)):
    """
    Plot error metrics over time or distance.
    
    Args:
        errors (list): List of error arrays
        labels (list, optional): List of labels for each error array
        title (str, optional): Title of the plot
        save_path (str, optional): Path to save the figure
        figsize (tuple, optional): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x values (frame indices)
    x = np.arange(len(errors[0]))
    
    # Plot errors
    for i, error in enumerate(errors):
        label = labels[i] if labels and i < len(labels) else f"Error {i+1}"
        ax.plot(x, error, label=label)
    
    # Setup axes
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Error')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Error metrics plot saved to {save_path}")
    
    return fig

def visualize_paired_images(img1, img2, depth1=None, depth2=None, rel_pose=None, 
                           title=None, save_path=None, figsize=(12, 8)):
    """
    Visualize a pair of images with optional depth maps and relative pose.
    
    Args:
        img1 (numpy.ndarray): First RGB image
        img2 (numpy.ndarray): Second RGB image
        depth1 (numpy.ndarray, optional): First depth image
        depth2 (numpy.ndarray, optional): Second depth image
        rel_pose (numpy.ndarray, optional): Relative pose
        title (str, optional): Title of the plot
        save_path (str, optional): Path to save the figure
        figsize (tuple, optional): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Determine number of rows based on inputs
    rows = 1 if depth1 is None else 2
    
    # Create figure
    fig, axs = plt.subplots(rows, 2, figsize=figsize)
    
    # Handle the case when rows=1
    if rows == 1:
        axs = np.array([axs])
    
    # Plot RGB images
    axs[0, 0].imshow(img1)
    axs[0, 0].set_title("Frame 1")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(img2)
    axs[0, 1].set_title("Frame 2")
    axs[0, 1].axis('off')
    
    # Plot depth images if available
    if depth1 is not None and depth2 is not None:
        axs[1, 0].imshow(depth1, cmap='viridis')
        axs[1, 0].set_title("Depth 1")
        axs[1, 0].axis('off')
        plt.colorbar(axs[1, 0].imshow(depth1, cmap='viridis'), ax=axs[1, 0], fraction=0.046, pad=0.04)
        
        axs[1, 1].imshow(depth2, cmap='viridis')
        axs[1, 1].set_title("Depth 2")
        axs[1, 1].axis('off')
        plt.colorbar(axs[1, 1].imshow(depth2, cmap='viridis'), ax=axs[1, 1], fraction=0.046, pad=0.04)
    
    # Add title and relative pose information
    if rel_pose is not None:
        pose_str = f"Quaternion: [{rel_pose[0]:.4f}, {rel_pose[1]:.4f}, {rel_pose[2]:.4f}, {rel_pose[3]:.4f}]\n"
        pose_str += f"Translation: [{rel_pose[4]:.4f}, {rel_pose[5]:.4f}, {rel_pose[6]:.4f}]"
        
        if title:
            title = f"{title}\n{pose_str}"
        else:
            title = pose_str
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Image pair visualization saved to {save_path}")
    
    return fig

def visualize_matches(img1, img2, keypoints1, keypoints2, matches, mask=None,
                    title=None, save_path=None, figsize=(12, 8)):
    """
    Visualize feature matches between two images.
    
    Args:
        img1 (numpy.ndarray): First RGB image
        img2 (numpy.ndarray): Second RGB image
        keypoints1 (list): List of keypoints in the first image
        keypoints2 (list): List of keypoints in the second image
        matches (list): List of matches
        mask (numpy.ndarray, optional): Mask for good matches
        title (str, optional): Title of the plot
        save_path (str, optional): Path to save the figure
        figsize (tuple, optional): Figure size
        
    Returns:
        numpy.ndarray: The match visualization image
    """
    # Create match visualization
    if mask is not None:
        # Keep only good matches
        matchesMask = mask.ravel().tolist()
        draw_params = dict(
            matchColor=(0, 255, 0),  # Green for good matches
            singlePointColor=None,
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT
        )
    else:
        # Draw all matches
        matchesMask = None
        draw_params = dict(
            matchColor=(0, 255, 0),  # Green color for all matches
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_DEFAULT
        )
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)
    
    # Convert to RGB for matplotlib
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=figsize)
    plt.imshow(img_matches)
    plt.axis('off')
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Match visualization saved to {save_path}")
    
    return img_matches

def plot_error_distribution(errors, xlabel='Error', title='Error Distribution', 
                          bins=30, save_path=None, figsize=(10, 6)):
    """
    Plot the distribution of errors.
    
    Args:
        errors (numpy.ndarray): Array of errors
        xlabel (str, optional): Label for the x-axis
        title (str, optional): Title of the plot
        bins (int, optional): Number of bins for the histogram
        save_path (str, optional): Path to save the figure
        figsize (tuple, optional): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(errors, bins=bins)
    
    # Add mean and median lines
    mean = np.mean(errors)
    median = np.median(errors)
    
    ax.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='g', linestyle='--', label=f'Median: {median:.4f}')
    
    # Setup axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Error distribution plot saved to {save_path}")
    
    return fig

def visualize_odometry_results(gt_poses, est_poses, trans_errors, rot_errors,
                             output_dir, prefix='results'):
    """
    Visualize odometry results including trajectory plots and error metrics.
    
    Args:
        gt_poses (numpy.ndarray): Ground truth poses
        est_poses (numpy.ndarray): Estimated poses
        trans_errors (numpy.ndarray): Translational errors
        rot_errors (numpy.ndarray): Rotational errors
        output_dir (str): Directory to save visualizations
        prefix (str, optional): Prefix for output files
        
    Returns:
        dict: Dictionary of created figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create result dictionary
    figures = {}
    
    # 1. Plot trajectories
    fig_traj = plot_trajectories(
        gt_poses, 
        est_poses, 
        labels=['Ground Truth', 'Estimated'],
        title='Camera Trajectory Comparison',
        save_path=os.path.join(output_dir, f'{prefix}_trajectory.png')
    )
    figures['trajectory'] = fig_traj
    
    # 2. Plot error metrics over time
    fig_errors = plot_error_metrics(
        [trans_errors, rot_errors],
        labels=['Translation Error (meters)', 'Rotation Error (degrees)'],
        title='Pose Estimation Errors Over Time',
        save_path=os.path.join(output_dir, f'{prefix}_errors.png')
    )
    figures['errors'] = fig_errors
    
    # 3. Plot error distributions
    fig_trans_dist = plot_error_distribution(
        trans_errors,
        xlabel='Translation Error (meters)',
        title='Translation Error Distribution',
        save_path=os.path.join(output_dir, f'{prefix}_trans_dist.png')
    )
    figures['trans_dist'] = fig_trans_dist
    
    fig_rot_dist = plot_error_distribution(
        rot_errors,
        xlabel='Rotation Error (degrees)',
        title='Rotation Error Distribution',
        save_path=os.path.join(output_dir, f'{prefix}_rot_dist.png')
    )
    figures['rot_dist'] = fig_rot_dist
    
    # 4. Create trajectory animation
    anim = animate_trajectory(
        gt_poses,
        interval=50,
        save_path=os.path.join(output_dir, f'{prefix}_animation.mp4')
    )
    figures['animation'] = anim
    
    return figures 