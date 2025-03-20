#!/usr/bin/env python3
"""
Utility functions for the Visual Odometry system.

This module contains various helper functions used throughout the project:
- Quaternion conversions (to Euler angles and rotation matrices)
- Relative pose calculations
- Trajectory computation
- Error metrics calculation (ATE)
- Visualization helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from scipy.spatial.transform import Rotation as R
import os
import json
import config


def quaternion_to_euler(q):
    """
    Convert quaternions to Euler angles (in degrees).
    
    Args:
        q (numpy.ndarray): Quaternions in (w, x, y, z) format, shape (N, 4)
        
    Returns:
        numpy.ndarray: Euler angles in (x, y, z) format, shape (N, 3)
    """
    if isinstance(q, torch.Tensor):
        q = q.cpu().numpy()
    
    if q.ndim == 1:
        q = q.reshape(1, -1)
    
    # scipy's Rotation requires quaternions in scalar-last format (x,y,z,w)
    # but our quaternions are in scalar-first format (w,x,y,z), so we rearrange them
    q_scipy = np.zeros_like(q)
    q_scipy[:, 0] = q[:, 1]  # x
    q_scipy[:, 1] = q[:, 2]  # y
    q_scipy[:, 2] = q[:, 3]  # z
    q_scipy[:, 3] = q[:, 0]  # w
    
    rot = R.from_quat(q_scipy)
    euler = rot.as_euler('xyz', degrees=True)
    
    return euler


def quaternion_to_matrix(q):
    """
    Convert quaternions to rotation matrices.
    
    Args:
        q (numpy.ndarray or torch.Tensor): Quaternions in (w, x, y, z) format, shape (N, 4)
        
    Returns:
        numpy.ndarray: Rotation matrices, shape (N, 3, 3)
    """
    if isinstance(q, torch.Tensor):
        q = q.cpu().numpy()
    
    if q.ndim == 1:
        q = q.reshape(1, -1)
    
    # scipy's Rotation requires quaternions in scalar-last format (x,y,z,w)
    q_scipy = np.zeros_like(q)
    q_scipy[:, 0] = q[:, 1]  # x
    q_scipy[:, 1] = q[:, 2]  # y
    q_scipy[:, 2] = q[:, 3]  # z
    q_scipy[:, 3] = q[:, 0]  # w
    
    rot = R.from_quat(q_scipy)
    matrices = rot.as_matrix()
    
    return matrices


def euler_to_quaternion(euler, degrees=True):
    """
    Convert Euler angles to quaternions.
    
    Args:
        euler (numpy.ndarray): Euler angles in (x, y, z) format, shape (N, 3)
        degrees (bool): If True, input is in degrees, otherwise radians
        
    Returns:
        numpy.ndarray: Quaternions in (w, x, y, z) format, shape (N, 4)
    """
    if isinstance(euler, torch.Tensor):
        euler = euler.cpu().numpy()
    
    if euler.ndim == 1:
        euler = euler.reshape(1, -1)
    
    rot = R.from_euler('xyz', euler, degrees=degrees)
    q_scipy = rot.as_quat()  # Returns in scalar-last format (x,y,z,w)
    
    # Convert to scalar-first format (w,x,y,z)
    q = np.zeros_like(q_scipy)
    q[:, 0] = q_scipy[:, 3]  # w
    q[:, 1] = q_scipy[:, 0]  # x
    q[:, 2] = q_scipy[:, 1]  # y
    q[:, 3] = q_scipy[:, 2]  # z
    
    return q


def matrix_to_quaternion(matrix):
    """
    Convert rotation matrices to quaternions.
    
    Args:
        matrix (numpy.ndarray): Rotation matrices, shape (N, 3, 3)
        
    Returns:
        numpy.ndarray: Quaternions in (w, x, y, z) format, shape (N, 4)
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    
    if matrix.ndim == 2:
        matrix = matrix.reshape(1, 3, 3)
    
    rot = R.from_matrix(matrix)
    q_scipy = rot.as_quat()  # Returns in scalar-last format (x,y,z,w)
    
    # Convert to scalar-first format (w,x,y,z)
    q = np.zeros_like(q_scipy)
    q[:, 0] = q_scipy[:, 3]  # w
    q[:, 1] = q_scipy[:, 0]  # x
    q[:, 2] = q_scipy[:, 1]  # y
    q[:, 3] = q_scipy[:, 2]  # z
    
    return q


def compute_relative_pose(pose1, pose2):
    """
    Compute the relative pose from pose1 to pose2.
    
    Args:
        pose1 (numpy.ndarray): First pose as [quaternion(w,x,y,z), translation(x,y,z)], shape (7,)
        pose2 (numpy.ndarray): Second pose as [quaternion(w,x,y,z), translation(x,y,z)], shape (7,)
        
    Returns:
        numpy.ndarray: Relative pose as [quaternion(w,x,y,z), translation(x,y,z)], shape (7,)
    """
    # Extract quaternions and translations
    q1 = pose1[:4]  # w, x, y, z
    t1 = pose1[4:7]  # x, y, z
    
    q2 = pose2[:4]  # w, x, y, z
    t2 = pose2[4:7]  # x, y, z
    
    # Convert quaternions to rotation matrices
    R1 = quaternion_to_matrix(q1)[0]  # 3x3 rotation matrix
    R2 = quaternion_to_matrix(q2)[0]  # 3x3 rotation matrix
    
    # Compute relative rotation: R_rel = R2 * R1^T
    R_rel = np.dot(R2, R1.T)
    
    # Compute relative translation: t_rel = t2 - R_rel * t1
    t_rel = t2 - np.dot(R_rel, t1)
    
    # Convert relative rotation matrix back to quaternion
    q_rel = matrix_to_quaternion(R_rel)[0]  # w, x, y, z
    
    # Combine into relative pose
    relative_pose = np.concatenate([q_rel, t_rel])
    
    return relative_pose


def normalize_quaternion(q):
    """
    Normalize quaternion to unit length.
    
    Args:
        q (numpy.ndarray or torch.Tensor): Quaternion(s) to normalize, shape (N, 4) or (4,)
        
    Returns:
        numpy.ndarray or torch.Tensor: Normalized quaternion(s), same shape as input
    """
    if isinstance(q, torch.Tensor):
        norm = torch.norm(q, p=2, dim=-1, keepdim=True)
        return q / norm
    else:
        if q.ndim == 1:
            return q / np.linalg.norm(q)
        else:
            norm = np.linalg.norm(q, axis=1, keepdims=True)
            return q / norm


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (quaternion composition).
    
    Args:
        q1 (numpy.ndarray): First quaternion (w, x, y, z), shape (4,)
        q2 (numpy.ndarray): Second quaternion (w, x, y, z), shape (4,)
        
    Returns:
        numpy.ndarray: Resulting quaternion (w, x, y, z), shape (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def rotate_vector_by_quaternion(v, q):
    """
    Rotate a vector by a quaternion.
    
    Args:
        v (numpy.ndarray): Vector to rotate (x, y, z), shape (3,)
        q (numpy.ndarray): Quaternion for rotation (w, x, y, z), shape (4,)
        
    Returns:
        numpy.ndarray: Rotated vector (x, y, z), shape (3,)
    """
    # Convert vector to pure quaternion (0, x, y, z)
    v_quat = np.array([0, v[0], v[1], v[2]])
    
    # Normalize quaternion to ensure it represents a valid rotation
    q = normalize_quaternion(q)
    
    # q * v * q^-1 (where q^-1 is the conjugate for unit quaternions)
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    
    # Apply rotation: first q*v
    temp = quaternion_multiply(q, v_quat)
    
    # Then (q*v)*q^-1
    rotated_v_quat = quaternion_multiply(temp, q_inv)
    
    # Extract the vector part (x, y, z)
    rotated_v = rotated_v_quat[1:4]
    
    return rotated_v


def compute_trajectory(relative_poses, initial_pose=None):
    """
    Compute the absolute trajectory by chaining relative poses.
    
    Args:
        relative_poses (numpy.ndarray): Relative poses as [quaternion(w,x,y,z), translation(x,y,z)], shape (N, 7)
        initial_pose (numpy.ndarray, optional): Initial pose as [quaternion(w,x,y,z), translation(x,y,z)], shape (7,)
            If None, identity pose is used
            
    Returns:
        numpy.ndarray: Absolute trajectory as positions, shape (N+1, 3)
    """
    if initial_pose is None:
        # Identity pose (no rotation, no translation)
        initial_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Number of relative poses
    num_poses = len(relative_poses)
    
    # Initialize absolute poses array with initial pose
    absolute_poses = np.zeros((num_poses + 1, 7))
    absolute_poses[0] = initial_pose
    
    # Iteratively apply relative poses
    for i in range(num_poses):
        # Get the previous absolute pose
        prev_pose = absolute_poses[i]
        
        # Get the current relative pose
        rel_pose = relative_poses[i]
        
        # Compute the new absolute pose
        prev_q = prev_pose[:4]  # w, x, y, z
        prev_t = prev_pose[4:7]  # x, y, z
        
        rel_q = rel_pose[:4]  # w, x, y, z
        rel_t = rel_pose[4:7]  # x, y, z
        
        # Combine rotations: q_new = q_prev * q_rel
        new_q = quaternion_multiply(prev_q, rel_q)
        
        # Normalize the new quaternion
        new_q = normalize_quaternion(new_q)
        
        # Apply rotation to the relative translation to get it in the global frame
        R_prev = quaternion_to_matrix(prev_q)[0]  # 3x3 rotation matrix
        rotated_rel_t = np.dot(R_prev, rel_t)
        
        # Combine translations: t_new = t_prev + rotated_rel_t
        new_t = prev_t + rotated_rel_t
        
        # Set the new absolute pose
        absolute_poses[i+1, :4] = new_q
        absolute_poses[i+1, 4:7] = new_t
    
    # Extract positions (translations) from absolute poses
    positions = absolute_poses[:, 4:7]
    
    return positions


def optimize_trajectory(trajectory, relative_poses, window_size=20, iterations=3):
    """
    Optimize the trajectory by applying a windowed pose graph optimization.
    This reduces drift by enforcing local consistency constraints.
    
    Args:
        trajectory (numpy.ndarray): Initial trajectory as positions, shape (N, 3)
        relative_poses (numpy.ndarray): Relative poses used to generate the trajectory, shape (N-1, 7)
        window_size (int): Size of the sliding window for local optimization
        iterations (int): Number of optimization iterations
        
    Returns:
        numpy.ndarray: Optimized trajectory as positions, shape (N, 3)
    """
    # Make a copy of the trajectory to avoid modifying the input
    optimized_trajectory = np.copy(trajectory)
    num_points = len(trajectory)
    
    if num_points <= window_size:
        return optimized_trajectory
    
    # Extract only translations from relative poses for easier handling
    relative_translations = relative_poses[:, 4:7]
    
    # Iterative optimization
    for _ in range(iterations):
        # Use a sliding window approach
        for i in range(num_points - window_size):
            window_end = i + window_size
            
            # Extract the window
            window = optimized_trajectory[i:window_end]
            
            # Compute observed relative translations within the window
            observed_relative = np.diff(window, axis=0)
            
            # Get the predicted relative translations for this window
            predicted_relative = relative_translations[i:window_end-1]
            
            # Compute the scale factor to align the window
            # Use robust estimation - median ratio of magnitudes
            observed_magnitudes = np.linalg.norm(observed_relative, axis=1)
            predicted_magnitudes = np.linalg.norm(predicted_relative, axis=1)
            
            # Avoid division by zero
            valid_indices = (predicted_magnitudes > 1e-6) & (observed_magnitudes > 1e-6)
            if np.sum(valid_indices) > 0:
                ratios = observed_magnitudes[valid_indices] / predicted_magnitudes[valid_indices]
                scale_factor = np.median(ratios)
            else:
                scale_factor = 1.0
            
            # Apply correction to the latter part of the trajectory
            if window_end < num_points:
                # Compute the correction vector (the drift at the end of the window)
                end_drift = optimized_trajectory[window_end] - (optimized_trajectory[i] + 
                                                              np.sum(predicted_relative * scale_factor, axis=0))
                
                # Apply a gradual correction to the remainder of the trajectory
                # The correction decreases linearly with distance
                remaining_length = num_points - window_end
                for j in range(window_end, num_points):
                    weight = 1.0 - (j - window_end) / remaining_length
                    optimized_trajectory[j] -= end_drift * weight
    
    return optimized_trajectory


def align_trajectories(pred_trajectory, gt_trajectory, align_type='scale'):
    """
    Align predicted trajectory to ground truth using different types of alignment.
    
    Args:
        pred_trajectory (numpy.ndarray): Predicted trajectory, shape (N, 3)
        gt_trajectory (numpy.ndarray): Ground truth trajectory, shape (N, 3)
        align_type (str): Type of alignment to perform
            - 'none': No alignment
            - 'scale': Scale alignment only
            - 'rigid': Rigid alignment (translation + rotation)
            - 'sim3': Similarity transformation (translation + rotation + scale)
            
    Returns:
        numpy.ndarray: Aligned predicted trajectory, shape (N, 3)
    """
    if align_type == 'none':
        return pred_trajectory
    
    # Make sure inputs are numpy arrays
    if isinstance(pred_trajectory, torch.Tensor):
        pred_trajectory = pred_trajectory.cpu().numpy()
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.cpu().numpy()
    
    # Center both trajectories
    pred_centroid = np.mean(pred_trajectory, axis=0)
    gt_centroid = np.mean(gt_trajectory, axis=0)
    
    pred_centered = pred_trajectory - pred_centroid
    gt_centered = gt_trajectory - gt_centroid
    
    if align_type == 'scale':
        # Only perform scale alignment
        
        # Compute scale factor
        scale = np.sum(np.linalg.norm(gt_centered, axis=1)) / np.sum(np.linalg.norm(pred_centered, axis=1) + 1e-8)
        
        # Apply scale and translation alignment
        pred_aligned = pred_centered * scale + gt_centroid
        
    elif align_type == 'rigid' or align_type == 'sim3':
        # Compute cross-covariance matrix
        H = np.dot(pred_centered.T, gt_centered)
        
        # SVD decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # Handle special reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        if align_type == 'rigid':
            # Apply rotation and translation alignment
            pred_aligned = np.dot(pred_centered, R.T) + gt_centroid
            
        else:  # sim3
            # Compute scale
            var_pred = np.mean(np.sum(pred_centered**2, axis=1))
            var_gt = np.mean(np.sum(gt_centered**2, axis=1))
            scale = np.sqrt(var_gt / (var_pred + 1e-8))
            
            # Apply similarity transformation
            pred_aligned = scale * np.dot(pred_centered, R.T) + gt_centroid
    else:
        pred_aligned = pred_trajectory
    
    return pred_aligned


def calculate_scale_error(pred_trajectory, gt_trajectory):
    """
    Calculate the scale error between predicted and ground truth trajectories.
    
    Args:
        pred_trajectory (numpy.ndarray): Predicted trajectory, shape (N, 3)
        gt_trajectory (numpy.ndarray): Ground truth trajectory, shape (N, 3)
        
    Returns:
        tuple: (scale_ratio, scale_consistency)
            - scale_ratio: Average scale ratio between predicted and ground truth segments
            - scale_consistency: Standard deviation of scale ratios (lower is better)
    """
    # Ensure inputs are numpy arrays
    if isinstance(pred_trajectory, torch.Tensor):
        pred_trajectory = pred_trajectory.cpu().numpy()
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.cpu().numpy()
    
    # Calculate segment lengths for consecutive poses
    pred_segments = np.linalg.norm(pred_trajectory[1:] - pred_trajectory[:-1], axis=1)
    gt_segments = np.linalg.norm(gt_trajectory[1:] - gt_trajectory[:-1], axis=1)
    
    # Calculate the ratio of predicted to ground truth segment lengths
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    scale_ratios = pred_segments / (gt_segments + epsilon)
    
    # Filter out outliers (e.g., when gt_segments is very small)
    valid_ratios = scale_ratios[gt_segments > 0.01]  # Only consider segments > 1cm
    
    if len(valid_ratios) == 0:
        return 0.0, 0.0
    
    # Calculate mean scale ratio and consistency
    mean_scale_ratio = np.mean(valid_ratios)
    scale_consistency = np.std(valid_ratios)
    
    return mean_scale_ratio, scale_consistency


def calculate_ate(pred_trajectory, gt_trajectory, align_scale=True):
    """
    Calculate Absolute Trajectory Error (ATE) between predicted and ground truth trajectories.
    
    Args:
        pred_trajectory (numpy.ndarray): Predicted trajectory, shape (N, 3)
        gt_trajectory (numpy.ndarray): Ground truth trajectory, shape (N, 3)
        align_scale (bool): Whether to align trajectories by scale before computing error
        
    Returns:
        float: RMSE of ATE
    """
    # Ensure inputs are numpy arrays
    if isinstance(pred_trajectory, torch.Tensor):
        pred_trajectory = pred_trajectory.cpu().numpy()
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.cpu().numpy()
    
    # Optionally align trajectories for fair comparison
    if align_scale:
        pred_aligned = align_trajectories(pred_trajectory, gt_trajectory, align_type='scale')
    else:
        pred_aligned = pred_trajectory
    
    # Calculate error
    error = np.linalg.norm(pred_aligned - gt_trajectory, axis=1)
    rmse = np.sqrt(np.mean(error ** 2))
    
    return rmse


def save_trajectory(trajectory, filename):
    """
    Save trajectory to a CSV file.
    
    Args:
        trajectory (numpy.ndarray): Trajectory as positions, shape (N, 3)
        filename (str): Output file name (CSV)
    """
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    
    # Create directory if it doesn't exist
    output_dir = os.path.join(config.RESULTS_DIR, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    
    # Write trajectory data
    with open(output_path, 'w') as f:
        f.write("# timestamp tx ty tz\n")
        for i, pos in enumerate(trajectory):
            # Use frame index as timestamp
            f.write(f"{i} {pos[0]} {pos[1]} {pos[2]}\n")
    
    print(f"Trajectory saved to {output_path}")


def save_metrics(metrics, output_path):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics (dict): Evaluation metrics
        output_path (str): Output file path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save metrics to JSON
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def plot_trajectory_2d(gt_trajectory, pred_trajectory, pred_aligned=None, save_path=None):
    """
    Plot 2D (top-down view) trajectory comparison.
    
    Args:
        gt_trajectory (numpy.ndarray): Ground truth trajectory, shape (N, 3)
        pred_trajectory (numpy.ndarray): Predicted trajectory, shape (N, 3)
        pred_aligned (numpy.ndarray, optional): Scale-aligned predicted trajectory, shape (N, 3)
        save_path (str, optional): Path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.cpu().numpy()
    if isinstance(pred_trajectory, torch.Tensor):
        pred_trajectory = pred_trajectory.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ground truth trajectory
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], 'g-', linewidth=2, label='Ground Truth')
    
    # Plot predicted trajectory
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r-', linewidth=1, label='Prediction')
    
    # Plot aligned trajectory if provided
    if pred_aligned is not None:
        if isinstance(pred_aligned, torch.Tensor):
            pred_aligned = pred_aligned.cpu().numpy()
        ax.plot(pred_aligned[:, 0], pred_aligned[:, 2], 'b-', linewidth=1, label='Prediction (Scale-Aligned)')
    
    # Mark start and end points
    ax.plot(gt_trajectory[0, 0], gt_trajectory[0, 2], 'go', markersize=8, label='Start')
    ax.plot(gt_trajectory[-1, 0], gt_trajectory[-1, 2], 'gx', markersize=8, label='End')
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('Trajectory Comparison (Top-Down View)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # Ensure equal aspect ratio (1:1)
    ax.set_aspect('equal')
    
    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_trajectory_3d(gt_trajectory, pred_trajectory, save_path=None):
    """
    Plot 3D trajectory comparison.
    
    Args:
        gt_trajectory (numpy.ndarray): Ground truth trajectory, shape (N, 3)
        pred_trajectory (numpy.ndarray): Predicted (aligned) trajectory, shape (N, 3)
        save_path (str, optional): Path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.cpu().numpy()
    if isinstance(pred_trajectory, torch.Tensor):
        pred_trajectory = pred_trajectory.cpu().numpy()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth trajectory
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], gt_trajectory[:, 1], 'g-', linewidth=2, label='Ground Truth')
    
    # Plot predicted trajectory
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], pred_trajectory[:, 1], 'b-', linewidth=1, label='Prediction (Aligned)')
    
    # Mark start and end points
    ax.plot([gt_trajectory[0, 0]], [gt_trajectory[0, 2]], [gt_trajectory[0, 1]], 'go', markersize=8, label='Start')
    ax.plot([gt_trajectory[-1, 0]], [gt_trajectory[-1, 2]], [gt_trajectory[-1, 1]], 'gx', markersize=8, label='End')
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_zlabel('Y (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('3D Trajectory Comparison', fontsize=14)
    ax.legend(fontsize=12)
    
    # Ensure equal aspect ratio (1:1:1)
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_error_analysis(gt_trajectory, pred_trajectory, save_path=None):
    """
    Plot translation error analysis.
    
    Args:
        gt_trajectory (numpy.ndarray): Ground truth trajectory, shape (N, 3)
        pred_trajectory (numpy.ndarray): Predicted (aligned) trajectory, shape (N, 3)
        save_path (str, optional): Path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.cpu().numpy()
    if isinstance(pred_trajectory, torch.Tensor):
        pred_trajectory = pred_trajectory.cpu().numpy()
    
    # Calculate per-frame errors
    errors = np.sqrt(np.sum((pred_trajectory - gt_trajectory)**2, axis=1))
    
    # Calculate per-axis errors
    axis_errors = np.abs(pred_trajectory - gt_trajectory)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot total error over frame indices
    ax1.plot(errors, 'b-', linewidth=1.5)
    ax1.set_xlabel('Frame Index', fontsize=12)
    ax1.set_ylabel('Error (m)', fontsize=12)
    ax1.set_title('Translation Error Magnitude Per Frame', fontsize=14)
    ax1.grid(True)
    
    # Add mean and median lines
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    ax1.axhline(y=mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.4f} m')
    ax1.axhline(y=median_error, color='g', linestyle='-.', label=f'Median: {median_error:.4f} m')
    ax1.legend(fontsize=12)
    
    # Plot per-axis errors
    ax2.plot(axis_errors[:, 0], 'r-', linewidth=1, label='X Error')
    ax2.plot(axis_errors[:, 1], 'g-', linewidth=1, label='Y Error')
    ax2.plot(axis_errors[:, 2], 'b-', linewidth=1, label='Z Error')
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Error (m)', fontsize=12)
    ax2.set_title('Per-Axis Translation Error', fontsize=14)
    ax2.grid(True)
    ax2.legend(fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_loss_curves(train_losses, val_losses, train_translation_losses=None, val_translation_losses=None, output_path=None):
    """
    Plot training and validation loss curves, with optional translation loss curves.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        train_translation_losses (list, optional): List of training translation losses per epoch
        val_translation_losses (list, optional): List of validation translation losses per epoch
        output_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2 if train_translation_losses is not None else 1, figsize=(15, 6))
    
    if train_translation_losses is not None:
        # Total loss plot
        ax1 = axes[0]
        ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
        ax1.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Training and Validation Total Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Translation loss plot
        ax2 = axes[1]
        ax2.plot(range(1, len(train_translation_losses) + 1), train_translation_losses, 'g-', label='Train Translation Loss')
        ax2.plot(range(1, len(val_translation_losses) + 1), val_translation_losses, 'y-', label='Val Translation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Translation Loss')
        ax2.set_title('Training and Validation Translation Loss')
        ax2.legend()
        ax2.grid(True)
    else:
        # Just plot total loss if no translation losses provided
        axes.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
        axes.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.set_title('Training and Validation Loss')
        axes.legend()
        axes.grid(True)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close() 