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


def compute_trajectory(initial_pose, relative_poses):
    """
    Compute absolute trajectory from initial pose and a sequence of relative poses.
    
    Args:
        initial_pose (numpy.ndarray): Initial pose as [quaternion(w,x,y,z), translation(x,y,z)], shape (7,)
        relative_poses (numpy.ndarray): Sequence of relative poses, each as 
                                       [quaternion(w,x,y,z), translation(x,y,z)], shape (N, 7)
        
    Returns:
        numpy.ndarray: Sequence of absolute poses, each as 
                      [quaternion(w,x,y,z), translation(x,y,z)], shape (N+1, 7)
    """
    n_poses = len(relative_poses) + 1
    absolute_poses = np.zeros((n_poses, 7))
    absolute_poses[0] = initial_pose
    
    # Extract initial rotation and translation
    current_q = initial_pose[:4]  # w, x, y, z
    current_t = initial_pose[4:7]  # x, y, z
    current_R = quaternion_to_matrix(current_q)[0]  # 3x3 rotation matrix
    
    for i in range(len(relative_poses)):
        # Get relative pose
        rel_q = relative_poses[i, :4]  # w, x, y, z
        rel_t = relative_poses[i, 4:7]  # x, y, z
        rel_R = quaternion_to_matrix(rel_q)[0]  # 3x3 rotation matrix
        
        # Update rotation: R_new = rel_R * R_current
        new_R = np.dot(rel_R, current_R)
        new_q = matrix_to_quaternion(new_R)[0]  # w, x, y, z
        
        # Update translation: t_new = t_current + R_current * rel_t
        new_t = current_t + np.dot(current_R, rel_t)
        
        # Update current pose
        current_q = new_q
        current_t = new_t
        current_R = new_R
        
        # Store the new absolute pose
        absolute_poses[i+1, :4] = current_q
        absolute_poses[i+1, 4:7] = current_t
    
    return absolute_poses


def calculate_ate(gt_poses, pred_poses):
    """
    Calculate the Absolute Trajectory Error (ATE) between ground truth and predicted poses.
    
    Args:
        gt_poses (numpy.ndarray): Ground truth poses, each as 
                                 [quaternion(w,x,y,z), translation(x,y,z)], shape (N, 7)
        pred_poses (numpy.ndarray): Predicted poses, each as 
                                   [quaternion(w,x,y,z), translation(x,y,z)], shape (N, 7)
        
    Returns:
        dict: Dictionary containing ATE metrics:
              - 'translation_error_mean': Mean translation error (m)
              - 'translation_error_median': Median translation error (m)
              - 'translation_error_std': Standard deviation of translation error (m)
              - 'translation_error_rmse': Root mean square translation error (m)
              - 'rotation_error_mean': Mean rotation error (degrees)
              - 'rotation_error_median': Median rotation error (degrees)
              - 'rotation_error_std': Standard deviation of rotation error (degrees)
              - 'rotation_error_rmse': Root mean square rotation error (degrees)
    """
    if len(gt_poses) != len(pred_poses):
        raise ValueError("Ground truth and predicted pose sequences must have same length")
    
    # Calculate translation errors
    translation_errors = []
    for gt, pred in zip(gt_poses, pred_poses):
        gt_t = gt[4:7]
        pred_t = pred[4:7]
        translation_error = np.linalg.norm(gt_t - pred_t)
        translation_errors.append(translation_error)
    
    translation_errors = np.array(translation_errors)
    
    # Calculate rotation errors
    rotation_errors = []
    for gt, pred in zip(gt_poses, pred_poses):
        gt_q = gt[:4]
        pred_q = pred[:4]
        
        # Convert to Euler angles
        gt_euler = quaternion_to_euler(gt_q)[0]
        pred_euler = quaternion_to_euler(pred_q)[0]
        
        # Calculate angular differences
        euler_diff = np.abs(gt_euler - pred_euler)
        
        # Account for angle wrap-around
        euler_diff = np.minimum(euler_diff, 360 - euler_diff)
        
        # Compute angular error as the Euclidean norm of the Euler angle differences
        rotation_error = np.linalg.norm(euler_diff)
        rotation_errors.append(rotation_error)
    
    rotation_errors = np.array(rotation_errors)
    
    # Compute statistics
    metrics = {
        'translation_error_mean': float(np.mean(translation_errors)),
        'translation_error_median': float(np.median(translation_errors)),
        'translation_error_std': float(np.std(translation_errors)),
        'translation_error_rmse': float(np.sqrt(np.mean(np.square(translation_errors)))),
        'rotation_error_mean': float(np.mean(rotation_errors)),
        'rotation_error_median': float(np.median(rotation_errors)),
        'rotation_error_std': float(np.std(rotation_errors)),
        'rotation_error_rmse': float(np.sqrt(np.mean(np.square(rotation_errors))))
    }
    
    return metrics


def save_metrics(metrics, file_path):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary of metrics to save
        file_path (str): Path to save the metrics file
    """
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def save_trajectory(poses, file_path):
    """
    Save trajectory poses to a CSV file.
    
    Args:
        poses (numpy.ndarray): Sequence of poses, each as 
                               [quaternion(w,x,y,z), translation(x,y,z)], shape (N, 7)
        file_path (str): Path to save the trajectory file
    """
    header = "qw,qx,qy,qz,tx,ty,tz"
    np.savetxt(file_path, poses, delimiter=',', header=header, comments='')


def load_trajectory(file_path):
    """
    Load trajectory poses from a CSV file.
    
    Args:
        file_path (str): Path to the trajectory file
        
    Returns:
        numpy.ndarray: Sequence of poses, each as 
                      [quaternion(w,x,y,z), translation(x,y,z)], shape (N, 7)
    """
    return np.loadtxt(file_path, delimiter=',', skiprows=1)


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


def plot_trajectory_2d(gt_poses, pred_poses, output_path):
    """
    Plot ground truth and predicted trajectories in 2D (top-down view).
    
    Args:
        gt_poses (numpy.ndarray): Ground truth poses, shape (N, 7)
        pred_poses (numpy.ndarray): Predicted poses, shape (N, 7)
        output_path (str): Path to save the plot
    """
    gt_translations = gt_poses[:, 4:7]
    pred_translations = pred_poses[:, 4:7]
    
    plt.figure(figsize=(10, 8))
    
    # Plot ground truth trajectory
    plt.plot(gt_translations[:, 0], gt_translations[:, 2], 'b-', linewidth=2, label='Ground Truth')
    plt.scatter(gt_translations[0, 0], gt_translations[0, 2], c='g', s=100, label='Start')
    plt.scatter(gt_translations[-1, 0], gt_translations[-1, 2], c='r', s=100, label='End (GT)')
    
    # Plot predicted trajectory
    plt.plot(pred_translations[:, 0], pred_translations[:, 2], 'c--', linewidth=2, label='Predicted')
    plt.scatter(pred_translations[-1, 0], pred_translations[-1, 2], c='m', s=100, label='End (Pred)')
    
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('Trajectory Comparison (Top-Down View)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_trajectory_3d(gt_poses, pred_poses, output_path):
    """
    Plot ground truth and predicted trajectories in 3D.
    
    Args:
        gt_poses (numpy.ndarray): Ground truth poses, shape (N, 7)
        pred_poses (numpy.ndarray): Predicted poses, shape (N, 7)
        output_path (str): Path to save the plot
    """
    gt_translations = gt_poses[:, 4:7]
    pred_translations = pred_poses[:, 4:7]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth trajectory
    ax.plot(gt_translations[:, 0], gt_translations[:, 1], gt_translations[:, 2], 
            'b-', linewidth=2, label='Ground Truth')
    ax.scatter(gt_translations[0, 0], gt_translations[0, 1], gt_translations[0, 2], 
               c='g', s=100, label='Start')
    ax.scatter(gt_translations[-1, 0], gt_translations[-1, 1], gt_translations[-1, 2], 
               c='r', s=100, label='End (GT)')
    
    # Plot predicted trajectory
    ax.plot(pred_translations[:, 0], pred_translations[:, 1], pred_translations[:, 2], 
            'c--', linewidth=2, label='Predicted')
    ax.scatter(pred_translations[-1, 0], pred_translations[-1, 1], pred_translations[-1, 2], 
               c='m', s=100, label='End (Pred)')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Comparison')
    
    # Add legend
    ax.legend()
    
    # Make plot cubic/equal aspect ratio
    max_range = np.array([
        max(gt_translations[:, 0].max(), pred_translations[:, 0].max()) - 
        min(gt_translations[:, 0].min(), pred_translations[:, 0].min()),
        
        max(gt_translations[:, 1].max(), pred_translations[:, 1].max()) - 
        min(gt_translations[:, 1].min(), pred_translations[:, 1].min()),
        
        max(gt_translations[:, 2].max(), pred_translations[:, 2].max()) - 
        min(gt_translations[:, 2].min(), pred_translations[:, 2].min())
    ]).max() / 2.0
    
    mid_x = (max(gt_translations[:, 0].max(), pred_translations[:, 0].max()) + 
             min(gt_translations[:, 0].min(), pred_translations[:, 0].min())) * 0.5
    mid_y = (max(gt_translations[:, 1].max(), pred_translations[:, 1].max()) + 
             min(gt_translations[:, 1].min(), pred_translations[:, 1].min())) * 0.5
    mid_z = (max(gt_translations[:, 2].max(), pred_translations[:, 2].max()) + 
             min(gt_translations[:, 2].min(), pred_translations[:, 2].min())) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_error_analysis(gt_poses, pred_poses, output_path):
    """
    Plot error analysis for the trajectory comparison.
    
    Args:
        gt_poses (numpy.ndarray): Ground truth poses, shape (N, 7)
        pred_poses (numpy.ndarray): Predicted poses, shape (N, 7)
        output_path (str): Path to save the plot
    """
    # Calculate translation and rotation errors
    translation_errors = []
    rotation_errors = []
    
    for gt, pred in zip(gt_poses, pred_poses):
        # Translation error
        gt_t = gt[4:7]
        pred_t = pred[4:7]
        translation_error = np.linalg.norm(gt_t - pred_t)
        translation_errors.append(translation_error)
        
        # Rotation error
        gt_q = gt[:4]
        pred_q = pred[:4]
        
        gt_euler = quaternion_to_euler(gt_q)[0]
        pred_euler = quaternion_to_euler(pred_q)[0]
        
        euler_diff = np.abs(gt_euler - pred_euler)
        euler_diff = np.minimum(euler_diff, 360 - euler_diff)
        rotation_error = np.linalg.norm(euler_diff)
        rotation_errors.append(rotation_error)
    
    translation_errors = np.array(translation_errors)
    rotation_errors = np.array(rotation_errors)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Translation error over time
    axs[0, 0].plot(range(len(translation_errors)), translation_errors)
    axs[0, 0].set_xlabel('Frame Index')
    axs[0, 0].set_ylabel('Translation Error (m)')
    axs[0, 0].set_title('Translation Error vs. Frame Index')
    axs[0, 0].grid(True)
    
    # Rotation error over time
    axs[0, 1].plot(range(len(rotation_errors)), rotation_errors)
    axs[0, 1].set_xlabel('Frame Index')
    axs[0, 1].set_ylabel('Rotation Error (degrees)')
    axs[0, 1].set_title('Rotation Error vs. Frame Index')
    axs[0, 1].grid(True)
    
    # Translation error distribution
    axs[1, 0].hist(translation_errors, bins=30, alpha=0.7, color='b')
    axs[1, 0].axvline(np.mean(translation_errors), color='r', linestyle='dashed', 
                    linewidth=2, label=f'Mean: {np.mean(translation_errors):.4f}m')
    axs[1, 0].axvline(np.median(translation_errors), color='g', linestyle='dashed', 
                     linewidth=2, label=f'Median: {np.median(translation_errors):.4f}m')
    axs[1, 0].set_xlabel('Translation Error (m)')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Translation Error Distribution')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Rotation error distribution
    axs[1, 1].hist(rotation_errors, bins=30, alpha=0.7, color='b')
    axs[1, 1].axvline(np.mean(rotation_errors), color='r', linestyle='dashed', 
                    linewidth=2, label=f'Mean: {np.mean(rotation_errors):.4f}°')
    axs[1, 1].axvline(np.median(rotation_errors), color='g', linestyle='dashed', 
                     linewidth=2, label=f'Median: {np.median(rotation_errors):.4f}°')
    axs[1, 1].set_xlabel('Rotation Error (degrees)')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Rotation Error Distribution')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_loss_curves(train_losses, val_losses, output_path):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 