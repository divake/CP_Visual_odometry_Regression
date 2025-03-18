#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation

def quaternion_to_matrix(quaternion):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quaternion (numpy.ndarray): Quaternion in w, x, y, z format
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    # Scipy expects quaternions in x, y, z, w format
    q_scipy = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    return Rotation.from_quat(q_scipy).as_matrix()

def matrix_to_quaternion(matrix):
    """
    Convert rotation matrix to quaternion.
    
    Args:
        matrix (numpy.ndarray): 3x3 rotation matrix
        
    Returns:
        numpy.ndarray: Quaternion in w, x, y, z format
    """
    # Scipy returns quaternions in x, y, z, w format
    q_scipy = Rotation.from_matrix(matrix).as_quat()
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

def normalize_quaternion(quaternion):
    """
    Normalize a quaternion to unit length.
    
    Args:
        quaternion (numpy.ndarray): Quaternion in w, x, y, z format
        
    Returns:
        numpy.ndarray: Normalized quaternion
    """
    return quaternion / np.linalg.norm(quaternion)

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (q1 * q2).
    
    Args:
        q1 (numpy.ndarray): First quaternion in w, x, y, z format
        q2 (numpy.ndarray): Second quaternion in w, x, y, z format
        
    Returns:
        numpy.ndarray: Result quaternion in w, x, y, z format
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])

def quaternion_inverse(quaternion):
    """
    Compute the inverse of a quaternion.
    
    Args:
        quaternion (numpy.ndarray): Quaternion in w, x, y, z format
        
    Returns:
        numpy.ndarray: Inverse quaternion
    """
    # For unit quaternion, the inverse is the conjugate
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])

def get_relative_pose(pose1, pose2):
    """
    Calculate the relative pose from pose1 to pose2.
    
    Args:
        pose1 (numpy.ndarray): First pose (7-element array: quaternion and translation)
        pose2 (numpy.ndarray): Second pose (7-element array: quaternion and translation)
        
    Returns:
        numpy.ndarray: Relative pose (7-element array: quaternion and translation)
    """
    # Extract quaternions and translations
    q1 = pose1[:4]  # w, x, y, z
    t1 = pose1[4:7]
    
    q2 = pose2[:4]
    t2 = pose2[4:7]
    
    # Convert quaternions to rotation matrices
    R1 = quaternion_to_matrix(q1)
    R2 = quaternion_to_matrix(q2)
    
    # Calculate relative rotation: R_rel = R2 * R1^T
    R_rel = np.dot(R2, R1.T)
    
    # Calculate relative translation: t_rel = t2 - R_rel * t1
    t_rel = t2 - np.dot(R_rel, t1)
    
    # Convert back to quaternion
    q_rel = matrix_to_quaternion(R_rel)
    
    # Normalize quaternion to unit length
    q_rel = normalize_quaternion(q_rel)
    
    return np.concatenate([q_rel, t_rel])

def compose_poses(pose1, pose2):
    """
    Compose two poses (pose1 * pose2).
    
    Args:
        pose1 (numpy.ndarray): First pose (7-element array: quaternion and translation)
        pose2 (numpy.ndarray): Second pose (7-element array: quaternion and translation)
        
    Returns:
        numpy.ndarray: Composed pose (7-element array: quaternion and translation)
    """
    # Extract quaternions and translations
    q1 = pose1[:4]  # w, x, y, z
    t1 = pose1[4:7]
    
    q2 = pose2[:4]
    t2 = pose2[4:7]
    
    # Convert quaternions to rotation matrices
    R1 = quaternion_to_matrix(q1)
    
    # Compose: first rotate by q1, then translate by t1, then rotate by q2, then translate by t2
    # Rotation: q_result = q2 * q1
    q_result = quaternion_multiply(q2, q1)
    
    # Translation: t_result = t2 + R2 * t1
    t_result = t2 + np.dot(quaternion_to_matrix(q2), t1)
    
    return np.concatenate([q_result, t_result])

def invert_pose(pose):
    """
    Invert a pose.
    
    Args:
        pose (numpy.ndarray): Pose to invert (7-element array: quaternion and translation)
        
    Returns:
        numpy.ndarray: Inverted pose (7-element array: quaternion and translation)
    """
    # Extract quaternion and translation
    q = pose[:4]  # w, x, y, z
    t = pose[4:7]
    
    # Invert quaternion
    q_inv = quaternion_inverse(q)
    
    # Invert translation: t_inv = -R^T * t = -(R^-1) * t
    R_inv = quaternion_to_matrix(q_inv)
    t_inv = -np.dot(R_inv, t)
    
    return np.concatenate([q_inv, t_inv])

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in degrees.
    
    Args:
        quaternion (numpy.ndarray): Quaternion in w, x, y, z format
        
    Returns:
        tuple: (roll, pitch, yaw) in degrees
    """
    # Scipy expects quaternions in x, y, z, w format
    q_scipy = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    euler = Rotation.from_quat(q_scipy).as_euler('xyz', degrees=True)
    return euler[0], euler[1], euler[2]  # roll, pitch, yaw

def euler_to_quaternion(roll, pitch, yaw, degrees=True):
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll (float): Roll angle
        pitch (float): Pitch angle
        yaw (float): Yaw angle
        degrees (bool): Whether the angles are in degrees (True) or radians (False)
        
    Returns:
        numpy.ndarray: Quaternion in w, x, y, z format
    """
    # Scipy expects angles in radians
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    
    # Scipy returns quaternions in x, y, z, w format
    q_scipy = rot.as_quat()
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

def pose_to_transformation_matrix(pose):
    """
    Convert a pose to a 4x4 transformation matrix.
    
    Args:
        pose (numpy.ndarray): Pose (7-element array: quaternion and translation)
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    # Extract quaternion and translation
    q = pose[:4]  # w, x, y, z
    t = pose[4:7]
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_matrix(q)
    
    # Create transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def transformation_matrix_to_pose(T):
    """
    Convert a 4x4 transformation matrix to a pose.
    
    Args:
        T (numpy.ndarray): 4x4 transformation matrix
        
    Returns:
        numpy.ndarray: Pose (7-element array: quaternion and translation)
    """
    # Extract rotation matrix and translation
    R = T[:3, :3]
    t = T[:3, 3]
    
    # Convert rotation matrix to quaternion
    q = matrix_to_quaternion(R)
    
    return np.concatenate([q, t])

def compute_absolute_trajectory_error(gt_poses, estimated_poses):
    """
    Compute the Absolute Trajectory Error (ATE) between ground truth and estimated poses.
    
    Args:
        gt_poses (list): List of ground truth poses
        estimated_poses (list): List of estimated poses
        
    Returns:
        float: Root Mean Square Error (RMSE) of the translational error
    """
    assert len(gt_poses) == len(estimated_poses), "Pose lists must have the same length"
    
    # Compute translation errors
    errors = []
    for gt_pose, est_pose in zip(gt_poses, estimated_poses):
        gt_t = gt_pose[4:7]
        est_t = est_pose[4:7]
        error = np.linalg.norm(gt_t - est_t)
        errors.append(error)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    return rmse

def compute_relative_pose_error(gt_poses, estimated_poses, delta=1):
    """
    Compute the Relative Pose Error (RPE) between ground truth and estimated poses.
    
    Args:
        gt_poses (list): List of ground truth poses
        estimated_poses (list): List of estimated poses
        delta (int): Frame distance to compute relative poses
        
    Returns:
        tuple: (translational_error, rotational_error) as (RMSE in meters, mean angular error in degrees)
    """
    assert len(gt_poses) == len(estimated_poses), "Pose lists must have the same length"
    
    # Compute relative poses
    trans_errors = []
    rot_errors = []
    
    for i in range(len(gt_poses) - delta):
        # Ground truth relative pose
        gt_rel_pose = get_relative_pose(gt_poses[i], gt_poses[i + delta])
        
        # Estimated relative pose
        est_rel_pose = get_relative_pose(estimated_poses[i], estimated_poses[i + delta])
        
        # Compute errors
        # Translation error
        gt_t = gt_rel_pose[4:7]
        est_t = est_rel_pose[4:7]
        trans_error = np.linalg.norm(gt_t - est_t)
        trans_errors.append(trans_error)
        
        # Rotation error
        gt_q = gt_rel_pose[:4]
        est_q = est_rel_pose[:4]
        
        # Convert to matrices
        gt_R = quaternion_to_matrix(gt_q)
        est_R = quaternion_to_matrix(est_q)
        
        # Error rotation matrix: E = est_R * gt_R^T
        E = np.dot(est_R, gt_R.T)
        
        # Convert to axis-angle representation
        rot_error = np.arccos(np.clip((np.trace(E) - 1) / 2, -1.0, 1.0)) * 180 / np.pi
        rot_errors.append(rot_error)
    
    # Compute mean errors
    trans_rmse = np.sqrt(np.mean(np.array(trans_errors) ** 2))
    rot_mean = np.mean(np.array(rot_errors))
    
    return trans_rmse, rot_mean

def filter_trajectory(poses, window_size=5):
    """
    Apply a simple moving average filter to a trajectory.
    
    Args:
        poses (list): List of poses
        window_size (int): Window size for averaging
        
    Returns:
        list: Filtered list of poses
    """
    filtered_poses = []
    
    # For positions at the start and end, use smaller windows
    for i in range(len(poses)):
        # Determine window bounds
        start = max(0, i - window_size // 2)
        end = min(len(poses), i + window_size // 2 + 1)
        
        # Get poses in the window
        window = poses[start:end]
        
        # Average quaternions
        q_sum = np.zeros(4)
        for pose in window:
            q = pose[:4]
            
            # Ensure consistent sign (handle antipodal quaternions)
            if i > 0 and np.dot(q, filtered_poses[-1][:4]) < 0:
                q = -q
                
            q_sum += q
        
        q_avg = q_sum / len(window)
        q_avg = normalize_quaternion(q_avg)
        
        # Average translations
        t_avg = np.zeros(3)
        for pose in window:
            t_avg += pose[4:7]
        t_avg /= len(window)
        
        # Combine and store
        filtered_poses.append(np.concatenate([q_avg, t_avg]))
    
    return filtered_poses 