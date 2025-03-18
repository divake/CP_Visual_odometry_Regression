#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

class DatasetExplorer:
    def __init__(self, dataset_path, scene_id="02"):
        """
        Initialize the dataset explorer with the path to the dataset and scene ID.
        
        Args:
            dataset_path (str): Path to the RGB-D v.2 dataset directory
            scene_id (str): Scene ID to explore (default: "02")
        """
        self.dataset_path = dataset_path
        self.scene_id = scene_id
        self.scene_img_path = os.path.join(dataset_path, "imgs", f"scene_{scene_id}")
        self.pose_file_path = os.path.join(dataset_path, "pc", f"{scene_id}.pose")
        
        # Verify paths
        if not os.path.exists(self.scene_img_path):
            raise FileNotFoundError(f"Scene path not found: {self.scene_img_path}")
        if not os.path.exists(self.pose_file_path):
            raise FileNotFoundError(f"Pose file not found: {self.pose_file_path}")
        
        # Get image lists
        self.color_images = sorted([f for f in os.listdir(self.scene_img_path) if f.endswith("-color.png")])
        self.depth_images = sorted([f for f in os.listdir(self.scene_img_path) if f.endswith("-depth.png")])
        
        # Load pose data
        self.poses = self._load_poses()
        
        print(f"Dataset Explorer initialized for scene_{scene_id}")
        print(f"Found {len(self.color_images)} color images and {len(self.depth_images)} depth images")
        print(f"Loaded {len(self.poses)} poses")
        
    def _load_poses(self):
        """
        Load camera poses from the pose file.
        
        Returns:
            numpy.ndarray: Array of camera poses (quaternions and translations)
        """
        poses = []
        
        with open(self.pose_file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            values = [float(v) for v in line.strip().split()]
            if len(values) == 7:  # Ensure valid pose line
                poses.append(values)
        
        return np.array(poses)
    
    def explore_sample_images(self, num_samples=5, save_path=None):
        """
        Display sample RGB images from the dataset.
        
        Args:
            num_samples (int): Number of sample images to display
            save_path (str, optional): Path to save the figure
        """
        # Ensure we have enough images
        num_samples = min(num_samples, len(self.color_images))
        
        # Select evenly spaced samples
        indices = np.linspace(0, len(self.color_images) - 1, num_samples, dtype=int)
        
        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(indices):
            img_path = os.path.join(self.scene_img_path, self.color_images[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img)
            plt.title(f"Frame {self.color_images[idx].split('-')[0]}")
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Sample images saved to {save_path}")
        
        plt.show()
    
    def visualize_depth_samples(self, num_samples=5, save_path=None):
        """
        Display sample depth images from the dataset.
        
        Args:
            num_samples (int): Number of sample images to display
            save_path (str, optional): Path to save the figure
        """
        # Ensure we have enough images
        num_samples = min(num_samples, len(self.depth_images))
        
        # Select evenly spaced samples
        indices = np.linspace(0, len(self.depth_images) - 1, num_samples, dtype=int)
        
        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(indices):
            img_path = os.path.join(self.scene_img_path, self.depth_images[idx])
            depth = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            
            # Normalize for visualization (depth values are in millimeters)
            depth_normalized = np.clip(depth.astype(np.float32) / 5000.0, 0, 1)
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(depth_normalized, cmap='viridis')
            plt.title(f"Frame {self.depth_images[idx].split('-')[0]}")
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Depth samples saved to {save_path}")
        
        plt.show()
    
    def analyze_pose_data(self, save_dir=None):
        """
        Analyze and visualize the distribution of translation and rotation components.
        
        Args:
            save_dir (str, optional): Directory to save figures
        """
        # Extract quaternions and translations
        quaternions = self.poses[:, 0:4]  # w, x, y, z
        translations = self.poses[:, 4:7]  # x, y, z
        
        # Plot translation distributions
        plt.figure(figsize=(15, 5))
        for i, label in enumerate(['X', 'Y', 'Z']):
            plt.subplot(1, 3, i+1)
            plt.hist(translations[:, i], bins=30)
            plt.title(f'Translation {label} Distribution')
            plt.xlabel(f'{label} (meters)')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "translation_distributions.png"))
            print(f"Translation distributions saved to {save_dir}")
        
        plt.show()
        
        # Plot quaternion distributions
        plt.figure(figsize=(15, 5))
        for i, label in enumerate(['W', 'X', 'Y', 'Z']):
            plt.subplot(1, 4, i+1)
            plt.hist(quaternions[:, i], bins=30)
            plt.title(f'Quaternion {label} Distribution')
            plt.xlabel(f'{label}')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "quaternion_distributions.png"))
            print(f"Quaternion distributions saved to {save_dir}")
        
        plt.show()
        
        return quaternions, translations
    
    def visualize_trajectory(self, save_path=None):
        """
        Visualize the 3D trajectory of the camera.
        
        Args:
            save_path (str, optional): Path to save the figure
        """
        # Extract translations
        translations = self.poses[:, 4:7]  # x, y, z
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera positions
        ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], 'b-', linewidth=2)
        ax.scatter(translations[0, 0], translations[0, 1], translations[0, 2], c='g', marker='o', s=100, label='Start')
        ax.scatter(translations[-1, 0], translations[-1, 1], translations[-1, 2], c='r', marker='o', s=100, label='End')
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('Camera Trajectory (Ground Truth)')
        ax.legend()
        
        # Set equal aspect ratio for all axes
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
        
        # Add coordinate axes for orientation
        length = max_range * 0.05
        origin = [translations[0, 0], translations[0, 1], translations[0, 2]]
        
        # X axis
        ax.quiver(origin[0], origin[1], origin[2], length, 0, 0, color='r', arrow_length_ratio=0.1)
        # Y axis
        ax.quiver(origin[0], origin[1], origin[2], 0, length, 0, color='g', arrow_length_ratio=0.1)
        # Z axis
        ax.quiver(origin[0], origin[1], origin[2], 0, 0, length, color='b', arrow_length_ratio=0.1)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Trajectory visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_relative_motion(self, step=10, save_path=None):
        """
        Visualize the relative motion between consecutive frames at regular intervals.
        
        Args:
            step (int): Step size for selecting frames
            save_path (str, optional): Path to save the figure
        """
        # Extract translations and quaternions
        translations = self.poses[:, 4:7]
        quaternions = self.poses[:, 0:4]
        
        # Calculate relative translations and rotations
        rel_translations = []
        rel_rotation_angles = []
        
        for i in range(0, len(translations) - step, step):
            # Get frames with the step interval
            q1 = quaternions[i]
            t1 = translations[i]
            
            q2 = quaternions[i + step]
            t2 = translations[i + step]
            
            # Convert quaternions to rotation matrices
            R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()  # scipy uses x,y,z,w
            R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]]).as_matrix()
            
            # Calculate relative rotation: R_rel = R2 * R1^T
            R_rel = np.dot(R2, R1.T)
            
            # Calculate relative translation: t_rel = t2 - R_rel * t1
            t_rel = t2 - np.dot(R_rel, t1)
            
            # Calculate rotation angle (in degrees)
            rot = Rotation.from_matrix(R_rel)
            angle = np.linalg.norm(rot.as_rotvec()) * 180 / np.pi
            
            rel_translations.append(np.linalg.norm(t_rel))
            rel_rotation_angles.append(angle)
        
        # Plot the relative motion
        plt.figure(figsize=(15, 5))
        
        # Plot translation magnitudes
        plt.subplot(1, 2, 1)
        plt.plot(range(0, len(rel_translations) * step, step), rel_translations)
        plt.xlabel('Frame Index')
        plt.ylabel('Translation Magnitude (meters)')
        plt.title('Relative Translation between Frames')
        plt.grid(True)
        
        # Plot rotation angles
        plt.subplot(1, 2, 2)
        plt.plot(range(0, len(rel_rotation_angles) * step, step), rel_rotation_angles)
        plt.xlabel('Frame Index')
        plt.ylabel('Rotation Angle (degrees)')
        plt.title('Relative Rotation between Frames')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Relative motion visualization saved to {save_path}")
        
        plt.show()

def main():
    """
    Main function to run the dataset explorer.
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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization/outputs")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset explorer
    explorer = DatasetExplorer(dataset_path, scene_id)
    
    # Explore sample images
    explorer.explore_sample_images(num_samples=5, save_path=os.path.join(output_dir, "sample_images.png"))
    
    # Visualize depth samples
    explorer.visualize_depth_samples(num_samples=5, save_path=os.path.join(output_dir, "depth_samples.png"))
    
    # Analyze pose data
    explorer.analyze_pose_data(save_dir=output_dir)
    
    # Visualize trajectory
    explorer.visualize_trajectory(save_path=os.path.join(output_dir, "ground_truth_trajectory.png"))
    
    # Visualize relative motion
    explorer.visualize_relative_motion(step=10, save_path=os.path.join(output_dir, "relative_motion.png"))

if __name__ == "__main__":
    main() 