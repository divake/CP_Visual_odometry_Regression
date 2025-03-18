#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from dataset_explorer import DatasetExplorer
from data_preprocessing import create_data_loaders, verify_data_pipeline

def main():
    """
    Main function for Phase 1: Dataset Understanding and Preprocessing.
    """
    # Get the absolute path of the workspace
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dataset_path = os.path.join(workspace_dir, "dataset_rgbd_scenes_v2")
    
    parser = argparse.ArgumentParser(description='Visual Odometry Dataset Exploration and Preprocessing')
    parser.add_argument('--dataset_path', type=str, default=default_dataset_path,
                        help='Path to the RGB-D dataset')
    parser.add_argument('--scene_id', type=str, default='02', 
                        help='Scene ID to explore')
    parser.add_argument('--output_dir', type=str, default='visualization/outputs',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for data loaders')
    parser.add_argument('--use_depth', action='store_true',
                        help='Whether to use depth images')
    parser.add_argument('--explore', action='store_true',
                        help='Run dataset exploration')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing')
    
    args = parser.parse_args()
    
    # Verify that the dataset path exists
    if not os.path.exists(args.dataset_path):
        # Try some common locations
        potential_paths = [
            args.dataset_path,
            os.path.join(workspace_dir, "dataset_rgbd_scenes_v2"),
            "/ssd_4TB/divake/LSF_regression/dataset_rgbd_scenes_v2",
            "../dataset_rgbd_scenes_v2",
            "../../dataset_rgbd_scenes_v2"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                args.dataset_path = path
                print(f"Found dataset at: {args.dataset_path}")
                break
        else:
            print(f"ERROR: Dataset not found at {args.dataset_path}")
            print("Please specify the correct path with --dataset_path")
            return
    
    # Convert output_dir to absolute path if it's relative
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print arguments
    print("=== Visual Odometry Dataset Exploration and Preprocessing ===")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Scene ID: {args.scene_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Use depth: {args.use_depth}")
    print("=" * 60)
    
    # Run dataset exploration if requested
    if args.explore:
        print("\n=== Starting Dataset Exploration ===")
        explorer = DatasetExplorer(args.dataset_path, args.scene_id)
        
        # Explore sample images
        print("\nExploring sample RGB images...")
        explorer.explore_sample_images(
            num_samples=5, 
            save_path=os.path.join(args.output_dir, "sample_images.png")
        )
        
        # Visualize depth samples
        print("\nVisualizing depth samples...")
        explorer.visualize_depth_samples(
            num_samples=5, 
            save_path=os.path.join(args.output_dir, "depth_samples.png")
        )
        
        # Analyze pose data
        print("\nAnalyzing pose data...")
        explorer.analyze_pose_data(
            save_dir=args.output_dir
        )
        
        # Visualize trajectory
        print("\nVisualizing ground truth trajectory...")
        explorer.visualize_trajectory(
            save_path=os.path.join(args.output_dir, "ground_truth_trajectory.png")
        )
        
        # Visualize relative motion
        print("\nVisualizing relative motion between frames...")
        explorer.visualize_relative_motion(
            step=10, 
            save_path=os.path.join(args.output_dir, "relative_motion.png")
        )
        
        print("\n=== Dataset Exploration Complete ===")
    
    # Run data preprocessing if requested
    if args.preprocess:
        print("\n=== Starting Data Preprocessing ===")
        
        # Create data loaders
        print("\nCreating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_path=args.dataset_path,
            scene_id=args.scene_id,
            batch_size=args.batch_size,
            use_depth=args.use_depth
        )
        
        # Print data loader information
        print(f"\nTraining samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Testing samples: {len(test_loader.dataset)}")
        
        # Verify the data pipeline
        print("\nVerifying training data pipeline...")
        verify_data_pipeline(
            train_loader,
            save_path=os.path.join(args.output_dir, "train_samples.png")
        )
        
        # Verify validation data pipeline
        print("\nVerifying validation data pipeline...")
        verify_data_pipeline(
            val_loader,
            save_path=os.path.join(args.output_dir, "val_samples.png")
        )
        
        # Verify test data pipeline
        print("\nVerifying test data pipeline...")
        verify_data_pipeline(
            test_loader,
            save_path=os.path.join(args.output_dir, "test_samples.png")
        )
        
        print("\n=== Data Preprocessing Complete ===")
    
    # If neither explore nor preprocess is specified, run both
    if not args.explore and not args.preprocess:
        print("No specific action specified. Run with --explore or --preprocess to perform these actions.")
        print("Example: python main.py --explore --preprocess")

if __name__ == "__main__":
    main() 