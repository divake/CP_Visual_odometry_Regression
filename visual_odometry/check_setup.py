#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to check if the visual odometry project structure is set up correctly.
"""

import os
import sys
import importlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_file_exists(file_path):
    """Check if a file exists."""
    exists = os.path.isfile(file_path)
    logger.info(f"Checking {file_path}: {'✓' if exists else '✗'}")
    return exists

def check_directory_exists(dir_path):
    """Check if a directory exists."""
    exists = os.path.isdir(dir_path)
    logger.info(f"Checking {dir_path}: {'✓' if exists else '✗'}")
    return exists

def check_module_imports():
    """Check if all required modules can be imported."""
    modules = [
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "cv2",
        "tqdm",
        "tensorboard",
        "PIL",
    ]
    
    all_imports_successful = True
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            logger.info(f"Importing {module_name}: ✓")
        except ImportError:
            logger.error(f"Importing {module_name}: ✗")
            all_imports_successful = False
    
    return all_imports_successful

def check_project_structure():
    """Check if the project structure is set up correctly."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Check if all required files exist
    required_files = [
        os.path.join(project_root, "config.py"),
        os.path.join(project_root, "dataset.py"),
        os.path.join(project_root, "train.py"),
        os.path.join(project_root, "test.py"),
        os.path.join(project_root, "main.py"),
        os.path.join(project_root, "models", "base_model.py"),
        os.path.join(project_root, "models", "loss.py"),
        os.path.join(project_root, "utils", "evaluation.py"),
        os.path.join(project_root, "utils", "visualization.py"),
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if not check_file_exists(file_path):
            all_files_exist = False
    
    # Check if all required directories exist
    required_dirs = [
        os.path.join(project_root, "models"),
        os.path.join(project_root, "utils"),
    ]
    
    all_dirs_exist = True
    for dir_path in required_dirs:
        if not check_directory_exists(dir_path):
            all_dirs_exist = False
    
    # Check if the dataset directory exists
    dataset_dir = "/ssd_4TB/divake/LSF_regression/dataset_rgbd_scenes_v2"
    dataset_exists = check_directory_exists(dataset_dir)
    
    # Check if the dataset has the required structure
    if dataset_exists:
        imgs_dir = os.path.join(dataset_dir, "imgs")
        pc_dir = os.path.join(dataset_dir, "pc")
        
        dataset_structure_correct = (
            check_directory_exists(imgs_dir) and
            check_directory_exists(pc_dir)
        )
    else:
        dataset_structure_correct = False
    
    # Check if all required modules can be imported
    all_imports_successful = check_module_imports()
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"All required files exist: {'✓' if all_files_exist else '✗'}")
    logger.info(f"All required directories exist: {'✓' if all_dirs_exist else '✗'}")
    logger.info(f"Dataset exists: {'✓' if dataset_exists else '✗'}")
    logger.info(f"Dataset structure is correct: {'✓' if dataset_structure_correct else '✗'}")
    logger.info(f"All required modules can be imported: {'✓' if all_imports_successful else '✗'}")
    
    # Overall status
    overall_status = (
        all_files_exist and
        all_dirs_exist and
        dataset_exists and
        dataset_structure_correct and
        all_imports_successful
    )
    
    if overall_status:
        logger.info("\n✓ Project structure is set up correctly!")
    else:
        logger.error("\n✗ Project structure has issues. Please fix them before proceeding.")
    
    return overall_status

if __name__ == "__main__":
    check_project_structure() 