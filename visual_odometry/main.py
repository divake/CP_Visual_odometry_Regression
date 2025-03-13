#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for the visual odometry system.

This script serves as the entry point for the visual odometry system.
It parses command-line arguments and launches either training or testing.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from visual_odometry.config import create_directories, LOG_CONFIG
from visual_odometry.train import main as train_main
from visual_odometry.test import main as test_main

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual Odometry System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for train and test commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test the model")
    test_parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    test_parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save results")
    test_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    test_parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    test_parser.add_argument("--include_images", action="store_true", help="Include images in visualizations")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Run the appropriate command
    if args.command == "train":
        logger.info("Starting training...")
        train_main(args)
    elif args.command == "test":
        logger.info("Starting testing...")
        test_main(args)
    else:
        logger.error("Invalid command. Use 'train' or 'test'.")
        sys.exit(1)


if __name__ == "__main__":
    main() 