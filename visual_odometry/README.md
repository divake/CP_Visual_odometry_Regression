# Visual Odometry Project

This project implements a visual odometry system using the RGB-D Scenes Dataset v2. The goal is to estimate camera trajectory from RGB-D image sequences.

## Phase 1: Dataset Understanding and Preprocessing

The current implementation focuses on Phase 1, which includes:
1. Dataset exploration
2. Data preprocessing for training a deep learning model
3. Visualization tools for dataset and trajectories

## Directory Structure

```
visual_odometry/
├── data/                   # Data loading and preprocessing modules
├── utils/                  # Utility functions and helper tools
├── visualization/          # Visualization tools for data and results
├── dataset_explorer.py     # Script for exploring the dataset
├── data_preprocessing.py   # Data preprocessing pipeline
├── main.py                 # Main script to run exploration and preprocessing
├── project_details.mdc     # Detailed project documentation
└── README.md               # This file
```

## Prerequisites

The project requires the following Python packages:
- NumPy
- Matplotlib
- OpenCV
- PyTorch
- scikit-learn
- SciPy

## Dataset

This project uses the RGB-D Scenes Dataset v2, which should be structured as follows:
```
dataset_rgbd_scenes_v2/
├── README.txt              # Original dataset documentation
├── imgs/                   # RGB and depth images
│   ├── scene_01/           # Scene 1 images
│   ├── scene_02/           # Scene 2 images
│   └── ...                 # Other scene directories
└── pc/                     # Point cloud data
    ├── 01.ply              # 3D point cloud for scene 1
    ├── 01.pose             # Camera poses for scene 1
    ├── 01.label            # Object labels for scene 1
    └── ...                 # Files for other scenes
```

## Usage

### Dataset Exploration

To explore the dataset and visualize sample images, poses, and trajectories:

```bash
python main.py --explore --dataset_path ../dataset_rgbd_scenes_v2 --scene_id 02 --output_dir visualization/outputs
```

This will generate visualizations for:
- Sample RGB images
- Sample depth images
- Pose distribution histograms
- Ground truth camera trajectory
- Relative motion between frames

### Data Preprocessing

To process the data and create data loaders for training:

```bash
python main.py --preprocess --dataset_path ../dataset_rgbd_scenes_v2 --scene_id 02 --output_dir visualization/outputs --use_depth
```

This will:
- Create training, validation, and test data loaders
- Verify the data pipeline by displaying sample image pairs with their relative poses
- Save sample visualizations

### Running Both Exploration and Preprocessing

To run both dataset exploration and data preprocessing:

```bash
python main.py --explore --preprocess --dataset_path ../dataset_rgbd_scenes_v2 --scene_id 02 --output_dir visualization/outputs --use_depth
```

## Implemented Features

- **Dataset Explorer**:
  - Loading and displaying RGB and depth images
  - Parsing camera poses 
  - Analyzing pose distributions
  - Visualizing 3D camera trajectory
  - Analyzing relative camera motion

- **Data Preprocessing**:
  - Custom PyTorch Dataset for RGB-D image pairs and poses
  - Training, validation, and test set splitting
  - Image transformations and normalization
  - Relative pose calculation between frames
  - Optional depth image inclusion

- **Utility Functions**:
  - Quaternion and pose operations
  - Pose transformation and composition
  - Error metric calculation (ATE, RPE)
  - Trajectory filtering

- **Visualization Tools**:
  - 3D trajectory plotting
  - Multi-trajectory comparison
  - Trajectory animation
  - Error metric plotting
  - Image pair visualization
  - Feature match visualization
  - Error distribution analysis

## Next Steps

The upcoming phases will include:
1. **Phase 2**: Model Implementation
   - Design and implement a deep learning model for pose estimation
   - Train the model on the preprocessed data
   - Evaluate model performance

2. **Phase 3**: Trajectory Reconstruction
   - Use the trained model to estimate camera trajectory
   - Visualize and compare with ground truth
   - Optimize for better performance and reduced drift 