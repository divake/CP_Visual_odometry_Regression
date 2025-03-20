# Visual Odometry with RGB-D Scenes Dataset v2

This project implements a visual odometry system using deep learning on the RGB-D Scenes Dataset v2. The system processes pairs of consecutive RGB images to predict relative camera poses and reconstructs the full camera trajectory.

## Dataset Focus

This implementation focuses on **scene_02** from the RGB-D Scenes Dataset v2:
- We use only scene_02 for both training and testing
- 80% of frames from scene_02 are used for training (further split into 80% training, 20% validation)
- 100% of frames are used for testing to reconstruct the full trajectory
- Once the model works well on scene_02, we'll scale to the entire dataset

## Directory Structure

```
visual_odometry/
├── src/
│   ├── config.py          # Configuration settings (dataset paths, hyperparameters)
│   ├── dataloader.py      # Dataset loading and preprocessing 
│   ├── model.py           # Neural network architecture (ResNet18-based)
│   ├── loss.py            # Custom loss functions for poses and rotations
│   ├── utils.py           # Utility functions (quaternion conversions, trajectory computation)
│   ├── train.py           # Training loop and checkpoint management
│   ├── test.py            # Evaluation on test data
│   └── visualization.py   # Visualization of trajectories and errors
├── Results/               # Results directory (created at runtime)
│   ├── checkpoints/       # Model checkpoints
│   ├── logs/              # Training and evaluation logs
│   ├── predictions/       # Predicted trajectories
│   └── visualizations/    # Plots and visualizations
└── Other/                 # Other utility scripts and experiments
    └── explore_dataset.py # Dataset exploration script
```

## Dataset Structure

The RGB-D Scenes Dataset v2 is structured as follows:
```
dataset_rgbd_scenes_v2/
├── imgs/                  # RGB and depth images
│   ├── scene_01/          # Scene 1 images
│   ├── scene_02/          # Scene 2 images (our focus)
│   └── ...
└── pc/                    # Point cloud data
    ├── 01.ply             # 3D point cloud for scene 1
    ├── 01.pose            # Camera poses for scene 1
    ├── 02.pose            # Camera poses for scene 2 (our focus)
    └── ...
```

## System Design

1. **Data Preparation**: 
   - Extract consecutive frame pairs from scene_02
   - Compute relative poses between frames
   - Apply data transformations (resizing, normalization)
   - Split into training, validation, and test sets

2. **Model Architecture**:
   - ResNet18-based siamese network
   - Dual regression heads for translations and rotations
   - Translation head outputs (x, y, z)
   - Rotation head outputs quaternion (w, x, y, z)

3. **Loss Functions**:
   - Translation loss: Mean Squared Error (MSE)
   - Rotation loss: Quaternion-specific loss accounting for q = -q ambiguity
   - Combined weighted loss with quaternion normalization term

4. **Training Process**:
   - Adam optimizer with learning rate scheduling
   - Regular validation to monitor performance
   - Checkpoint saving for best models

5. **Evaluation**:
   - Predict relative poses for each consecutive frame pair
   - Reconstruct the full trajectory by chaining relative poses
   - Calculate Absolute Trajectory Error (ATE) against ground truth
   - Visualize predicted vs. ground truth trajectories

## Usage

1. **Setup**: Make sure the dataset is available at the path specified in `config.py`.

2. **Explore the dataset**:
   ```bash
   cd visual_odometry/Other
   python explore_dataset.py
   ```

3. **Train the model**:
   ```bash
   cd visual_odometry/src
   python train.py
   ```

4. **Evaluate the model**:
   ```bash
   cd visual_odometry/src
   python test.py
   ```

5. **Visualize results**:
   ```bash
   cd visual_odometry/src
   python visualization.py
   ```

## Results

The system is evaluated on scene_02 with the following metrics:
- Translation Error (RMSE): Measures accuracy of position estimation
- Rotation Error (RMSE): Measures accuracy of orientation estimation
- Full trajectory visualization: Compares predicted and ground truth paths

## Future Work

1. Scale to the full RGB-D Scenes Dataset (all 14 scenes)
2. Incorporate depth information for improved pose estimation
3. Implement loop closure detection for drift correction
4. Add bundle adjustment for trajectory optimization
