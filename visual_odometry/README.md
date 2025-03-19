# Visual Odometry with RGB-D Dataset

This project implements a visual odometry system using the RGB-D Scenes Dataset v2. The system uses deep learning to estimate camera poses between consecutive frames.

## Dataset

The project uses the RGB-D Scenes Dataset v2 located at:
```
/ssd_4TB/divake/LSF_regression/dataset_rgbd_scenes_v2
```

The dataset consists of:
- RGB and depth images for 14 scenes
- Ground truth camera poses for each frame
- 3D point cloud data for each scene
- Object labels for semantic understanding

## Project Structure

```
visual_odometry/
├── dataloader.py       # Dataset handling, preprocessing, and creating image pairs
├── model.py            # Neural network architecture
├── loss.py             # Custom loss functions (especially for rotations)
├── train.py            # Main training loop and validation
├── test.py             # Evaluation on test set
├── visualization.py    # For plotting trajectories and errors
├── utils.py            # Helper functions like quaternion conversions
├── config.py           # Hyperparameters and configuration settings
└── Results/            # For saving model checkpoints, logs, and visualizations
```

## Workflow

1. **Data Preparation**: The dataloader processes RGB-D images and ground truth poses.
2. **Model Training**: The model is trained to predict relative poses between consecutive frames.
3. **Evaluation**: The model is evaluated on test sequences.
4. **Visualization**: Results are visualized by comparing predicted trajectories to ground truth.

## Usage

To be implemented:
- Training the visual odometry model
- Testing the model on new sequences
- Visualizing the results
