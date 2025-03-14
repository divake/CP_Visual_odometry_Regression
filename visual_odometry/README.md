# Visual Odometry System for RGB-D Scenes Dataset v2

This project implements a complete visual odometry system using the RGB-D Scenes Dataset v2. The system uses deep learning to estimate camera poses from RGB-D image pairs.

## Features

- ResNet backbone for feature extraction from RGB-D images
- **Separate regression heads for rotation and translation prediction**
- Quaternion representation for rotations with proper normalization
- Customizable loss functions with weighted components
- Complete training and evaluation pipeline

## Project Structure

```
visual_odometry/
├── models/
│   ├── base_model.py     # ResNet-based visual odometry model with separate heads
│   └── loss.py           # Loss functions for training
├── utils/
│   ├── evaluation.py     # Metrics for evaluating performance
│   └── visualization.py  # Functions for visualizing results
├── config.py             # Configuration parameters
├── train.py              # Training script
├── test.py               # Testing and evaluation script
├── main.py               # Main script to launch training/testing
└── README.md             # This file
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Matplotlib
- OpenCV
- tqdm
- tensorboard

You can install the required packages using:

```bash
pip install torch torchvision numpy matplotlib opencv-python tqdm tensorboard
```

## Dataset

The system uses the RGB-D Scenes Dataset v2, which consists of 14 scenes containing furniture and objects. Each scene includes:

- RGB and depth images
- Camera poses
- 3D point clouds and labels

The dataset is organized as follows:

```
dataset_rgbd_scenes_v2/
├── imgs/                 # RGB and depth images
│   ├── scene_01/         # Scene 1 images
│   ├── scene_02/         # Scene 2 images
│   └── ...               # Other scene directories
└── pc/                   # Point cloud data
    ├── 01.ply            # 3D point cloud for scene 1
    ├── 01.pose           # Camera poses for scene 1
    ├── 01.label          # Object labels for scene 1
    └── ...               # Files for other scenes
```

## Model Architecture

The visual odometry model uses a ResNet backbone with separate regression heads:

1. **Feature Extraction**: A modified ResNet processes RGB-D image pairs
2. **Shared Processing**: Extracted features go through shared layers
3. **Separate Heads**:
   - **Rotation Head**: Specialized layers for quaternion prediction with explicit normalization
   - **Translation Head**: Dedicated layers for translation vector prediction

This separation allows the model to better handle the different characteristics of rotation and translation components, improving overall accuracy.

## Usage

### Training

To train the model:

```bash
python main.py train
```

To resume training from a checkpoint:

```bash
python main.py train --resume /path/to/checkpoint.pth
```

### Testing

To test the model:

```bash
python main.py test --model_path /path/to/model.pth --output_dir test_results
```

Additional options:
- `--batch_size`: Batch size for testing (default: 16)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--include_images`: Include images in visualizations

## Configuration

The system configuration is defined in `config.py`. Key parameters include:

### Model Configuration
- Backbone: ResNet-18
- Input channels: 4 (RGB + Depth)
- FC layers: [512*2, 256, 128, 7]
- Dropout rate: 0.2
- Weight initialization: Kaiming

### Training Configuration
- Batch size: 16
- Learning rate: 1e-4
- Weight decay: 1e-5
- Epochs: 100
- Early stopping patience: 10
- Loss weights: rotation (10.0), translation (1.0)
- Learning rate scheduler: ReduceLROnPlateau

### Dataset Configuration
- Image size: (640, 480)
- Maximum depth: 10.0 meters
- Maximum frame gap: 1

### Evaluation Configuration
- Metrics: ATE, RPE, translation error, rotation error, drift
- Visualization options: trajectory plots, error plots, animations

## Evaluation Metrics

The system evaluates performance using the following metrics:

- **Absolute Trajectory Error (ATE)**: Measures global consistency of the trajectory
- **Relative Pose Error (RPE)**: Measures local accuracy over fixed time intervals
- **Translation Error**: Error in position estimation (in meters)
- **Rotation Error**: Error in orientation estimation (in degrees)
- **Drift**: Error accumulation over trajectory length

## Visualization

The system generates various visualizations:

- 3D trajectory plots (predicted vs. ground truth)
- Per-frame pose errors
- Error histograms
- Trajectory animations
- Visual odometry videos (optional)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RGB-D Scenes Dataset v2 for providing the data
- PyTorch team for the deep learning framework
- ResNet authors for the backbone architecture

## Dataset Insights and Considerations

Based on our analysis, here are important considerations for working with this dataset:

1. **Depth Scaling**: 
   - The depth values in the dataset require proper scaling to convert to metric units (meters)
   - Our analysis shows that the relationship between raw depth values and camera positions is not straightforward
   - Based on common practices for RGB-D sensors (like Kinect), a scaling factor of 1/1000 is recommended:
     ```python
     depth_meters = depth_raw / 1000.0
     ```
   - This would convert the typical depth value of ~10304 to ~10.3 meters, which is reasonable for indoor scenes
   - The maximum depth value of 29842 would convert to ~29.8 meters

2. **Small Motion**: The relative motion between consecutive frames is small (mean translation ~6.1mm, mean rotation ~0.43 degrees), which means:
   - The model needs to be sensitive to small changes
   - Data augmentation should preserve these small motions
   - Loss function should be carefully designed to handle small values

3. **Quaternion Representation**: The ground truth poses use quaternions for rotation. Our model should:
   - Ensure quaternion normalization
   - Consider using a specialized loss function for quaternions
   - Possibly use a separate head for rotation and translation prediction

4. **Valid Depth Regions**: Only ~75% of depth pixels have valid values. The model should:
   - Handle missing depth values appropriately
   - Consider using a mask for valid depth regions
   - Possibly use confidence weighting based on depth validity

5. **Trajectory Characteristics**: The camera moves primarily in the horizontal plane (X-Z), with less variation in the vertical direction (Y). This pattern should be reflected in the model's predictions. 