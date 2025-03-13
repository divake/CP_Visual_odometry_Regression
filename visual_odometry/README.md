# Visual Odometry System for RGB-D Scenes Dataset v2

This project implements a complete visual odometry system using the RGB-D Scenes Dataset v2. The system uses deep learning to estimate camera poses from RGB-D image pairs.

## Project Structure

```
visual_odometry/
├── models/
│   ├── base_model.py     # ResNet-based visual odometry model
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

The visual odometry model consists of:

- ResNet-18 backbone (pretrained on ImageNet)
- Modified first convolutional layer to accept 4-channel input (RGB + Depth)
- Regression head with fully connected layers: 512*2 → 256 → 128 → 7
- Output: 7-dimensional pose vector [qw, qx, qy, qz, tx, ty, tz]
  - First 4 values for quaternion rotation
  - Last 3 values for translation

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