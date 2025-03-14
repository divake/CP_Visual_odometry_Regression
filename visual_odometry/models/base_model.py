#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base model for visual odometry using a ResNet backbone.

This module implements a visual odometry model with:
- ResNet-18 backbone from torchvision (pretrained on ImageNet)
- Modified first convolutional layer to accept 4-channel input (RGB + Depth)
- Regression head to predict 7-dimensional pose vector [qw, qx, qy, qz, tx, ty, tz]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedVOModel(nn.Module):
    """Improved Visual Odometry model with separate heads for rotation and translation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the improved visual odometry model.
        
        Args:
            config: Model configuration dictionary
        """
        super(ImprovedVOModel, self).__init__()
        
        self.config = config
        
        # Initialize ResNet backbone
        self._init_backbone()
        
        # Initialize shared layers
        self.fc_shared = nn.Linear(self.feature_dim * 2, 256)
        
        # Initialize separate heads for rotation and translation
        self._init_rotation_head()
        self._init_translation_head()
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized improved {config['backbone']} model with {config['input_channels']} input channels")
    
    def _init_backbone(self):
        """Initialize the ResNet backbone with modified input layer."""
        # Select backbone architecture
        if self.config['backbone'] == 'resnet18':
            self.backbone = models.resnet18(pretrained=self.config['pretrained'])
            self.feature_dim = 512
        elif self.config['backbone'] == 'resnet34':
            self.backbone = models.resnet34(pretrained=self.config['pretrained'])
            self.feature_dim = 512
        elif self.config['backbone'] == 'resnet50':
            self.backbone = models.resnet50(pretrained=self.config['pretrained'])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {self.config['backbone']}")
        
        # Modify first convolutional layer to accept 4-channel input (RGB + Depth)
        if self.config['input_channels'] != 3:
            original_conv = self.backbone.conv1
            
            # Create new conv layer with same parameters but different in_channels
            new_conv = nn.Conv2d(
                in_channels=self.config['input_channels'],
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Copy weights from original conv for the first 3 channels
            if self.config['pretrained']:
                with torch.no_grad():
                    new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
                    
                    # Initialize the depth channel with mean of RGB channels
                    new_conv.weight[:, 3:, :, :] = original_conv.weight[:, :3, :, :].mean(dim=1, keepdim=True)
                    
                    if original_conv.bias is not None:
                        new_conv.bias = original_conv.bias.clone()
            
            # Replace the original conv layer
            self.backbone.conv1 = new_conv
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
    
    def _init_rotation_head(self):
        """Initialize the rotation head for quaternion prediction."""
        dropout_rate = self.config['dropout_rate']
        
        self.rotation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 4)  # Output quaternion [qw, qx, qy, qz]
        )
    
    def _init_translation_head(self):
        """Initialize the translation head for position prediction."""
        dropout_rate = self.config['dropout_rate']
        
        self.translation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3)  # Output translation [tx, ty, tz]
        )
    
    def _init_weights(self):
        """Initialize weights for the new layers."""
        init_method = self.config['init_method']
        
        for module in [self.fc_shared, self.rotation_head, self.translation_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    if init_method == 'xavier':
                        nn.init.xavier_uniform_(m.weight)
                    elif init_method == 'kaiming':
                        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    else:
                        raise ValueError(f"Unsupported initialization method: {init_method}")
                    
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone network.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Features tensor of shape (B, feature_dim)
        """
        features = self.backbone(x)
        return features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
    
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            data: Dictionary containing:
                - rgb1: RGB image from first frame (B, 3, H, W)
                - depth1: Depth image from first frame (B, 1, H, W)
                - rgb2: RGB image from second frame (B, 3, H, W)
                - depth2: Depth image from second frame (B, 1, H, W)
            
        Returns:
            Predicted relative pose as tensor of shape (B, 7)
            [qw, qx, qy, qz, tx, ty, tz]
        """
        # Concatenate RGB and depth channels
        x1 = torch.cat([data['rgb1'], data['depth1']], dim=1)  # (B, 4, H, W)
        x2 = torch.cat([data['rgb2'], data['depth2']], dim=1)  # (B, 4, H, W)
        
        # Extract features from both frames
        features1 = self._extract_features(x1)  # (B, feature_dim)
        features2 = self._extract_features(x2)  # (B, feature_dim)
        
        # Concatenate features from both frames
        combined_features = torch.cat([features1, features2], dim=1)  # (B, feature_dim*2)
        
        # Shared layers
        shared_features = F.relu(self.fc_shared(combined_features))
        
        # Rotation head
        rotation = self.rotation_head(shared_features)
        # Normalize quaternion
        rotation = F.normalize(rotation, p=2, dim=1)
        
        # Translation head
        translation = self.translation_head(shared_features)
        
        # Concatenate rotation and translation
        pose = torch.cat([rotation, translation], dim=1)
        
        return pose


def create_model(config: Dict) -> nn.Module:
    """
    Create a visual odometry model with the given configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized VOModel
    """
    # Use the improved model
    model = ImprovedVOModel(config)
    return model


if __name__ == "__main__":
    # Example usage
    from visual_odometry.config import MODEL_CONFIG, DEVICE
    
    # Create model
    model = create_model(MODEL_CONFIG)
    model = model.to(DEVICE)
    
    # Print model summary
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    img_height, img_width = 480, 640
    
    dummy_data = {
        'rgb1': torch.randn(batch_size, 3, img_height, img_width).to(DEVICE),
        'depth1': torch.randn(batch_size, 1, img_height, img_width).to(DEVICE),
        'rgb2': torch.randn(batch_size, 3, img_height, img_width).to(DEVICE),
        'depth2': torch.randn(batch_size, 1, img_height, img_width).to(DEVICE)
    }
    
    output = model(dummy_data)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}") 