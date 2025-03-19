#!/usr/bin/env python3
"""
Neural network model for visual odometry.

This module defines the neural network architecture for visual odometry:
- Uses ResNet18 as a feature extractor in a siamese architecture
- Processes image pairs to predict relative pose changes
- Has separate regression heads for translation and rotation
"""

import torch
import torch.nn as nn
import torchvision.models as models

import config


class VisualOdometryModel(nn.Module):
    """
    Visual Odometry Model that predicts relative pose between two consecutive frames.
    
    Architecture:
    - ResNet18 feature extractor (shared weights)
    - Feature fusion
    - Separate regression heads for translation and rotation
    
    Attributes:
        feature_extractor (nn.Module): ResNet18 feature extractor
        fc_translation (nn.Sequential): Translation regression head
        fc_rotation (nn.Sequential): Rotation regression head
    """
    
    def __init__(self, pretrained=True):
        """
        Initialize the model.
        
        Args:
            pretrained (bool): Whether to use pretrained ResNet18 weights
        """
        super(VisualOdometryModel, self).__init__()
        
        # Load ResNet18 without the final fc layer
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get feature dimension
        self.feature_dim = config.FEATURE_DIMENSION
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT)
        )
        
        # Regression head for translation (x, y, z)
        self.fc_translation = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(128, config.TRANSLATION_DIM)
        )
        
        # Regression head for rotation (quaternion: w, x, y, z)
        self.fc_rotation = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(128, config.ROTATION_DIM)
        )
    
    def forward(self, img1, img2):
        """
        Forward pass through the network.
        
        Args:
            img1 (torch.Tensor): First image tensor, shape (batch_size, 3, H, W)
            img2 (torch.Tensor): Second image tensor, shape (batch_size, 3, H, W)
            
        Returns:
            tuple: (rotation, translation)
                - rotation: Predicted rotation quaternion (w, x, y, z), shape (batch_size, 4)
                - translation: Predicted translation (x, y, z), shape (batch_size, 3)
        """
        # Extract features from both images
        features1 = self.feature_extractor(img1).flatten(1)
        features2 = self.feature_extractor(img2).flatten(1)
        
        # Concatenate features
        combined_features = torch.cat([features1, features2], dim=1)
        
        # Fuse features
        fused_features = self.fusion(combined_features)
        
        # Predict translation and rotation
        translation = self.fc_translation(fused_features)
        rotation = self.fc_rotation(fused_features)
        
        # Normalize quaternion
        rotation = self._normalize_quaternion(rotation)
        
        return rotation, translation
    
    def _normalize_quaternion(self, q):
        """
        Normalize quaternion to unit norm.
        
        Args:
            q (torch.Tensor): Quaternion batch, shape (batch_size, 4)
            
        Returns:
            torch.Tensor: Normalized quaternions, shape (batch_size, 4)
        """
        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        return q / (norm + 1e-8)  # Add small epsilon to avoid division by zero


class SiameseVisualOdometryModel(nn.Module):
    """
    A Siamese Network architecture for visual odometry that explicitly models the two image branches.
    
    This variant makes the siamese architecture more explicit and allows for more
    customization in how the two branches interact.
    
    Attributes:
        cnn (nn.Module): CNN feature extractor (shared between branches)
        fusion (nn.Module): Feature fusion layer
        fc_translation (nn.Sequential): Translation regression head
        fc_rotation (nn.Sequential): Rotation regression head
    """
    
    def __init__(self, pretrained=True):
        """
        Initialize the Siamese Visual Odometry model.
        
        Args:
            pretrained (bool): Whether to use pretrained weights for the feature extractor
        """
        super(SiameseVisualOdometryModel, self).__init__()
        
        # Load ResNet18 without final layer as the base CNN
        resnet = models.resnet18(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        
        # Feature dimension after CNN
        self.feature_dim = config.FEATURE_DIMENSION
        
        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT)
        )
        
        # Fully connected layers for translation prediction
        self.fc_translation = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(128, config.TRANSLATION_DIM)  # x, y, z
        )
        
        # Fully connected layers for rotation prediction
        self.fc_rotation = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(128, config.ROTATION_DIM)  # quaternion: w, x, y, z
        )
    
    def forward_one(self, x):
        """
        Forward pass for one branch of the siamese network.
        
        Args:
            x (torch.Tensor): Input image tensor, shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Image features, shape (batch_size, feature_dim)
        """
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        return x
    
    def forward(self, img1, img2):
        """
        Forward pass through the siamese network.
        
        Args:
            img1 (torch.Tensor): First image tensor, shape (batch_size, 3, H, W)
            img2 (torch.Tensor): Second image tensor, shape (batch_size, 3, H, W)
            
        Returns:
            tuple: (rotation, translation)
                - rotation: Predicted rotation quaternion (w, x, y, z), shape (batch_size, 4)
                - translation: Predicted translation (x, y, z), shape (batch_size, 3)
        """
        # Get features from both images
        features1 = self.forward_one(img1)
        features2 = self.forward_one(img2)
        
        # Concatenate features
        combined_features = torch.cat([features1, features2], dim=1)
        
        # Fuse features
        fused_features = self.fusion(combined_features)
        
        # Predict translation and rotation
        translation = self.fc_translation(fused_features)
        rotation = self.fc_rotation(fused_features)
        
        # Normalize quaternion to unit length
        rotation = self._normalize_quaternion(rotation)
        
        return rotation, translation
    
    def _normalize_quaternion(self, q):
        """
        Normalize quaternion to unit norm.
        
        Args:
            q (torch.Tensor): Quaternion batch, shape (batch_size, 4)
            
        Returns:
            torch.Tensor: Normalized quaternions, shape (batch_size, 4)
        """
        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        return q / (norm + 1e-8)  # Add small epsilon to avoid division by zero


def get_model(model_type="standard", pretrained=True):
    """
    Factory function to get a visual odometry model.
    
    Args:
        model_type (str): Type of model to use ("standard" or "siamese")
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: Visual odometry model
    """
    if model_type == "siamese":
        return SiameseVisualOdometryModel(pretrained=pretrained)
    else:
        return VisualOdometryModel(pretrained=pretrained)


if __name__ == "__main__":
    # Test the model
    model = get_model(model_type="standard", pretrained=False)
    print(model)
    
    # Create random input tensors
    batch_size = 4
    img1 = torch.randn(batch_size, 3, config.IMG_HEIGHT, config.IMG_WIDTH)
    img2 = torch.randn(batch_size, 3, config.IMG_HEIGHT, config.IMG_WIDTH)
    
    # Forward pass
    rotation, translation = model(img1, img2)
    
    # Print output shapes
    print(f"Rotation shape: {rotation.shape}")
    print(f"Translation shape: {translation.shape}")
    
    # Check quaternion normalization
    quat_norms = torch.norm(rotation, p=2, dim=1)
    print(f"Quaternion norms: {quat_norms}") 