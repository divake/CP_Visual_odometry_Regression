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
import torch.nn.functional as F
import torchvision.models as models
import math

import config


class VisualOdometryModel(nn.Module):
    """
    Visual Odometry Model that predicts relative translation between two consecutive frames.
    
    Architecture:
    - ResNet18 feature extractor (shared weights)
    - Feature fusion
    - Regression head for translation only
    
    Attributes:
        feature_extractor (nn.Module): ResNet18 feature extractor
        fc_translation (nn.Sequential): Translation regression head
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
        
        # No rotation head in this translation-only version
    
    def forward(self, img1, img2):
        """
        Forward pass through the network.
        
        Args:
            img1 (torch.Tensor): First image tensor, shape (batch_size, 3, H, W)
            img2 (torch.Tensor): Second image tensor, shape (batch_size, 3, H, W)
            
        Returns:
            tuple: (rotation, translation)
                - rotation: Dummy rotation quaternion (w, x, y, z), shape (batch_size, 4)
                - translation: Predicted translation (x, y, z), shape (batch_size, 3)
        """
        # Extract features from both images
        features1 = self.feature_extractor(img1).flatten(1)
        features2 = self.feature_extractor(img2).flatten(1)
        
        # Concatenate features
        combined_features = torch.cat([features1, features2], dim=1)
        
        # Fuse features
        fused_features = self.fusion(combined_features)
        
        # Predict translation only
        translation = self.fc_translation(fused_features)
        
        # Create dummy rotation (identity quaternion)
        batch_size = translation.size(0)
        rotation = torch.zeros((batch_size, config.ROTATION_DIM), device=translation.device)
        rotation[:, 0] = 1.0  # Set w component to 1 for identity rotation
        
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
    customization in how the two branches interact. In this version, it only predicts translation.
    
    Attributes:
        cnn (nn.Module): CNN feature extractor (shared between branches)
        fusion (nn.Module): Feature fusion layer
        fc_translation (nn.Sequential): Translation regression head
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
        
        # No rotation head in this translation-only version
    
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
                - rotation: Dummy rotation quaternion (w, x, y, z), shape (batch_size, 4)
                - translation: Predicted translation (x, y, z), shape (batch_size, 3)
        """
        # Get features from both images
        features1 = self.forward_one(img1)
        features2 = self.forward_one(img2)
        
        # Concatenate features
        combined_features = torch.cat([features1, features2], dim=1)
        
        # Fuse features
        fused_features = self.fusion(combined_features)
        
        # Predict translation only
        translation = self.fc_translation(fused_features)
        
        # Create dummy rotation (identity quaternion)
        batch_size = translation.size(0)
        rotation = torch.zeros((batch_size, config.ROTATION_DIM), device=translation.device)
        rotation[:, 0] = 1.0  # Set w component to 1 for identity rotation
        
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


class CorrelationLayer(nn.Module):
    """
    Improved correlation layer to compute relationships between feature maps.
    Based on proven approaches in optical flow networks.
    This implementation is more efficient and provides better motion estimation.
    """
    def __init__(self, max_displacement=4, stride=1):
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement
        self.stride = stride
        self.pad_size = max_displacement
        
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        
        # Pad the second tensor for efficient correlation computation
        x2_padded = F.pad(x2, [self.pad_size] * 4)
        
        # Create output tensor
        output = torch.zeros(B, (2 * self.max_displacement // self.stride + 1) ** 2 - 1, 
                            H, W, device=x1.device)
        
        kernel_size = 1
        output_idx = 0
        
        # Compute correlations for different displacement vectors
        for i in range(-self.max_displacement, self.max_displacement + 1, self.stride):
            for j in range(-self.max_displacement, self.max_displacement + 1, self.stride):
                if i == 0 and j == 0:
                    # Skip the center point to avoid redundancy
                    continue
                
                # Extract shifted x2 patch
                x2_slice = x2_padded[:, :, 
                                    self.pad_size + i:self.pad_size + i + H,
                                    self.pad_size + j:self.pad_size + j + W]
                
                # Compute correlation (dot product)
                correlation = torch.sum(x1 * x2_slice, dim=1, keepdim=True)
                
                # Normalize by feature dimension for stable gradients
                correlation = correlation / math.sqrt(C)
                
                # Add to output tensor
                output[:, output_idx:output_idx+1] = correlation
                output_idx += 1
        
        return output


class EnhancedVisualOdometryModel(nn.Module):
    """
    Enhanced Visual Odometry Model with improved feature extraction and correlation.
    This model focuses only on translation prediction with advanced architecture.
    """
    
    def __init__(self, pretrained=True):
        super(EnhancedVisualOdometryModel, self).__init__()
        
        # Use a ResNet18 with more detailed feature extraction
        resnet = models.resnet18(pretrained=pretrained)
        
        # Extract feature layers before the final pooling
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.conv2 = resnet.layer1
        self.conv3 = resnet.layer2
        self.conv4 = resnet.layer3
        
        # Feature refinement layers with residual connections
        self.refinement_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Improved correlation layer with larger displacement for better motion handling
        max_displacement = 6  # Increased from 4
        stride = 1
        self.correlation = CorrelationLayer(max_displacement=max_displacement, stride=stride)
        
        # Calculate output channels from correlation layer: (2*max_displacement//stride + 1)^2 - 1
        # Subtract 1 because we skip the center point (0,0)
        corr_output_channels = (2 * max_displacement // stride + 1) ** 2 - 1
        
        # Self-attention layer for correlation features
        self.attention = nn.Sequential(
            nn.Conv2d(corr_output_channels, corr_output_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Convolutional layers after correlation with deeper capacity
        self.conv_after_corr = nn.Sequential(
            nn.Conv2d(corr_output_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Global context module with squeeze-excitation
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Calculate feature dimensions
        high_level_dim = 256
        refined_feature_dim = 128
        corr_feature_dim = 64
        global_context_dim = 128
        
        # Scale prediction branch - separate network to specifically focus on scale
        self.scale_branch = nn.Sequential(
            nn.Linear(refined_feature_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single scale factor
            nn.Softplus()  # Ensures positive scale
        )
        
        # Total input dimension for the fully connected layer
        total_feature_dim = corr_feature_dim + refined_feature_dim * 2 + global_context_dim
        
        # Fully connected layers for translation prediction with regularization
        self.fc_translation = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(256, config.TRANSLATION_DIM)
        )
        
    def extract_features(self, x):
        """Extract hierarchical features from input"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        low_level_features = x
        x = self.conv4(x)
        high_level_features = x
        
        # Refine high-level features
        refined_features = self.refinement_conv(high_level_features)
        
        # Extract global context
        global_context = self.global_context(high_level_features)
        global_context = global_context.flatten(1)
        
        return low_level_features, refined_features, high_level_features, global_context
    
    def forward(self, img1, img2):
        """
        Forward pass through the network.
        
        Args:
            img1 (torch.Tensor): First image tensor, shape (batch_size, 3, H, W)
            img2 (torch.Tensor): Second image tensor, shape (batch_size, 3, H, W)
            
        Returns:
            tuple: (dummy_rotation, translation)
                - dummy_rotation: Dummy rotation quaternion (w, x, y, z), shape (batch_size, 4)
                - translation: Predicted translation (x, y, z), shape (batch_size, 3)
        """
        # Extract features from both images
        low_features1, refined_features1, high_features1, global_context1 = self.extract_features(img1)
        low_features2, refined_features2, high_features2, global_context2 = self.extract_features(img2)
        
        # Compute correlation between low-level features (motion information)
        corr_features = self.correlation(low_features1, low_features2)
        
        # Apply attention to focus on important correlation features
        attention_weights = self.attention(corr_features)
        corr_features = corr_features * attention_weights
        
        # Process correlation features
        corr_features = self.conv_after_corr(corr_features).flatten(1)
        
        # Process refined features (semantic information)
        refined_features1 = F.adaptive_avg_pool2d(refined_features1, (1, 1)).flatten(1)
        refined_features2 = F.adaptive_avg_pool2d(refined_features2, (1, 1)).flatten(1)
        
        # Predict scale factor from refined features
        scale_input = torch.cat([refined_features1, refined_features2], dim=1)
        scale_factor = self.scale_branch(scale_input)
        
        # Combine correlation, refined features, and global context
        combined_features = torch.cat([
            corr_features, 
            refined_features1, 
            refined_features2, 
            global_context1
        ], dim=1)
        
        # Predict translation
        translation_raw = self.fc_translation(combined_features)
        
        # Apply scale factor to ensure consistent scale
        translation = translation_raw * scale_factor
        
        # Create dummy rotation (identity quaternion)
        batch_size = translation.size(0)
        rotation = torch.zeros((batch_size, config.ROTATION_DIM), device=translation.device)
        rotation[:, 0] = 1.0  # Set w component to 1 for identity rotation
        
        return rotation, translation


def get_model(model_type="standard", pretrained=True):
    """
    Factory function to get the appropriate model.
    
    Args:
        model_type (str): Type of model to use
            - "standard": Standard visual odometry model
            - "siamese": Siamese visual odometry model
            - "enhanced": Enhanced visual odometry model with correlation
        pretrained (bool): Whether to use pretrained weights for the feature extractor
            
    Returns:
        nn.Module: Model instance
    """
    if model_type == "siamese":
        return SiameseVisualOdometryModel(pretrained=pretrained)
    elif model_type == "enhanced":
        return EnhancedVisualOdometryModel(pretrained=pretrained)
    else:  # standard
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