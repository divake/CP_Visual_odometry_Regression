#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loss functions for visual odometry.

This module implements various loss functions for training the visual odometry model:
- MSE loss for the entire pose vector
- Weighted MSE loss that balances rotation and translation components
- Custom pose loss that handles quaternions correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MSELoss(nn.Module):
    """Simple MSE loss for the entire pose vector."""
    
    def __init__(self):
        """Initialize the MSE loss."""
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target pose vectors.
        
        Args:
            pred: Predicted pose tensor of shape (B, 7)
            target: Target pose tensor of shape (B, 7)
            
        Returns:
            MSE loss
        """
        return self.mse(pred, target)


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss that balances rotation and translation components."""
    
    def __init__(self, rotation_weight: float = 10.0, translation_weight: float = 1.0):
        """
        Initialize the weighted MSE loss.
        
        Args:
            rotation_weight: Weight for rotation component (quaternion)
            translation_weight: Weight for translation component
        """
        super(WeightedMSELoss, self).__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss between predicted and target pose vectors.
        
        Args:
            pred: Predicted pose tensor of shape (B, 7)
            target: Target pose tensor of shape (B, 7)
            
        Returns:
            Weighted MSE loss
        """
        # Compute MSE for each element
        mse_loss = self.mse(pred, target)
        
        # Apply weights to rotation and translation components
        weighted_loss = torch.cat([
            mse_loss[:, :4] * self.rotation_weight,    # Rotation (quaternion)
            mse_loss[:, 4:] * self.translation_weight  # Translation
        ], dim=1)
        
        # Return mean loss
        return weighted_loss.mean()


class PoseLoss(nn.Module):
    """
    Custom pose loss that handles quaternions correctly.
    
    This loss computes:
    - L2 loss for translation component
    - Quaternion distance loss for rotation component
    """
    
    def __init__(self, rotation_weight: float = 10.0, translation_weight: float = 1.0):
        """
        Initialize the pose loss.
        
        Args:
            rotation_weight: Weight for rotation component
            translation_weight: Weight for translation component
        """
        super(PoseLoss, self).__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
    
    def quaternion_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between two quaternions.
        
        The quaternion distance is defined as 1 - |q1·q2|, where · is the dot product.
        This gives a value between 0 (identical orientation) and 1 (opposite orientation).
        
        Args:
            q1: First quaternion of shape (B, 4)
            q2: Second quaternion of shape (B, 4)
            
        Returns:
            Quaternion distance of shape (B,)
        """
        # Normalize quaternions
        q1_norm = F.normalize(q1, p=2, dim=1)
        q2_norm = F.normalize(q2, p=2, dim=1)
        
        # Compute dot product
        dot_product = torch.sum(q1_norm * q2_norm, dim=1)
        
        # Handle double cover: d(q1, q2) = min(|q1 - q2|, |q1 + q2|)
        # This is equivalent to using 1 - |dot_product|
        distance = 1.0 - torch.abs(dot_product)
        
        return distance
    
    def translation_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 distance between two translation vectors.
        
        Args:
            t1: First translation vector of shape (B, 3)
            t2: Second translation vector of shape (B, 3)
            
        Returns:
            L2 distance of shape (B,)
        """
        return torch.norm(t1 - t2, p=2, dim=1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute pose loss between predicted and target pose vectors.
        
        Args:
            pred: Predicted pose tensor of shape (B, 7)
            target: Target pose tensor of shape (B, 7)
            
        Returns:
            Tuple containing:
            - Total loss
            - Dictionary with individual loss components
        """
        # Extract rotation and translation components
        pred_rotation = pred[:, :4]
        pred_translation = pred[:, 4:]
        
        target_rotation = target[:, :4]
        target_translation = target[:, 4:]
        
        # Compute rotation and translation losses
        rotation_loss = self.quaternion_distance(pred_rotation, target_rotation)
        translation_loss = self.translation_distance(pred_translation, target_translation)
        
        # Compute weighted total loss
        total_loss = (
            self.rotation_weight * rotation_loss.mean() + 
            self.translation_weight * translation_loss.mean()
        )
        
        # Return total loss and individual components
        loss_components = {
            'rotation_loss': rotation_loss.mean(),
            'translation_loss': translation_loss.mean(),
            'total_loss': total_loss
        }
        
        return total_loss, loss_components


class ImprovedPoseLoss(nn.Module):
    """
    Improved pose loss that handles quaternions correctly.
    
    This loss computes:
    - Quaternion geodesic distance for rotation component
    - L2 loss for translation component
    """
    
    def __init__(self, rotation_weight: float = 10.0, translation_weight: float = 1.0):
        """
        Initialize the improved pose loss.
        
        Args:
            rotation_weight: Weight for rotation component
            translation_weight: Weight for translation component
        """
        super(ImprovedPoseLoss, self).__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
    
    def quaternion_geodesic_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance between two quaternions.
        
        The geodesic distance is defined as 2*arccos(|q1·q2|), which gives the
        angle between the two orientations in radians.
        
        Args:
            q1: First quaternion of shape (B, 4)
            q2: Second quaternion of shape (B, 4)
            
        Returns:
            Geodesic distance of shape (B,)
        """
        # Normalize quaternions
        q1_norm = F.normalize(q1, p=2, dim=1)
        q2_norm = F.normalize(q2, p=2, dim=1)
        
        # Compute dot product
        dot_product = torch.sum(q1_norm * q2_norm, dim=1)
        
        # Handle double cover: d(q1, q2) = min(|q1 - q2|, |q1 + q2|)
        # This is equivalent to using the absolute value of the dot product
        dot_product = torch.abs(dot_product)
        
        # Clamp dot product to [-1, 1] to avoid numerical issues
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Compute geodesic distance (angle in radians)
        distance = 2.0 * torch.acos(dot_product)
        
        return distance
    
    def translation_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 distance between two translation vectors.
        
        Args:
            t1: First translation vector of shape (B, 3)
            t2: Second translation vector of shape (B, 3)
            
        Returns:
            L2 distance of shape (B,)
        """
        return torch.norm(t1 - t2, p=2, dim=1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute pose loss between predicted and target pose vectors.
        
        Args:
            pred: Predicted pose tensor of shape (B, 7)
            target: Target pose tensor of shape (B, 7)
            
        Returns:
            Tuple containing:
            - Total loss
            - Dictionary with individual loss components
        """
        # Extract rotation and translation components
        pred_rotation = pred[:, :4]
        pred_translation = pred[:, 4:]
        
        target_rotation = target[:, :4]
        target_translation = target[:, 4:]
        
        # Compute rotation and translation losses
        rotation_loss = self.quaternion_geodesic_distance(pred_rotation, target_rotation)
        translation_loss = self.translation_distance(pred_translation, target_translation)
        
        # Compute weighted total loss
        weighted_rotation_loss = self.rotation_weight * rotation_loss
        weighted_translation_loss = self.translation_weight * translation_loss
        
        total_loss = weighted_rotation_loss.mean() + weighted_translation_loss.mean()
        
        # Return total loss and individual components
        loss_components = {
            'rotation_loss': rotation_loss.mean(),
            'translation_loss': translation_loss.mean(),
            'weighted_rotation_loss': weighted_rotation_loss.mean(),
            'weighted_translation_loss': weighted_translation_loss.mean(),
            'total_loss': total_loss
        }
        
        return total_loss, loss_components


def create_loss_function(loss_type: str, config: Dict = None) -> nn.Module:
    """
    Create a loss function based on the specified type.
    
    Args:
        loss_type: Type of loss function ('mse', 'weighted_mse', 'pose', 'improved_pose')
        config: Configuration dictionary with loss parameters
        
    Returns:
        Initialized loss function
    """
    if config is None:
        config = {}
    
    rotation_weight = config.get('rotation_weight', 10.0)
    translation_weight = config.get('translation_weight', 1.0)
    
    if loss_type == 'mse':
        return MSELoss()
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(rotation_weight, translation_weight)
    elif loss_type == 'pose':
        return PoseLoss(rotation_weight, translation_weight)
    elif loss_type == 'improved_pose':
        return ImprovedPoseLoss(rotation_weight, translation_weight)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    
    # Create random pose vectors
    pred_pose = torch.randn(batch_size, 7)
    target_pose = torch.randn(batch_size, 7)
    
    # Normalize quaternion part
    pred_pose[:, :4] = F.normalize(pred_pose[:, :4], p=2, dim=1)
    target_pose[:, :4] = F.normalize(target_pose[:, :4], p=2, dim=1)
    
    # Test MSE loss
    mse_loss = MSELoss()
    mse_result = mse_loss(pred_pose, target_pose)
    print(f"MSE Loss: {mse_result.item():.6f}")
    
    # Test Weighted MSE loss
    weighted_mse_loss = WeightedMSELoss(rotation_weight=10.0, translation_weight=1.0)
    weighted_mse_result = weighted_mse_loss(pred_pose, target_pose)
    print(f"Weighted MSE Loss: {weighted_mse_result.item():.6f}")
    
    # Test Pose loss
    pose_loss = PoseLoss(rotation_weight=10.0, translation_weight=1.0)
    total_loss, components = pose_loss(pred_pose, target_pose)
    print(f"Pose Loss: {total_loss.item():.6f}")
    print(f"  Rotation Loss: {components['rotation_loss'].item():.6f}")
    print(f"  Translation Loss: {components['translation_loss'].item():.6f}") 