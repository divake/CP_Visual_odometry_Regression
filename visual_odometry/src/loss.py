#!/usr/bin/env python3
"""
Loss functions for visual odometry training.

This module contains custom loss functions for the visual odometry task:
- MSE loss for translation
- Quaternion-specific loss function for rotations
- Combined loss function with weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config


class TranslationLoss(nn.Module):
    """
    Loss function for translation prediction.
    
    Uses mean squared error (MSE) between predicted and ground truth translations.
    """
    
    def __init__(self):
        """Initialize the translation loss."""
        super(TranslationLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, pred_translation, gt_translation):
        """
        Calculate the translation loss.
        
        Args:
            pred_translation (torch.Tensor): Predicted translation (x, y, z), shape (batch_size, 3)
            gt_translation (torch.Tensor): Ground truth translation (x, y, z), shape (batch_size, 3)
            
        Returns:
            torch.Tensor: Translation loss
        """
        return self.mse_loss(pred_translation, gt_translation)


class QuaternionLoss(nn.Module):
    """
    Loss function for quaternion rotation prediction.
    
    Accounts for the fact that q and -q represent the same rotation, so it uses
    the minimum loss between the two representations.
    """
    
    def __init__(self):
        """Initialize the quaternion loss."""
        super(QuaternionLoss, self).__init__()
    
    def forward(self, pred_quaternion, gt_quaternion):
        """
        Calculate the quaternion loss.
        
        Args:
            pred_quaternion (torch.Tensor): Predicted quaternion (w, x, y, z), shape (batch_size, 4)
            gt_quaternion (torch.Tensor): Ground truth quaternion (w, x, y, z), shape (batch_size, 4)
            
        Returns:
            torch.Tensor: Quaternion loss
        """
        # Make sure quaternions are normalized
        pred_quaternion = F.normalize(pred_quaternion, p=2, dim=1)
        gt_quaternion = F.normalize(gt_quaternion, p=2, dim=1)
        
        # Calculate L2 distance between quaternions
        loss_q = torch.sum((pred_quaternion - gt_quaternion) ** 2, dim=1)
        
        # Calculate L2 distance between quaternion and its negative
        loss_q_neg = torch.sum((pred_quaternion + gt_quaternion) ** 2, dim=1)
        
        # Take the minimum of the two losses (accounting for q = -q ambiguity)
        loss = torch.min(loss_q, loss_q_neg)
        
        # Return mean loss over the batch
        return torch.mean(loss)


class RotationMSELoss(nn.Module):
    """
    Alternative rotation loss using MSE between predicted and ground truth quaternions.
    """
    
    def __init__(self):
        """Initialize the rotation MSE loss."""
        super(RotationMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, pred_quaternion, gt_quaternion):
        """
        Calculate the rotation MSE loss.
        
        Args:
            pred_quaternion (torch.Tensor): Predicted quaternion (w, x, y, z), shape (batch_size, 4)
            gt_quaternion (torch.Tensor): Ground truth quaternion (w, x, y, z), shape (batch_size, 4)
            
        Returns:
            torch.Tensor: Rotation MSE loss
        """
        # Normalize quaternions
        pred_quaternion = F.normalize(pred_quaternion, p=2, dim=1)
        gt_quaternion = F.normalize(gt_quaternion, p=2, dim=1)
        
        return self.mse_loss(pred_quaternion, gt_quaternion)


class QuaternionDotProductLoss(nn.Module):
    """
    Quaternion loss based on the dot product between quaternions.
    
    This loss is based on the angular difference between quaternions,
    which is a better metric for rotation similarity.
    """
    
    def __init__(self):
        """Initialize the quaternion dot product loss."""
        super(QuaternionDotProductLoss, self).__init__()
    
    def forward(self, pred_quaternion, gt_quaternion):
        """
        Calculate the quaternion dot product loss.
        
        Args:
            pred_quaternion (torch.Tensor): Predicted quaternion (w, x, y, z), shape (batch_size, 4)
            gt_quaternion (torch.Tensor): Ground truth quaternion (w, x, y, z), shape (batch_size, 4)
            
        Returns:
            torch.Tensor: Quaternion dot product loss
        """
        # Normalize quaternions
        pred_quaternion = F.normalize(pred_quaternion, p=2, dim=1)
        gt_quaternion = F.normalize(gt_quaternion, p=2, dim=1)
        
        # Calculate absolute dot product (account for q = -q ambiguity)
        dot_product = torch.abs(torch.sum(pred_quaternion * gt_quaternion, dim=1))
        
        # The perfect match would have dot_product = 1, so we use 1 - dot_product as the loss
        loss = 1.0 - dot_product
        
        # Return mean loss over the batch
        return torch.mean(loss)


class CombinedLoss(nn.Module):
    """
    Combined loss function for visual odometry.
    
    Combines translation and rotation losses with configurable weights.
    
    Attributes:
        translation_loss (nn.Module): Loss function for translation
        rotation_loss (nn.Module): Loss function for rotation
        translation_weight (float): Weight for translation loss
        rotation_weight (float): Weight for rotation loss
        quaternion_norm_weight (float): Weight for quaternion normalization loss
    """
    
    def __init__(
        self,
        translation_loss_fn=TranslationLoss(),
        rotation_loss_fn=QuaternionDotProductLoss(),
        translation_weight=config.TRANSLATION_LOSS_WEIGHT,
        rotation_weight=config.ROTATION_LOSS_WEIGHT,
        quaternion_norm_weight=config.QUATERNION_NORM_WEIGHT
    ):
        """
        Initialize the combined loss.
        
        Args:
            translation_loss_fn (nn.Module): Loss function for translation
            rotation_loss_fn (nn.Module): Loss function for rotation
            translation_weight (float): Weight for translation loss
            rotation_weight (float): Weight for rotation loss
            quaternion_norm_weight (float): Weight for quaternion normalization loss
        """
        super(CombinedLoss, self).__init__()
        
        self.translation_loss = translation_loss_fn
        self.rotation_loss = rotation_loss_fn
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.quaternion_norm_weight = quaternion_norm_weight
    
    def forward(self, pred_rotation, pred_translation, gt_rotation, gt_translation):
        """
        Calculate the combined loss.
        
        Args:
            pred_rotation (torch.Tensor): Predicted rotation quaternion, shape (batch_size, 4)
            pred_translation (torch.Tensor): Predicted translation, shape (batch_size, 3)
            gt_rotation (torch.Tensor): Ground truth rotation quaternion, shape (batch_size, 4)
            gt_translation (torch.Tensor): Ground truth translation, shape (batch_size, 3)
            
        Returns:
            tuple: (total_loss, translation_loss, rotation_loss, quaternion_norm_loss)
        """
        # Calculate individual losses
        trans_loss = self.translation_loss(pred_translation, gt_translation)
        rot_loss = self.rotation_loss(pred_rotation, gt_rotation)
        
        # Calculate quaternion normalization loss
        quat_norms = torch.norm(pred_rotation, p=2, dim=1)
        quat_norm_loss = torch.mean((quat_norms - 1.0) ** 2)
        
        # Calculate weighted sum
        total_loss = (
            self.translation_weight * trans_loss +
            self.rotation_weight * rot_loss +
            self.quaternion_norm_weight * quat_norm_loss
        )
        
        return total_loss, trans_loss, rot_loss, quat_norm_loss


def get_loss_function(loss_type="combined"):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        loss_type (str): Type of loss function to use
            - "combined": Combined translation and rotation loss
            - "mse": Simple MSE loss for both translation and rotation
            - "quaternion": Quaternion-specific loss for rotation, MSE for translation
            
    Returns:
        nn.Module: Loss function
    """
    if loss_type == "mse":
        return CombinedLoss(
            translation_loss_fn=TranslationLoss(),
            rotation_loss_fn=RotationMSELoss()
        )
    elif loss_type == "quaternion":
        return CombinedLoss(
            translation_loss_fn=TranslationLoss(),
            rotation_loss_fn=QuaternionLoss()
        )
    else:  # combined
        return CombinedLoss(
            translation_loss_fn=TranslationLoss(),
            rotation_loss_fn=QuaternionDotProductLoss()
        )


if __name__ == "__main__":
    # Test the loss functions
    batch_size = 4
    
    # Create random input tensors
    pred_translation = torch.randn(batch_size, 3)
    gt_translation = torch.randn(batch_size, 3)
    
    pred_rotation = torch.randn(batch_size, 4)
    gt_rotation = torch.randn(batch_size, 4)
    
    # Normalize quaternions
    pred_rotation = F.normalize(pred_rotation, p=2, dim=1)
    gt_rotation = F.normalize(gt_rotation, p=2, dim=1)
    
    # Test translation loss
    translation_loss = TranslationLoss()
    t_loss = translation_loss(pred_translation, gt_translation)
    print(f"Translation loss: {t_loss.item()}")
    
    # Test rotation losses
    quaternion_loss = QuaternionLoss()
    q_loss = quaternion_loss(pred_rotation, gt_rotation)
    print(f"Quaternion loss: {q_loss.item()}")
    
    rotation_mse_loss = RotationMSELoss()
    r_mse_loss = rotation_mse_loss(pred_rotation, gt_rotation)
    print(f"Rotation MSE loss: {r_mse_loss.item()}")
    
    dot_product_loss = QuaternionDotProductLoss()
    dp_loss = dot_product_loss(pred_rotation, gt_rotation)
    print(f"Dot product loss: {dp_loss.item()}")
    
    # Test combined loss
    combined_loss = get_loss_function("combined")
    total_loss, t_loss, r_loss, qn_loss = combined_loss(
        pred_rotation, pred_translation, gt_rotation, gt_translation
    )
    print(f"Combined loss: {total_loss.item()}")
    print(f"  Translation component: {t_loss.item()}")
    print(f"  Rotation component: {r_loss.item()}")
    print(f"  Quaternion norm component: {qn_loss.item()}") 