#!/usr/bin/env python3
"""
Loss functions for visual odometry training.

This module contains custom loss functions for the visual odometry task:
- MSE loss for translation
- Quaternion-specific loss function for rotations
- Combined loss function with weighting
- Robust translation loss functions for improved performance
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


class TranslationOnlyLoss(nn.Module):
    """
    Loss function that only focuses on translation prediction.
    
    This loss function is designed for the translation-only model variant
    where we ignore rotation predictions and only optimize for translation accuracy.
    """
    
    def __init__(self):
        """Initialize the translation-only loss."""
        super(TranslationOnlyLoss, self).__init__()
        self.translation_loss = TranslationLoss()
    
    def forward(self, pred_rotation, pred_translation, gt_rotation, gt_translation):
        """
        Calculate the translation-only loss.
        
        Args:
            pred_rotation (torch.Tensor): Predicted rotation quaternion (ignored), shape (batch_size, 4)
            pred_translation (torch.Tensor): Predicted translation, shape (batch_size, 3)
            gt_rotation (torch.Tensor): Ground truth rotation quaternion (ignored), shape (batch_size, 4)
            gt_translation (torch.Tensor): Ground truth translation, shape (batch_size, 3)
            
        Returns:
            tuple: (total_loss, translation_loss, dummy_rotation_loss, dummy_quat_norm_loss)
                - total_loss: Same as translation_loss
                - translation_loss: Mean squared error between predicted and ground truth translations
                - dummy_rotation_loss: Zero tensor (placeholder)
                - dummy_quat_norm_loss: Zero tensor (placeholder)
        """
        # Calculate translation loss
        trans_loss = self.translation_loss(pred_translation, gt_translation)
        
        # Create dummy losses for rotation and quaternion normalization
        # (to maintain compatibility with existing code)
        rot_loss = torch.tensor(0.0, device=trans_loss.device)
        quat_norm_loss = torch.tensor(0.0, device=trans_loss.device)
        
        # Total loss is just the translation loss
        total_loss = trans_loss
        
        return total_loss, trans_loss, rot_loss, quat_norm_loss


class HuberTranslationLoss(nn.Module):
    """
    Translation loss using Huber loss for robustness to outliers.
    Huber loss acts like MSE for small errors and like L1 loss for large errors,
    making it more robust to outliers.
    """
    
    def __init__(self, delta=1.0):
        """
        Initialize the Huber translation loss.
        
        Args:
            delta (float): Threshold at which to switch from quadratic to linear loss
        """
        super(HuberTranslationLoss, self).__init__()
        self.delta = delta
    
    def forward(self, pred_translation, gt_translation):
        """
        Calculate the Huber translation loss.
        
        Args:
            pred_translation (torch.Tensor): Predicted translation (x, y, z), shape (batch_size, 3)
            gt_translation (torch.Tensor): Ground truth translation (x, y, z), shape (batch_size, 3)
            
        Returns:
            torch.Tensor: Huber translation loss
        """
        # Calculate difference
        diff = pred_translation - gt_translation
        
        # Apply Huber loss
        abs_diff = torch.abs(diff)
        delta = torch.tensor(self.delta, device=diff.device)
        quadratic_mask = abs_diff <= delta
        linear_mask = ~quadratic_mask
        
        # Compute quadratic and linear terms
        loss = torch.zeros_like(diff)
        loss[quadratic_mask] = 0.5 * diff[quadratic_mask]**2
        loss[linear_mask] = delta * (abs_diff[linear_mask] - 0.5 * delta)
        
        # Return mean loss over all elements
        return torch.mean(loss)


class WeightedTranslationLoss(nn.Module):
    """
    Weighted MSE loss for translation with different weights for each axis.
    This allows for different emphasis on each dimension of translation.
    """
    
    def __init__(self, weights=None):
        """
        Initialize the weighted translation loss.
        
        Args:
            weights (list, optional): Weights for each dimension (x, y, z).
                If None, equal weights (1, 1, 1) are used.
        """
        super(WeightedTranslationLoss, self).__init__()
        
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        
        # Store weights as a tensor but don't register as buffer to avoid device issues
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, pred_translation, gt_translation):
        """
        Calculate the weighted translation loss.
        
        Args:
            pred_translation (torch.Tensor): Predicted translation (x, y, z), shape (batch_size, 3)
            gt_translation (torch.Tensor): Ground truth translation (x, y, z), shape (batch_size, 3)
            
        Returns:
            torch.Tensor: Weighted translation loss
        """
        # Calculate squared error for each dimension
        squared_error = (pred_translation - gt_translation) ** 2
        
        # Ensure weights are on the same device as input tensors
        weights = self.weights.to(squared_error.device)
        
        # Apply weights to each dimension
        weighted_squared_error = squared_error * weights.view(1, -1)
        
        # Return mean loss over all elements
        return torch.mean(weighted_squared_error)


class ScaleInvariantTranslationLoss(nn.Module):
    """
    Scale-invariant translation loss.
    This loss is less sensitive to the overall scale of the translations,
    focusing more on the relative proportions between dimensions.
    """
    
    def __init__(self, alpha=0.5, eps=1e-8):
        """
        Initialize the scale-invariant translation loss.
        
        Args:
            alpha (float): Weight for the scale-invariant term
            eps (float): Small epsilon to avoid division by zero
        """
        super(ScaleInvariantTranslationLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, pred_translation, gt_translation):
        """
        Calculate the scale-invariant translation loss.
        
        Args:
            pred_translation (torch.Tensor): Predicted translation (x, y, z), shape (batch_size, 3)
            gt_translation (torch.Tensor): Ground truth translation (x, y, z), shape (batch_size, 3)
            
        Returns:
            torch.Tensor: Scale-invariant translation loss
        """
        # Add small epsilon to avoid numerical issues with log
        eps_tensor = torch.tensor(self.eps, device=pred_translation.device)
        
        # Calculate log differences
        diff = torch.log(torch.abs(pred_translation) + eps_tensor) - torch.log(torch.abs(gt_translation) + eps_tensor)
        
        # Calculate MSE term
        mse_term = torch.mean(diff ** 2)
        
        # Calculate scale-invariant term
        scale_term = self.alpha * (torch.mean(diff) ** 2)
        
        # Also include standard MSE loss for stabilization
        direct_mse = self.mse_loss(pred_translation, gt_translation)
        
        # Combine terms
        return mse_term - scale_term + 0.1 * direct_mse


class ScaleNormalizedTranslationLoss(nn.Module):
    """
    Scale-normalized translation loss to address scale ambiguity.
    
    This loss normalizes both predicted and ground truth translations to unit length
    before computing the loss, focusing on the direction of translation rather than magnitude.
    """
    
    def __init__(self, epsilon=1e-8):
        """
        Initialize the scale-normalized translation loss.
        
        Args:
            epsilon (float): Small constant to avoid division by zero
        """
        super(ScaleNormalizedTranslationLoss, self).__init__()
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, pred_translation, gt_translation):
        """
        Calculate the scale-normalized translation loss.
        
        Args:
            pred_translation (torch.Tensor): Predicted translation (x, y, z), shape (batch_size, 3)
            gt_translation (torch.Tensor): Ground truth translation (x, y, z), shape (batch_size, 3)
            
        Returns:
            torch.Tensor: Scale-normalized translation loss
        """
        # Calculate the magnitude of translations
        pred_magnitude = torch.norm(pred_translation, p=2, dim=1, keepdim=True)
        gt_magnitude = torch.norm(gt_translation, p=2, dim=1, keepdim=True)
        
        # Normalize translations to unit length
        pred_normalized = pred_translation / (pred_magnitude + self.epsilon)
        gt_normalized = gt_translation / (gt_magnitude + self.epsilon)
        
        # Calculate direction loss (MSE between normalized vectors)
        direction_loss = self.mse_loss(pred_normalized, gt_normalized)
        
        # Calculate scale loss (relative scale difference)
        scale_ratio = pred_magnitude / (gt_magnitude + self.epsilon)
        # We want scale_ratio to be 1, so we use (scale_ratio - 1)^2
        scale_diff = scale_ratio - 1.0
        scale_loss = torch.mean(scale_diff ** 2)
        
        return direction_loss + 0.1 * scale_loss


class GeometricConsistencyLoss(nn.Module):
    """
    Geometric consistency loss to ensure the predicted motion follows geometric constraints.
    
    This loss enforces physical constraints on the predicted motion, helping to
    produce more realistic trajectories.
    """
    
    def __init__(self, epsilon=1e-8):
        """
        Initialize the geometric consistency loss.
        
        Args:
            epsilon (float): Small constant to avoid division by zero
        """
        super(GeometricConsistencyLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred_translation, gt_translation):
        """
        Calculate the geometric consistency loss.
        
        Args:
            pred_translation (torch.Tensor): Predicted translation (x, y, z), shape (batch_size, 3)
            gt_translation (torch.Tensor): Ground truth translation (x, y, z), shape (batch_size, 3)
            
        Returns:
            torch.Tensor: Geometric consistency loss
        """
        batch_size = pred_translation.size(0)
        
        # In indoor scenes, motion is mostly horizontal
        # Penalize vertical motion more heavily
        vertical_component = torch.abs(pred_translation[:, 1])  # y-axis is vertical
        vertical_penalty = torch.mean(vertical_component)
        
        # Motion smoothness constraint
        # For future frames this would involve sequential predictions
        # For now, we'll use a simplified version that encourages
        # predicted motion to have similar properties to ground truth
        
        # Motion magnitude should be similar
        pred_magnitude = torch.norm(pred_translation, p=2, dim=1)
        gt_magnitude = torch.norm(gt_translation, p=2, dim=1)
        magnitude_diff = torch.abs(pred_magnitude - gt_magnitude)
        magnitude_loss = torch.mean(magnitude_diff)
        
        return 0.5 * vertical_penalty + 0.5 * magnitude_loss


class RobustTranslationOnlyLoss(nn.Module):
    """
    Robust loss function for translation-only prediction.
    Combines multiple loss components for better performance.
    """
    
    def __init__(self, huber_weight=None, weighted_weight=None, scale_inv_weight=None,
                scale_norm_weight=0.4, geometric_weight=0.2):
        """
        Initialize the robust translation-only loss.
        
        Args:
            huber_weight (float, optional): Weight for the Huber loss component.
                If None, uses the value from config.
            weighted_weight (float, optional): Weight for the weighted MSE component.
                If None, uses the value from config.
            scale_inv_weight (float, optional): Weight for the scale-invariant component.
                If None, uses the value from config.
            scale_norm_weight (float): Weight for the scale-normalized component.
            geometric_weight (float): Weight for the geometric consistency component.
        """
        super(RobustTranslationOnlyLoss, self).__init__()
        
        # Use config values if not provided
        # Apply defensive programming to avoid AttributeError if config doesn't have these attributes
        self.huber_weight = huber_weight if huber_weight is not None else getattr(config, 'HUBER_WEIGHT', 0.4)
        self.weighted_weight = weighted_weight if weighted_weight is not None else getattr(config, 'WEIGHTED_WEIGHT', 0.2)
        self.scale_inv_weight = scale_inv_weight if scale_inv_weight is not None else getattr(config, 'SCALE_INV_WEIGHT', 0.1)
        self.scale_norm_weight = scale_norm_weight
        self.geometric_weight = geometric_weight
        
        self.huber_loss = HuberTranslationLoss(delta=0.5)
        self.weighted_loss = WeightedTranslationLoss(weights=[1.0, 1.0, 2.0])  # Higher weight for depth
        self.scale_inv_loss = ScaleInvariantTranslationLoss(alpha=0.5)
        self.scale_norm_loss = ScaleNormalizedTranslationLoss()
        self.geometric_loss = GeometricConsistencyLoss()
    
    def forward(self, pred_rotation, pred_translation, gt_rotation, gt_translation):
        """
        Calculate the robust translation-only loss.
        
        Args:
            pred_rotation (torch.Tensor): Predicted rotation quaternion (ignored), shape (batch_size, 4)
            pred_translation (torch.Tensor): Predicted translation, shape (batch_size, 3)
            gt_rotation (torch.Tensor): Ground truth rotation quaternion (ignored), shape (batch_size, 4)
            gt_translation (torch.Tensor): Ground truth translation, shape (batch_size, 3)
            
        Returns:
            tuple: (total_loss, translation_loss, dummy_rotation_loss, dummy_quat_norm_loss)
                - total_loss: Weighted combination of translation loss components
                - translation_loss: Huber loss component
                - dummy_rotation_loss: Zero tensor (placeholder)
                - dummy_quat_norm_loss: Zero tensor (placeholder)
        """
        # Calculate translation loss components
        huber_loss = self.huber_loss(pred_translation, gt_translation)
        weighted_loss = self.weighted_loss(pred_translation, gt_translation)
        scale_inv_loss = self.scale_inv_loss(pred_translation, gt_translation)
        scale_norm_loss = self.scale_norm_loss(pred_translation, gt_translation)
        geometric_loss = self.geometric_loss(pred_translation, gt_translation)
        
        # Combine translation loss components
        trans_loss = (
            self.huber_weight * huber_loss +
            self.weighted_weight * weighted_loss +
            self.scale_inv_weight * scale_inv_loss +
            self.scale_norm_weight * scale_norm_loss +
            self.geometric_weight * geometric_loss
        )
        
        # Create dummy losses for rotation and quaternion normalization
        # Ensure they're on the same device as trans_loss
        device = trans_loss.device
        rot_loss = torch.tensor(0.0, device=device)
        quat_norm_loss = torch.tensor(0.0, device=device)
        
        # Total loss is just the translation loss
        total_loss = trans_loss
        
        return total_loss, huber_loss, rot_loss, quat_norm_loss


def get_loss_function(loss_type="combined"):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        loss_type (str): Type of loss function to use
            - "combined": Combined translation and rotation loss
            - "mse": Simple MSE loss for both translation and rotation
            - "quaternion": Quaternion-specific loss for rotation, MSE for translation
            - "translation_only": Only translation loss, ignores rotation
            - "robust_translation_only": Robust translation-only loss
            
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
    elif loss_type == "translation_only":
        return TranslationOnlyLoss()
    elif loss_type == "robust_translation_only":
        return RobustTranslationOnlyLoss()
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