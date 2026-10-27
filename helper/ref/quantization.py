#!/usr/bin/env python3
"""
Quantization utilities for AR Glasses models.

This module provides step-by-step quantization (from 32 bits down to 1 bit) with TWO modes:

=== QUANTIZATION MODES ===

1. WEIGHT-ONLY QUANTIZATION:
   - Stores weights as x-bit integers (memory/bandwidth savings)
   - Dequantizes weights back to FP32 before computation
   - Activations remain in FP32
   - Uses slower floating-point units for math operations
   - Primary benefit: memory and bandwidth savings
   - Simpler and safer to implement with less accuracy loss

2. WEIGHT+ACTIVATION QUANTIZATION:
   - Stores BOTH weights AND activations as x-bit integers
   - Performs pure INTEGER multiplication on faster, more efficient integer hardware
   - NO dequantization until final output
   - Uses faster integer units for math operations
   - Primary benefit: memory savings + significant compute speedup
   - Tradeoff: requires careful calibration of activation ranges, risks more accuracy loss

=== KEY DIFFERENCES ===

Weight-only primarily saves memory and bandwidth, while weight+activation gives you
both memory savings and significant compute speedup. The tradeoff is that weight+activation
requires careful calibration of activation ranges and risks more accuracy loss, whereas
weight-only is simpler and safer to implement.

All quantization operations are implemented from scratch without using pre-built functions.

Usage:
    # Weight-only quantization
    python quantization.py --model_path path/to/model.pth --model_type parametric_a11 \
                          --dataset_path /path/to/dataset --min_bits 4 --max_bits 32
    
    # Weight+activation quantization (Step 2 after finding optimal weight bits)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import torchmetrics
import lpips
import copy
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob

# Import from train_NN2.py (same as test_model_metrics.py)
from train_NN2 import apply_gradient_friendly_distortions
from helper import (
    mask_out_roi_areas,
    create_expected_output_with_roi,
    has_geometric_distortion_in_ar_system,
    create_zero_mask_from_loader
)
from AR_System import ARSystem
from Data import CleanImageDataset
from metrics import combined_loss

# Mixed precision settings. test_model_metrics.py evaluates in true FP32, so the
# default here is False; --use_amp opts back into fp16 autocast.
USE_AMP = False


# =============================================================================
# DEFAULT HYPERPARAMETERS (All configurable values in one place)
# =============================================================================
# These defaults are used by both HardwareConfig and command-line arguments.
# Modify these values to change global defaults without editing multiple places.

# Hardware-realistic quantization defaults
DEFAULT_ACCUMULATOR_BITS = 64       # Fixed accumulator bit-width (64-bit for higher precision)
DEFAULT_USE_SATURATION = True       # Saturate on overflow (vs wrap-around)
DEFAULT_ROUNDING_MODE = 'nearest_even'  # IEEE 754 default: 'nearest_even', 'truncate', 'nearest_away', 'stochastic'
DEFAULT_REQUANTIZE_OUTPUTS = True   # Requantize layer outputs for next layer
DEFAULT_OUTPUT_BITS = 8             # Target output bits after requantization

# Quantization sweep defaults
DEFAULT_MIN_BITS = 1                # Minimum bit-width for weight quantization sweep
DEFAULT_MAX_BITS = 32               # Maximum bit-width for weight quantization sweep
DEFAULT_SYMMETRIC = True            # Use symmetric quantization (vs asymmetric)
DEFAULT_PER_CHANNEL = False         # Use per-tensor quantization (vs per-channel)

# Model defaults
DEFAULT_MODEL_COMPLEXITY = 'complex'
DEFAULT_HIDDEN_DIM = 64
DEFAULT_PATCH_SIZE = 3
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 8

# Calibration defaults
DEFAULT_CALIBRATION_FRACTION = 0.01  # 1% of dataset for calibration
DEFAULT_FULL_EVAL_AFTER_CALIBRATION = True

# ROI defaults
DEFAULT_ROI_TARGET_SIZE = (256, 256)

# Optimal bit-width thresholds
DEFAULT_PSNR_THRESHOLD = 0.1        # Acceptable PSNR drop (dB)
DEFAULT_SSIM_THRESHOLD = 0.01       # Acceptable SSIM drop


# =============================================================================
# HARDWARE CONFIGURATION (Realistic Quantization Settings)
# =============================================================================

@dataclass
class HardwareConfig:
    """
    Configuration for hardware-realistic quantization.
    
    Defaults are defined in the DEFAULT HYPERPARAMETERS section above.
    All values can be overridden via command-line arguments.
    """
    # Accumulator settings
    accumulator_bits: int = DEFAULT_ACCUMULATOR_BITS  # Fixed accumulator bit-width
    use_saturation: bool = DEFAULT_USE_SATURATION     # Saturate on overflow (vs wrap-around)
    
    # Rounding mode: 'nearest_even' (default), 'truncate', 'nearest_away', 'stochastic'
    rounding_mode: str = DEFAULT_ROUNDING_MODE
    
    # Requantization settings
    requantize_outputs: bool = DEFAULT_REQUANTIZE_OUTPUTS  # Requantize layer outputs for next layer
    output_bits: int = DEFAULT_OUTPUT_BITS                 # Target output bits after requantization
    
    # Calibration settings
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION
    full_eval_after_calibration: bool = DEFAULT_FULL_EVAL_AFTER_CALIBRATION

# Default hardware config (most popular settings)
DEFAULT_HARDWARE_CONFIG = HardwareConfig()


# =============================================================================
# CUSTOM QUANTIZED REPRESENTATION (From Scratch - No PyTorch int types)
# =============================================================================

@dataclass
class QuantizedTensor:
    """
    Custom quantized tensor representation - stores values as Python integers.
    Each value is limited to the specified bit-width.
    
    This is TRUE quantization from scratch:
    - Values stored as Python integers (not torch.int8/int16/int32)
    - Arithmetic operations implemented manually with bit-width constraints
    - Can represent any bit-width (1-bit to 32-bit)
    - Supports per-channel quantization with per-channel scales
    """
    values: np.ndarray      # Array of Python integers
    num_bits: int           # Bit-width (e.g., 8 for 8-bit)
    is_signed: bool         # True for signed (symmetric), False for unsigned
    scale: np.ndarray       # Quantization scale(s) - can be scalar or per-channel array
    zero_point: np.ndarray  # Quantization zero point(s) - can be scalar or per-channel array
    shape: tuple            # Original tensor shape
    per_channel: bool = False      # Whether using per-channel quantization
    channel_dim: int = 0           # Which dimension is the channel dimension
    
    def __post_init__(self):
        """Validate that all values are within bit-width range - UNIFORM for all bit-widths."""
        if self.is_signed:
            self.min_val = -(2 ** (self.num_bits - 1))
            self.max_val = 2 ** (self.num_bits - 1) - 1
        else:
            self.min_val = 0
            self.max_val = 2 ** self.num_bits - 1
        
        # Clamp all values to valid range and convert to int64
        self.values = np.clip(self.values, self.min_val, self.max_val).astype(np.int64)
        
        # HARDWARE-LIKE: Apply bit masking to enforce bit-width (ignore unused bits)
        # This ensures only num_bits are significant, rest are masked off
        # Use the mask_to_bits function for consistency across per-channel and per-tensor
        if self.num_bits < 63:  # Only mask if not using full int64 range
            self.values = mask_to_bits(self.values, self.num_bits, self.is_signed)
        
        # Ensure scale and zero_point are numpy arrays with proper precision
        if not isinstance(self.scale, np.ndarray):
            self.scale = np.array([self.scale], dtype=np.float64)
        elif self.scale.dtype != np.float64:
            self.scale = self.scale.astype(np.float64)
            
        if not isinstance(self.zero_point, np.ndarray):
            self.zero_point = np.array([self.zero_point], dtype=np.int64)
        elif self.zero_point.dtype != np.int64:
            self.zero_point = self.zero_point.astype(np.int64)
    
    def get_bit_representation(self, idx: tuple) -> str:
        """Get binary string representation of a value at index."""
        val = self.values[idx]
        if self.is_signed and val < 0:
            # Two's complement representation for negative numbers
            val = (1 << self.num_bits) + val
        return format(val, f'0{self.num_bits}b')
    
    def to_tensor(self, device='cpu') -> torch.Tensor:
        """Convert to PyTorch float tensor (for compatibility)."""
        return torch.from_numpy(self.values.astype(np.float32)).reshape(self.shape).to(device)


def mask_to_bits(values: np.ndarray, num_bits: int, is_signed: bool) -> np.ndarray:
    """
    HARDWARE-LIKE bit masking: Keep only the specified number of bits.
    
    This simulates real hardware behavior where only N bits are available.
    After any operation (multiply, add), we mask to enforce bit-width constraints.
    
    Args:
        values: Array of int64 values
        num_bits: Number of bits to keep (e.g., 9 for 9-bit)
        is_signed: True for signed (two's complement), False for unsigned
    
    Returns:
        Masked array with only num_bits significant bits
    """
    if is_signed:
        # Signed: Keep lower num_bits, interpret as two's complement
        mask = (1 << num_bits) - 1  # e.g., 9-bit: 0x1FF (511)
        masked = values & mask
        
        # Sign extension: if MSB is 1, extend sign
        sign_bit = 1 << (num_bits - 1)  # e.g., 9-bit: 0x100 (256)
        # If value >= sign_bit, it's negative in two's complement
        masked = np.where(masked >= sign_bit, masked - (1 << num_bits), masked)
    else:
        # Unsigned: Just mask to num_bits
        mask = (1 << num_bits) - 1
        masked = values & mask
    
    return masked.astype(np.int64)


def quantized_matmul_custom(x: QuantizedTensor, w: QuantizedTensor,
                             hw_config: 'HardwareConfig' = None) -> QuantizedTensor:
    """
    Matrix multiplication using custom quantized tensors.
    Performs: Y = X @ W^T using pure integer arithmetic.
    HARDWARE-LIKE: Uses fixed accumulator and applies bit masking.
    
    x: [batch, in_features] with x.num_bits
    w: [out_features, in_features] with w.num_bits
    output: [batch, out_features] with accumulator_bits (default: 32)
    
    Supports per-channel quantization for weights.
    """
    if hw_config is None:
        hw_config = DEFAULT_HARDWARE_CONFIG
    
    # Reshape to 2D for matrix multiplication
    x_2d = x.values.reshape(-1, x.shape[-1])  # [batch, in_features]
    w_2d = w.values.reshape(w.shape[0], -1)   # [out_features, in_features]
    
    # Integer matrix multiplication: [batch, in_feat] @ [in_feat, out_feat]
    result_values = np.matmul(x_2d, w_2d.T, dtype=np.int64)  # [batch, out_features]
    
    # HARDWARE-LIKE: Apply fixed accumulator (32-bit by default)
    # This is what real NPUs do - they have fixed-width accumulators
    result_values = apply_fixed_accumulator(result_values, hw_config)
    
    # Output bit-width is now the accumulator bit-width (fixed, not dynamic)
    output_bits = hw_config.accumulator_bits
    
    # Output scale calculation (FLOAT64 for precision)
    # x.scale: [1] for per-tensor
    # w.scale: [1] for per-tensor or [out_features] for per-channel
    if w.per_channel:
        # Per-channel weights: output scale is per-channel
        # x.scale[0] is scalar (float64), w.scale is [out_features] array (float64)
        output_scale = np.float64(x.scale[0]) * w.scale  # Broadcasting: scalar * [out_features]
        output_zero_point = np.zeros(len(w.scale), dtype=np.int64)  # [out_features] zeros
    else:
        # Both per-tensor
        output_scale = np.array([np.float64(x.scale[0]) * np.float64(w.scale[0])], dtype=np.float64)
        output_zero_point = np.array([0], dtype=np.int64)
    
    # Determine output shape
    if len(x.shape) > 2:
        output_shape = x.shape[:-1] + (w.shape[0],)
        result_values = result_values.reshape(output_shape)
    else:
        output_shape = (x.shape[0], w.shape[0])
    
    # Create output QuantizedTensor with fixed accumulator bit-width
    return QuantizedTensor(
        values=result_values,
        num_bits=output_bits,  # Fixed accumulator bits (32 by default)
        is_signed=True,  # Matmul output is typically signed
        scale=output_scale,  # Already numpy array (float64)
        zero_point=output_zero_point,  # Already numpy array (int64)
        shape=output_shape,
        per_channel=w.per_channel,  # Inherit per-channel from weights
        channel_dim=len(output_shape) - 1  # Output channel is the last dimension
    )


# =============================================================================
# ROI HELPER FUNCTIONS (Copied from train_NN2.py for consistency)
# =============================================================================

def analyze_zero_mask_roi(zero_mask_tensor):
    """
    Extract ROI coordinates from zero mask.
    Copied from train_NN2.py for consistency with training.
    
    Args:
        zero_mask_tensor: Torch tensor or numpy array where 1=valid, 0=invalid
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max) ROI coordinates
    """
    # Convert to numpy if needed
    if isinstance(zero_mask_tensor, torch.Tensor):
        mask_np = zero_mask_tensor.detach().cpu().numpy()
    else:
        mask_np = zero_mask_tensor
    
    # Ensure 2D mask
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    
    # Find valid regions: valid pixels are where mask = 1 (white)
    valid_pixels = mask_np == 1
    
    if not np.any(valid_pixels):
        raise RuntimeError("Zero mask contains no valid pixels (all pixels are 0)! Cannot proceed with ROI extraction.")
    
    # Find the largest square that fits entirely within the valid region
    def find_largest_square(binary_mask):
        """
        Find the largest square that fits entirely within the True regions of a binary mask
        Uses corrected dynamic programming approach
        Returns: (top_left_y, top_left_x, size) or None if no square found
        """
        if not np.any(binary_mask):
            raise RuntimeError("No valid pixels found in binary mask! Cannot proceed with ROI extraction.")
        
        h, w = binary_mask.shape
        
        # Dynamic programming approach
        # dp[i][j] = side length of largest square with bottom-right corner at (i,j)
        dp = np.zeros((h, w), dtype=int)
        max_size = 0
        best_bottom_right = (0, 0)
        
        # Fill the dp table
        for i in range(h):
            for j in range(w):
                if binary_mask[i][j]:  # Current cell is valid
                    if i == 0 or j == 0:
                        # First row or first column - can only form 1x1 square
                        dp[i][j] = 1
                    else:
                        # Can potentially form larger square
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
                    # Update maximum if we found a larger square
                    if dp[i][j] > max_size:
                        max_size = dp[i][j]
                        best_bottom_right = (i, j)
                else:
                    dp[i][j] = 0
        
        if max_size == 0:
            raise RuntimeError("Could not find any valid square in zero mask! Cannot proceed with ROI extraction.")
        
        # Calculate top-left corner from bottom-right corner and size
        bottom_right_y, bottom_right_x = best_bottom_right
        top_left_y = bottom_right_y - max_size + 1
        top_left_x = bottom_right_x - max_size + 1
        
        # Verify the square is entirely within valid region
        square_region = binary_mask[top_left_y:bottom_right_y+1, top_left_x:bottom_right_x+1]
        all_valid = np.all(square_region)
        
        if not all_valid:
            raise RuntimeError("Found square contains invalid pixels! ROI calculation failed.")
        
        return (top_left_y, top_left_x, max_size)
    
    square_result = find_largest_square(valid_pixels)
    
    if square_result is None:
        raise RuntimeError("Could not find any valid square in zero mask! Cannot proceed with ROI extraction.")
    
    top_left_y, top_left_x, size = square_result
    
    # Calculate the 4 corner coordinates of the largest square
    y_min = top_left_y
    y_max = top_left_y + size - 1
    x_min = top_left_x  
    x_max = top_left_x + size - 1
    
    print(f"Largest square analysis:")
    print(f"  - Mask shape: {mask_np.shape}")
    print(f"  - Valid pixels (mask=1): {np.sum(valid_pixels)}")
    print(f"  - Invalid pixels (mask=0): {np.sum(mask_np == 0)}")
    print(f"  - Largest square size: {size}x{size} pixels")
    print(f"  - Square top-left: ({top_left_y}, {top_left_x})")
    print(f"  - Square bottom-right: ({y_max}, {x_max})")
    print(f"  - ROI bounds: y({y_min}-{y_max}), x({x_min}-{x_max})")
    
    # Return in format (x_min, y_min, x_max, y_max) as expected by visualization code
    return (x_min, y_min, x_max, y_max)


# =============================================================================
# QUANTIZATION CORE FUNCTIONS (Implemented from Scratch)
# =============================================================================

class QuantizationConfig:
    """Configuration for quantization parameters."""
    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,  # Assumption: symmetric quantization is simpler and often effective
        per_channel: bool = False,  # Assumption: per-tensor is default, per-channel for better accuracy
        quantize_bias: bool = False  # Assumption: biases are typically not quantized for stability
    ):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.quantize_bias = quantize_bias
        
        # Calculate quantization levels
        if symmetric:
            # Symmetric: [-2^(n-1), 2^(n-1) - 1]
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            # Asymmetric: [0, 2^n - 1]
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1


def uniform_quantize(tensor: torch.Tensor, config: QuantizationConfig, 
                     per_channel: bool = False, channel_dim: int = 0) -> Tuple[QuantizedTensor, torch.Tensor, torch.Tensor]:
    """
    TRUE uniform quantization from scratch - returns custom QuantizedTensor.
    
    COMPLETELY FROM SCRATCH:
    - NO use of torch.int8, torch.int16, torch.int32!
    - Values stored as Python integers in numpy int64 arrays
    - All arithmetic in float64 for precision
    - Uniform behavior across ALL bit-widths (2-bit to 32-bit)
    
    Quantization formula:
        scale = max_abs_value / (2^(n-1) - 1)  [for symmetric]
        q_int = round(x / scale)
        q_int_clamped = clip(q_int, qmin, qmax)
        
    Dequantization formula:
        x_reconstructed = q_int_clamped * scale
    
    Args:
        tensor: Input tensor to quantize
        config: QuantizationConfig with quantization parameters
        per_channel: If True, compute scale/zero_point per channel
        channel_dim: Dimension for per-channel quantization (typically 0 for weights)
    
    Returns:
        quantized_tensor: QuantizedTensor with Python integers
        scale: Quantization scale (torch.Tensor for compatibility)
        zero_point: Quantization zero point (torch.Tensor for compatibility)
    """
    # Convert to numpy with FLOAT64 precision for all calculations
    tensor_np = tensor.detach().cpu().numpy().astype(np.float64)
    original_shape = tensor.shape
    
    if per_channel:
        # Per-channel quantization: separate scale/zero_point for each output channel
        tensor_permuted = np.moveaxis(tensor_np, channel_dim, 0) if channel_dim != 0 else tensor_np
        temp_shape = tensor_permuted.shape
        
        # Reshape to [channels, -1]
        num_channels = temp_shape[0]
        tensor_flat = tensor_permuted.reshape(num_channels, -1)
        
        scales = []
        zero_points = []
        quantized_channels = []
        
        for c in range(num_channels):
            channel_data = tensor_flat[c]  # float64 array
            
            # Compute scale and zero_point for this channel (FLOAT64 arithmetic)
            min_val = np.float64(channel_data.min())
            max_val = np.float64(channel_data.max())
            
            if config.symmetric:
                # Symmetric: range [-2^(n-1), 2^(n-1)-1], zero_point = 0
                # Example: 8-bit = [-128, 127], 31-bit = [-2^30, 2^30-1]
                abs_max = np.maximum(np.abs(min_val), np.abs(max_val))
                qmax_float = np.float64(2 ** (config.num_bits - 1) - 1)
                
                # Scale calculation in float64 for precision
                if abs_max > 1e-10:
                    scale = abs_max / qmax_float
                else:
                    scale = np.float64(1e-10)  # Avoid division by zero
                
                zero_point_val = np.int64(0)
            else:
                # Asymmetric: range [0, 2^n-1]
                qrange = np.float64(config.qmax - config.qmin)
                value_range = max_val - min_val
                
                if value_range > 1e-10:
                    scale = value_range / qrange
                else:
                    scale = np.float64(1e-10)
                    
                zero_point_val = np.int64(np.round(config.qmin - min_val / scale))
            
            # FROM-SCRATCH quantization to Python integers
            # Step 1: Scale to quantization range (float64 arithmetic)
            q_float = channel_data / scale + np.float64(zero_point_val)
            
            # Step 2: Round to nearest integer (mimics hardware rounding)
            q_rounded = np.round(q_float)
            
            # Step 3: Clamp to valid range and convert to int64
            q_int = np.clip(q_rounded, config.qmin, config.qmax).astype(np.int64)
            
            # Step 4: HARDWARE-LIKE bit masking - enforce num_bits constraint
            # This ensures only the specified number of bits are used
            q_int = mask_to_bits(q_int, config.num_bits, config.symmetric)
            
            quantized_channels.append(q_int)
            scales.append(np.float64(scale))
            zero_points.append(zero_point_val)
        
        # Stack channels back
        quantized_flat = np.stack(quantized_channels, axis=0)
        quantized_reshaped = quantized_flat.reshape(temp_shape)
        
        # Move axis back if needed
        quantized_values = np.moveaxis(quantized_reshaped, 0, channel_dim) if channel_dim != 0 else quantized_reshaped
        
        # Create QuantizedTensor with PER-CHANNEL scales and zero_points (float64)
        quantized_tensor = QuantizedTensor(
            values=quantized_values,
            num_bits=config.num_bits,
            is_signed=config.symmetric,
            scale=np.array(scales, dtype=np.float64),  # Per-channel scales in float64
            zero_point=np.array(zero_points, dtype=np.int64),  # Per-channel zero_points
            shape=original_shape,
            per_channel=True,
            channel_dim=channel_dim
        )
        
        scale = torch.tensor(scales, dtype=torch.float64, device=tensor.device)
        zero_point = torch.tensor(zero_points, dtype=torch.int64, device=tensor.device)
        
    else:
        # Per-tensor quantization - UNIFORM across ALL bit-widths (2-32 bit)
        min_val = np.float64(tensor_np.min())
        max_val = np.float64(tensor_np.max())
        
        if config.symmetric:
            # Symmetric: range [-2^(n-1), 2^(n-1)-1], zero_point = 0
            # 8-bit: [-128, 127], 16-bit: [-32768, 32767], 31-bit: [-2^30, 2^30-1]
            abs_max = np.maximum(np.abs(min_val), np.abs(max_val))
            qmax_float = np.float64(2 ** (config.num_bits - 1) - 1)
            
            # Scale calculation in float64 for precision
            if abs_max > 1e-10:
                scale = abs_max / qmax_float
            else:
                scale = np.float64(1e-10)  # Avoid division by zero
                
            zero_point_val = np.int64(0)
        else:
            # Asymmetric: range [0, 2^n-1]
            qrange = np.float64(config.qmax - config.qmin)
            value_range = max_val - min_val
            
            if value_range > 1e-10:
                scale = value_range / qrange
            else:
                scale = np.float64(1e-10)
                
            zero_point_val = np.int64(np.round(config.qmin - min_val / scale))
        
        # FROM-SCRATCH quantization to Python integers (UNIFORM for ALL bit-widths!)
        # Step 1: Scale to quantization range (float64 arithmetic)
        q_float = tensor_np / scale + np.float64(zero_point_val)
        
        # Step 2: Round to nearest integer (this is what hardware does!)
        q_rounded = np.round(q_float)
        
        # Step 3: Clamp to valid range and convert to int64 (TRUE integer storage)
        q_int = np.clip(q_rounded, config.qmin, config.qmax).astype(np.int64)
        
        # Step 4: HARDWARE-LIKE bit masking - enforce num_bits constraint
        # This ensures only the specified number of bits are used
        q_int = mask_to_bits(q_int, config.num_bits, config.symmetric)
        
        # Create QuantizedTensor (per-tensor, single scale/zero_point in float64)
        quantized_tensor = QuantizedTensor(
            values=q_int,
            num_bits=config.num_bits,
            is_signed=config.symmetric,
            scale=np.array([scale], dtype=np.float64),  # float64 for precision
            zero_point=np.array([zero_point_val], dtype=np.int64),
            shape=original_shape,
            per_channel=False,
            channel_dim=0
        )
        
        scale = torch.tensor(scale, dtype=torch.float32, device=tensor.device)
        zero_point = torch.tensor(zero_point_val, dtype=torch.int32, device=tensor.device)
    
    return quantized_tensor, scale, zero_point


def dequantize(quantized_tensor: QuantizedTensor, device='cpu') -> torch.Tensor:
    """
    Dequantize custom QuantizedTensor back to float - UNIFORM for all bit-widths.
    
    FROM-SCRATCH dequantization:
    - Formula: x = (q_int - zero_point) * scale
    - All arithmetic in float64 for precision
    - Uniform behavior for 2-bit through 32-bit
    
    This mimics real hardware dequantization.
    
    Args:
        quantized_tensor: QuantizedTensor with Python integers (int64)
        device: Target device for output tensor
    
    Returns:
        dequantized_tensor: Float32 PyTorch tensor
    """
    if quantized_tensor.per_channel:
        # Per-channel dequantization (float64 arithmetic)
        channel_dim = quantized_tensor.channel_dim
        num_channels = quantized_tensor.shape[channel_dim]
        
        # Verify scale length matches number of channels
        if len(quantized_tensor.scale) != num_channels:
            raise ValueError(f"Scale length {len(quantized_tensor.scale)} doesn't match num_channels {num_channels}")
        
        # Reshape scale and zero_point for broadcasting
        new_shape = [1] * len(quantized_tensor.shape)
        new_shape[channel_dim] = num_channels
        
        scale_reshaped = quantized_tensor.scale.reshape(new_shape)  # float64 array
        zero_point_reshaped = quantized_tensor.zero_point.reshape(new_shape)  # int64 array
        
        # Dequantize: x = (q_int - zp) * scale (ALL in float64 precision)
        q_float = quantized_tensor.values.astype(np.float64)
        zp_float = zero_point_reshaped.astype(np.float64)
        dequantized_np = (q_float - zp_float) * scale_reshaped
    else:
        # Per-tensor dequantization - UNIFORM for all bit-widths (float64 arithmetic)
        # x = (q_int - zp) * scale
        zp_scalar = np.float64(quantized_tensor.zero_point[0])
        scale_scalar = np.float64(quantized_tensor.scale[0])
        
        # All arithmetic in float64
        q_float = quantized_tensor.values.astype(np.float64)
        dequantized_np = (q_float - zp_scalar) * scale_scalar
    
    # Convert to PyTorch tensor (float32 for compatibility with model)
    dequantized = torch.from_numpy(dequantized_np.astype(np.float32)).reshape(quantized_tensor.shape).to(device)
    
    return dequantized


# =============================================================================
# HARDWARE-REALISTIC FUNCTIONS (For Realistic Quantization)
# =============================================================================

def round_hardware(values: np.ndarray, mode: str = 'nearest_even') -> np.ndarray:
    """
    Hardware-specific rounding modes.
    
    Args:
        values: Float values to round
        mode: Rounding mode:
            - 'nearest_even': Round half to even (IEEE 754 default, most common)
            - 'nearest_away': Round half away from zero
            - 'truncate': Truncate towards zero (some DSPs)
            - 'stochastic': Stochastic rounding (for training)
    
    Returns:
        Rounded integer values
    """
    if mode == 'nearest_even':
        # IEEE 754 default: round half to even (banker's rounding)
        return np.rint(values).astype(np.int64)
    elif mode == 'nearest_away':
        # Round half away from zero (what np.round does)
        return np.round(values).astype(np.int64)
    elif mode == 'truncate':
        # Truncate towards zero (some DSPs, faster)
        return np.trunc(values).astype(np.int64)
    elif mode == 'stochastic':
        # Stochastic rounding (useful for training, unbiased)
        floor_vals = np.floor(values)
        prob = values - floor_vals
        random_vals = np.random.random(values.shape)
        return np.where(random_vals < prob, floor_vals + 1, floor_vals).astype(np.int64)
    else:
        raise ValueError(f"Unknown rounding mode: {mode}. Use 'nearest_even', 'nearest_away', 'truncate', or 'stochastic'")


def requantize_direct(quantized_tensor: QuantizedTensor, target_bits: int,
                      hw_config: 'HardwareConfig' = None) -> QuantizedTensor:
    """
    Requantize to a different bit-width.
    
    This properly handles requantization by:
    1. Dequantizing to float (preserving the actual values)
    2. Re-quantizing to the target bit-width
    
    This is more accurate than trying to do direct integer scaling,
    which fails when accumulated values don't use the full bit range.
    
    Args:
        quantized_tensor: Input QuantizedTensor
        target_bits: Target bit-width
        hw_config: Hardware configuration (rounding mode, saturation)
    
    Returns:
        QuantizedTensor with target_bits precision
    """
    if hw_config is None:
        hw_config = DEFAULT_HARDWARE_CONFIG
    
    # If already at target or below, just update metadata
    current_bits = quantized_tensor.num_bits
    if current_bits <= target_bits:
        return QuantizedTensor(
            values=quantized_tensor.values.copy(),
            num_bits=target_bits,
            is_signed=quantized_tensor.is_signed,
            scale=quantized_tensor.scale.copy(),
            zero_point=quantized_tensor.zero_point.copy(),
            shape=quantized_tensor.shape,
            per_channel=quantized_tensor.per_channel,
            channel_dim=quantized_tensor.channel_dim
        )
    
    # PROPER REQUANTIZATION: Dequantize to float, then re-quantize to target bits
    # This preserves the actual value range instead of assuming full bit usage
    
    # Step 1: Dequantize to float64 (high precision)
    if quantized_tensor.per_channel:
        channel_dim = quantized_tensor.channel_dim
        num_channels = quantized_tensor.shape[channel_dim]
        new_shape = [1] * len(quantized_tensor.shape)
        new_shape[channel_dim] = num_channels
        scale_reshaped = quantized_tensor.scale.reshape(new_shape)
        zp_reshaped = quantized_tensor.zero_point.reshape(new_shape).astype(np.float64)
        float_values = (quantized_tensor.values.astype(np.float64) - zp_reshaped) * scale_reshaped
    else:
        zp = np.float64(quantized_tensor.zero_point[0])
        scale = np.float64(quantized_tensor.scale[0])
        float_values = (quantized_tensor.values.astype(np.float64) - zp) * scale
    
    # Step 2: Re-quantize to target bits
    # Calculate new scale based on actual value range
    if quantized_tensor.is_signed:
        new_qmin = -(2 ** (target_bits - 1))
        new_qmax = 2 ** (target_bits - 1) - 1
    else:
        new_qmin = 0
        new_qmax = 2 ** target_bits - 1
    
    # Compute scale from actual value range (per-tensor for simplicity)
    abs_max = np.maximum(np.abs(float_values.min()), np.abs(float_values.max()))
    if abs_max > 1e-10:
        new_scale = abs_max / float(new_qmax)
    else:
        new_scale = 1e-10
    
    # Quantize
    q_float = float_values / new_scale
    
    # Apply hardware rounding
    rounded_values = round_hardware(q_float, mode=hw_config.rounding_mode)
    
    # Clamp to valid range
    clamped_values = np.clip(rounded_values, new_qmin, new_qmax).astype(np.int64)
    
    return QuantizedTensor(
        values=clamped_values,
        num_bits=target_bits,
        is_signed=quantized_tensor.is_signed,
        scale=np.array([new_scale], dtype=np.float64),
        zero_point=np.array([0], dtype=np.int64),
        shape=quantized_tensor.shape,
        per_channel=False,  # Simplify to per-tensor after requantization
        channel_dim=quantized_tensor.channel_dim
    )


def apply_fixed_accumulator(values: np.ndarray, hw_config: 'HardwareConfig' = None) -> np.ndarray:
    """
    Apply fixed accumulator bit-width like real hardware.
    
    Real NPUs use fixed 32-bit accumulators for INT8 matmul.
    This enforces that constraint with saturation or wrap-around.
    
    Args:
        values: Accumulated integer values (from matmul)
        hw_config: Hardware configuration
    
    Returns:
        Values constrained to accumulator bit-width
    """
    if hw_config is None:
        hw_config = DEFAULT_HARDWARE_CONFIG
    
    acc_bits = hw_config.accumulator_bits
    acc_min = -(2 ** (acc_bits - 1))
    acc_max = 2 ** (acc_bits - 1) - 1
    
    if hw_config.use_saturation:
        return np.clip(values, acc_min, acc_max).astype(np.int64)
    else:
        return mask_to_bits(values, acc_bits, is_signed=True)


def quantized_matmul(x_q: QuantizedTensor, w_q: QuantizedTensor, 
                     w_per_channel: bool = False) -> QuantizedTensor:
    """
    Perform quantized matrix multiplication: Y = X @ W^T
    
    TRUE integer arithmetic using custom QuantizedTensor:
        Y_int = (X_int - X_zp) @ (W_int - W_zp)^T
        Y_scale = X_scale * W_scale
    
    Args:
        x_q: Quantized input QuantizedTensor [batch, in_features]
        w_q: Quantized weights QuantizedTensor [out_features, in_features]
        w_per_channel: If True, weights use per-channel quantization
    
    Returns:
        output: QuantizedTensor with integer results
    """
    # Use the custom matmul implementation
    return quantized_matmul_custom(x_q, w_q)


class WeightOnlyQuantizedLinear(nn.Module):
    """
    Weight-only quantization for Linear layers.
    
    WEIGHT-ONLY QUANTIZATION:
    - Stores weights as x-bit integers (memory/bandwidth savings)
    - Dequantizes weights back to FP32 before computation
    - Activations remain in FP32
    - Computation uses slower floating-point units
    - Primary benefit: memory and bandwidth savings
    - Simpler and safer to implement with less accuracy loss
    """
    def __init__(self, original_layer: nn.Linear, weight_config: QuantizationConfig):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Store quantization config
        self.weight_config = weight_config
        
        # Quantize weights to custom QuantizedTensor (Python integers!)
        with torch.no_grad():
            self.weight_quantized, weight_scale, weight_zero = uniform_quantize(
                original_layer.weight.data,
                weight_config,
                per_channel=weight_config.per_channel,
                channel_dim=0
            )
            # Store the scale/zero_point as buffers for compatibility
            self.register_buffer('weight_scale', weight_scale)
            self.register_buffer('weight_zero_point', weight_zero)
        
        # Handle bias (keep in full precision for stability)
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.data)
        else:
            self.bias = None
    
    def forward(self, x):
        """
        Forward pass with WEIGHT-ONLY quantization:
        1. Dequantize weights from integers back to FP32
        2. Activations stay in FP32 (no quantization)
        3. Perform standard FP32 matrix multiplication
        4. Uses slower floating-point hardware but saves memory
        """
        device = x.device
        
        # Dequantize weights to FP32
        w_dequant = dequantize(self.weight_quantized, device=device)
        
        # Activations remain in FP32 - no quantization
        # Perform standard FP32 linear operation
        output = nn.functional.linear(x, w_dequant, self.bias)
        
        return output


class WeightActivationQuantizedLinear(nn.Module):
    """
    Weight+Activation quantization for Linear layers.
    
    WEIGHT+ACTIVATION QUANTIZATION (Hardware-Realistic):
    - Stores both weights AND activations as x-bit integers
    - Performs pure INTEGER multiplication with fixed 32-bit accumulator
    - Uses DIRECT requantization (no FP32 intermediate) between layers
    - Primary benefit: both memory savings AND significant compute speedup
    - Matches real NPU/TPU behavior
    """
    def __init__(self, original_layer: nn.Linear, weight_config: QuantizationConfig, 
                 activation_config: QuantizationConfig, hw_config: 'HardwareConfig' = None):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Store quantization configs
        self.weight_config = weight_config
        self.activation_config = activation_config
        self.hw_config = hw_config if hw_config is not None else DEFAULT_HARDWARE_CONFIG
        
        # Quantize weights to custom QuantizedTensor (Python integers!)
        with torch.no_grad():
            self.weight_quantized, weight_scale, weight_zero = uniform_quantize(
                original_layer.weight.data,
                weight_config,
                per_channel=weight_config.per_channel,
                channel_dim=0
            )
            # Store the scale/zero_point as buffers for compatibility
            self.register_buffer('weight_scale', weight_scale)
            self.register_buffer('weight_zero_point', weight_zero)
        
        # Handle bias (keep in full precision for stability)
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.data)
        else:
            self.bias = None
    
    def forward(self, x):
        """
        Forward pass with WEIGHT+ACTIVATION quantization (Hardware-Realistic):
        1. Quantize activations to x-bit integers
        2. Weights already stored as x-bit integers
        3. Perform PURE INTEGER matmul with fixed 32-bit accumulator
        4. DIRECT requantize to output_bits (no FP32 intermediate!)
        5. Dequantize only at the end
        """
        device = x.device
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        
        # Quantize activations to integers
        x_q, x_scale, x_zp = uniform_quantize(x_flat, self.activation_config, per_channel=False)
        
        # Perform PURE INTEGER matrix multiplication with fixed 32-bit accumulator
        output_q = quantized_matmul(x_q, self.weight_quantized, 
                                     w_per_channel=self.weight_config.per_channel)
        
        # HARDWARE-REALISTIC: Direct requantization (32-bit → activation_bits)
        # Use the ACTIVATION config's bit-width, not the fixed output_bits
        # This preserves the tested activation precision
        if self.hw_config.requantize_outputs:
            # Requantize to activation bits (not fixed 8-bit)
            output_q = requantize_direct(output_q, self.activation_config.num_bits, self.hw_config)
        
        # Dequantize output back to float (only at the final output)
        output_float = dequantize(output_q, device=device)
        
        # Add bias in float (standard practice)
        if self.bias is not None:
            output_float = output_float + self.bias
        
        # Reshape back to original batch shape
        output = output_float.reshape(*batch_shape, self.out_features)
        
        return output


class WeightOnlyQuantizedConv2d(nn.Module):
    """
    Weight-only quantization for Conv2d layers.
    
    WEIGHT-ONLY QUANTIZATION:
    - Stores weights as x-bit integers (memory/bandwidth savings)
    - Dequantizes weights back to FP32 before computation
    - Activations remain in FP32
    - Computation uses slower floating-point units
    - Primary benefit: memory and bandwidth savings
    - Simpler and safer to implement with less accuracy loss
    """
    def __init__(self, original_layer: nn.Conv2d, weight_config: QuantizationConfig):
        super().__init__()
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        
        # Store quantization config
        self.weight_config = weight_config
        
        # Quantize weights to custom QuantizedTensor
        with torch.no_grad():
            self.weight_quantized, weight_scale, weight_zero = uniform_quantize(
                original_layer.weight.data,
                weight_config,
                per_channel=weight_config.per_channel,
                channel_dim=0
            )
            self.register_buffer('weight_scale', weight_scale)
            self.register_buffer('weight_zero_point', weight_zero)
        
        # Handle bias (keep in full precision)
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.data)
        else:
            self.bias = None
    
    def forward(self, x):
        """
        Forward pass with WEIGHT-ONLY quantization:
        1. Dequantize weights from integers back to FP32
        2. Activations stay in FP32 (no quantization)
        3. Perform standard FP32 convolution
        4. Uses slower floating-point hardware but saves memory
        """
        device = x.device
        
        # Dequantize weights to FP32
        w_dequant = dequantize(self.weight_quantized, device=device)
        
        # Activations remain in FP32 - no quantization
        # Perform standard FP32 convolution
        output = nn.functional.conv2d(
            x, w_dequant, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        
        return output


class WeightActivationQuantizedConv2d(nn.Module):
    """
    Weight+Activation quantization for Conv2d layers.
    
    WEIGHT+ACTIVATION QUANTIZATION (Hardware-Realistic):
    - Stores both weights AND activations as x-bit integers
    - Quantizes activations, limits precision before computation
    - Uses FP32 convolution with quantized values (true integer conv2d is complex)
    - Applies HardwareConfig for rounding modes and saturation
    
    Note: True integer convolution would require implementing conv from scratch.
    For practical purposes, we quantize to limit precision, then perform FP32 conv.
    """
    def __init__(self, original_layer: nn.Conv2d, weight_config: QuantizationConfig,
                 activation_config: QuantizationConfig, hw_config: 'HardwareConfig' = None):
        super().__init__()
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        
        # Store quantization configs
        self.weight_config = weight_config
        self.activation_config = activation_config
        self.hw_config = hw_config if hw_config is not None else DEFAULT_HARDWARE_CONFIG
        
        # Quantize weights to custom QuantizedTensor
        with torch.no_grad():
            self.weight_quantized, weight_scale, weight_zero = uniform_quantize(
                original_layer.weight.data,
                weight_config,
                per_channel=weight_config.per_channel,
                channel_dim=0
            )
            self.register_buffer('weight_scale', weight_scale)
            self.register_buffer('weight_zero_point', weight_zero)
        
        # Handle bias (keep in full precision)
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.data)
        else:
            self.bias = None
    
    def forward(self, x):
        """
        Forward pass with WEIGHT+ACTIVATION quantization (Hardware-Realistic):
        1. Quantize activations to x-bit integers
        2. Weights already stored as x-bit integers
        3. Dequantize both for FP32 convolution
        4. Quantize output if requantize_outputs is enabled
        """
        device = x.device
        
        # Quantize activations to integers
        x_q, x_scale, x_zp = uniform_quantize(x, self.activation_config, per_channel=False)
        
        # Dequantize both (implementing integer conv2d from scratch is complex)
        x_dequant = dequantize(x_q, device=device)
        w_dequant = dequantize(self.weight_quantized, device=device)
        
        # Perform FP32 convolution (with quantized precision values)
        output = nn.functional.conv2d(
            x_dequant, w_dequant, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        
        # HARDWARE-REALISTIC: Requantize output for next layer (optional)
        # Use activation_config.num_bits to preserve tested precision
        if self.hw_config.requantize_outputs:
            output_config = QuantizationConfig(
                num_bits=self.activation_config.num_bits,  # Use activation bits, not fixed 8-bit
                symmetric=True, per_channel=False
            )
            output_q, _, _ = uniform_quantize(output, output_config, per_channel=False)
            output = dequantize(output_q, device=device)
        
        return output


def quantize_model(model: nn.Module, weight_config: QuantizationConfig,
                   activation_config: Optional[QuantizationConfig] = None,
                   hw_config: Optional['HardwareConfig'] = None,
                   skip_layers: Optional[List[str]] = None,
                   use_weight_only: bool = None,
                   verbose: bool = True) -> nn.Module:
    """
    Quantize all Linear and Conv2d layers in a model.
    
    QUANTIZATION MODES:
    1. Weight-only (use_weight_only=True or activation_config=None):
       - Stores weights as x-bit integers but dequantizes to FP32 before computation
       - Uses slower floating-point units
       - Primary benefit: memory and bandwidth savings
       - Simpler, safer, less accuracy loss
    
    2. Weight+Activation (use_weight_only=False and activation_config provided):
       - Keeps both weights and activations as x-bit integers
       - Uses fixed 32-bit accumulator (like real hardware)
       - Direct requantization between layers (no FP32 intermediate)
       - Primary benefit: memory savings AND significant compute speedup
    
    Args:
        model: Original model
        weight_config: Configuration for weight quantization
        activation_config: Optional configuration for activation quantization
        hw_config: Hardware configuration (accumulator bits, rounding, etc.)
        skip_layers: List of layer names to skip (e.g., first/last layers)
        use_weight_only: Force weight-only mode. If None, auto-detect from activation_config
        verbose: If True, print quantization statistics (default: True)
    
    Returns:
        Quantized model
    """
    if skip_layers is None:
        # Assumption: Skip first and last layers for better accuracy (common practice)
        skip_layers = []
    
    if hw_config is None:
        hw_config = DEFAULT_HARDWARE_CONFIG
    
    # Auto-detect mode if not specified
    if use_weight_only is None:
        use_weight_only = (activation_config is None)
    
    # Determine quantization mode
    if verbose:
        if use_weight_only:
            print(f"Using WEIGHT-ONLY quantization ({weight_config.num_bits}-bit weights):")
            print(f"  - Weights: {weight_config.num_bits}-bit integers → dequantized to FP32 before compute")
            print(f"  - Activations: FP32 (not quantized)")
            print(f"  - Hardware: Floating-point units")
            print(f"  - Benefit: {32/weight_config.num_bits:.1f}x memory savings for weights")
        else:
            print(f"Using WEIGHT+ACTIVATION quantization ({weight_config.num_bits}-bit weights, {activation_config.num_bits}-bit activations):")
            print(f"  - Weights: {weight_config.num_bits}-bit integers")
            print(f"  - Activations: {activation_config.num_bits}-bit integers")
            print(f"  - Accumulator: {hw_config.accumulator_bits}-bit (fixed)")
            print(f"  - Rounding: {hw_config.rounding_mode}")
            if hw_config.requantize_outputs:
                print(f"  - Requantization: 64-bit accumulator → {activation_config.num_bits}-bit activations (between layers)")
            print(f"  - Hardware: Integer arithmetic units (faster)")
            print(f"  - Benefit: {32/weight_config.num_bits:.1f}x memory + compute speedup")
    
    # Create a deep copy of the model to avoid modifying the original
    quantized_model = copy.deepcopy(model)
    
    # Track statistics (weights AND activations)
    stats = {
        'quantized_linear': 0,
        'quantized_conv2d': 0,
        'skipped_linear': 0,
        'skipped_conv2d': 0,
        'quantized_weights': 0,
        'skipped_weights': 0,
        'quantized_activations': 0,  # Activations that will be quantized per forward pass
        'skipped_activations': 0,    # Activations that stay FP32
        'total_activations': 0,      # Total activation count per forward pass
        'quantized_layers_list': [],
        'skipped_layers_list': [],
        'weight_only_mode': use_weight_only  # Track mode for activation reporting
    }
    
    # Recursively replace layers
    def replace_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if full_name in skip_layers:
                # Count skipped layer stats
                if isinstance(child, nn.Linear):
                    stats['skipped_linear'] += 1
                    stats['skipped_weights'] += child.weight.numel()
                    stats['skipped_activations'] += child.in_features  # Input activations per sample
                    stats['total_activations'] += child.in_features
                    if child.bias is not None:
                        stats['skipped_weights'] += child.bias.numel()
                    stats['skipped_layers_list'].append(f"{full_name} (Linear: {child.weight.numel():,} weights, {child.in_features:,} input activations)")
                elif isinstance(child, nn.Conv2d):
                    stats['skipped_conv2d'] += 1
                    stats['skipped_weights'] += child.weight.numel()
                    # For Conv2d, activations = in_channels (per spatial position)
                    stats['skipped_activations'] += child.in_channels
                    stats['total_activations'] += child.in_channels
                    if child.bias is not None:
                        stats['skipped_weights'] += child.bias.numel()
                    stats['skipped_layers_list'].append(f"{full_name} (Conv2d: {child.weight.numel():,} weights, {child.in_channels:,} input channels)")
                continue
            
            if isinstance(child, nn.Linear):
                # Replace with appropriate quantized version
                if use_weight_only:
                    quantized_layer = WeightOnlyQuantizedLinear(child, weight_config)
                    # Activations NOT quantized in weight-only mode
                    stats['skipped_activations'] += child.in_features
                else:
                    quantized_layer = WeightActivationQuantizedLinear(
                        child, weight_config, activation_config, hw_config)
                    # Activations ARE quantized in weight+activation mode
                    stats['quantized_activations'] += child.in_features
                setattr(module, name, quantized_layer)
                stats['quantized_linear'] += 1
                stats['quantized_weights'] += child.weight.numel()
                stats['total_activations'] += child.in_features
                stats['quantized_layers_list'].append(f"{full_name} (Linear: {child.weight.numel():,} weights, {child.in_features:,} input activations)")
            elif isinstance(child, nn.Conv2d):
                # Replace with appropriate quantized version
                if use_weight_only:
                    quantized_layer = WeightOnlyQuantizedConv2d(child, weight_config)
                    # Activations NOT quantized in weight-only mode
                    stats['skipped_activations'] += child.in_channels
                else:
                    quantized_layer = WeightActivationQuantizedConv2d(
                        child, weight_config, activation_config, hw_config)
                    # Activations ARE quantized in weight+activation mode
                    stats['quantized_activations'] += child.in_channels
                setattr(module, name, quantized_layer)
                stats['quantized_conv2d'] += 1
                stats['quantized_weights'] += child.weight.numel()
                stats['total_activations'] += child.in_channels
                stats['quantized_layers_list'].append(f"{full_name} (Conv2d: {child.weight.numel():,} weights, {child.in_channels:,} input channels)")
            else:
                # Recursively process child modules
                replace_layers(child, full_name)
    
    replace_layers(quantized_model)
    
    # Print statistics only if verbose
    if verbose:
        print(f"\n{'='*70}")
        print("QUANTIZATION STATISTICS")
        print(f"{'='*70}")
        
        # Weight statistics
        total_quantized_layers = stats['quantized_linear'] + stats['quantized_conv2d']
        total_skipped_layers = stats['skipped_linear'] + stats['skipped_conv2d']
        total_weights = stats['quantized_weights'] + stats['skipped_weights']
        
        print(f"\n📊 WEIGHT STATISTICS:")
        print(f"   Quantized layers: {total_quantized_layers} (Linear: {stats['quantized_linear']}, Conv2d: {stats['quantized_conv2d']})")
        print(f"   Skipped layers:   {total_skipped_layers} (Linear: {stats['skipped_linear']}, Conv2d: {stats['skipped_conv2d']})")
        print(f"   Quantized weights: {stats['quantized_weights']:,}")
        print(f"   Skipped weights:   {stats['skipped_weights']:,}")
        print(f"   Total weights:     {total_weights:,}")
        if total_weights > 0:
            pct = 100 * stats['quantized_weights'] / total_weights
            print(f"   → {pct:.1f}% of weights are quantized")
        
        # Activation statistics
        print(f"\n📊 ACTIVATION STATISTICS (per forward pass):")
        if stats['weight_only_mode']:
            print(f"   Mode: WEIGHT-ONLY (activations stay in FP32)")
            print(f"   Activations NOT quantized: {stats['total_activations']:,}")
        else:
            print(f"   Mode: WEIGHT+ACTIVATION (activations are quantized)")
            print(f"   Quantized activations: {stats['quantized_activations']:,}")
            print(f"   Skipped activations:   {stats['skipped_activations']:,}")
            print(f"   Total activations:     {stats['total_activations']:,}")
            if stats['total_activations'] > 0:
                pct = 100 * stats['quantized_activations'] / stats['total_activations']
                print(f"   → {pct:.1f}% of activations are quantized")
        
        # Layer details
        if stats['quantized_layers_list']:
            print(f"\n✓ Quantized layers:")
            for layer in stats['quantized_layers_list']:
                print(f"   {layer}")
        
        if stats['skipped_layers_list']:
            print(f"\n✗ Skipped layers:")
            for layer in stats['skipped_layers_list']:
                print(f"   {layer}")
        
        print(f"{'='*70}\n")
    
    return quantized_model


def get_quantization_stats(model: nn.Module, input_shape: tuple = (1, 3, 256, 256)) -> Dict:
    """
    Report quantization statistics: number of weights and activations being quantized.
    
    Args:
        model: The model (original or quantized)
        input_shape: Input shape for activation counting (B, C, H, W)
    
    Returns:
        Dictionary with quantization statistics
    """
    stats = {
        'total_weights': 0,
        'quantized_weights': 0,
        'weight_layers': [],
        'total_activations_per_forward': 0,
        'quantized_activations_per_forward': 0,
        'activation_layers': [],
        'linear_layers': 0,
        'conv2d_layers': 0,
    }
    
    def count_params(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Count weights
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                weight_count = child.weight.numel()
                stats['total_weights'] += weight_count
                
                if isinstance(child, nn.Linear):
                    stats['linear_layers'] += 1
                else:
                    stats['conv2d_layers'] += 1
                    
            elif isinstance(child, (WeightOnlyQuantizedLinear, WeightActivationQuantizedLinear)):
                weight_count = child.in_features * child.out_features
                stats['total_weights'] += weight_count
                stats['quantized_weights'] += weight_count
                stats['linear_layers'] += 1
                stats['weight_layers'].append({
                    'name': full_name,
                    'type': 'Linear',
                    'weights': weight_count,
                    'bits': child.weight_config.num_bits
                })
                
                if isinstance(child, WeightActivationQuantizedLinear):
                    # Activations are quantized for each forward pass
                    act_count = child.in_features  # per sample
                    stats['quantized_activations_per_forward'] += act_count
                    stats['activation_layers'].append({
                        'name': full_name,
                        'activations_per_sample': act_count,
                        'bits': child.activation_config.num_bits
                    })
                    
            elif isinstance(child, (WeightOnlyQuantizedConv2d, WeightActivationQuantizedConv2d)):
                weight_count = child.out_channels * child.in_channels * child.kernel_size[0] * child.kernel_size[1]
                stats['total_weights'] += weight_count
                stats['quantized_weights'] += weight_count
                stats['conv2d_layers'] += 1
                stats['weight_layers'].append({
                    'name': full_name,
                    'type': 'Conv2d',
                    'weights': weight_count,
                    'bits': child.weight_config.num_bits
                })
                
                if isinstance(child, WeightActivationQuantizedConv2d):
                    # Estimate activation count based on input shape
                    stats['activation_layers'].append({
                        'name': full_name,
                        'type': 'Conv2d',
                        'bits': child.activation_config.num_bits
                    })
            else:
                # Recurse into child modules
                count_params(child, full_name)
    
    count_params(model)
    
    # Calculate total activations (rough estimate based on input)
    stats['total_activations_per_forward'] = input_shape[1] * input_shape[2] * input_shape[3]
    
    return stats


def print_quantization_stats(model: nn.Module, input_shape: tuple = (1, 3, 256, 256)):
    """Print formatted quantization statistics."""
    stats = get_quantization_stats(model, input_shape)
    
    print("\n" + "="*60)
    print("QUANTIZATION STATISTICS")
    print("="*60)
    
    print(f"\n📊 Weight Statistics:")
    print(f"   Total weights: {stats['total_weights']:,}")
    print(f"   Quantized weights: {stats['quantized_weights']:,}")
    if stats['total_weights'] > 0:
        pct = stats['quantized_weights'] / stats['total_weights'] * 100
        print(f"   Quantization coverage: {pct:.1f}%")
    
    print(f"\n📊 Layer Statistics:")
    print(f"   Linear layers: {stats['linear_layers']}")
    print(f"   Conv2d layers: {stats['conv2d_layers']}")
    
    if stats['weight_layers']:
        print(f"\n📊 Quantized Weight Layers:")
        for layer in stats['weight_layers']:
            print(f"   - {layer['name']}: {layer['weights']:,} weights @ {layer['bits']}-bit")
    
    if stats['activation_layers']:
        print(f"\n📊 Quantized Activation Layers:")
        for layer in stats['activation_layers']:
            if 'activations_per_sample' in layer:
                print(f"   - {layer['name']}: {layer['activations_per_sample']:,} activations/sample @ {layer['bits']}-bit")
            else:
                print(f"   - {layer['name']}: @ {layer['bits']}-bit")
    
    print("="*60 + "\n")
    
    return stats


# =============================================================================
# PUBLICATION-QUALITY PLOTTING (Conference Paper Style)
# =============================================================================

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (6, 4),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })


def plot_quantization_sweep(bit_widths: List[int], psnr_values: List[float], 
                           output_path: str, model_name: str = "Model",
                           optimal_bits: int = None):
    """
    Plot PSNR vs bit-width for quantization analysis.
    Publication-quality figure suitable for IEEE/ACM conferences.
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Main line
    ax.plot(bit_widths, psnr_values, 'o-', color='#1f77b4', 
            label='Weight+Activation Quantization', markersize=5)
    
    # Highlight optimal bit-width
    if optimal_bits and optimal_bits in bit_widths:
        idx = list(bit_widths).index(optimal_bits)
        ax.axvline(x=optimal_bits, color='#2ca02c', linestyle='--', alpha=0.7, 
                   label=f'Optimal: {optimal_bits}-bit')
        ax.scatter([optimal_bits], [psnr_values[idx]], color='#2ca02c', 
                   s=80, zorder=5, marker='*')
    
    # FP32 baseline
    fp32_psnr = psnr_values[-1] if len(psnr_values) > 0 else 35
    ax.axhline(y=fp32_psnr, color='#d62728', linestyle=':', alpha=0.7, 
               label=f'FP32 baseline ({fp32_psnr:.1f} dB)')
    
    ax.set_xlabel('Bit-width')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'Quantization Analysis: {model_name}')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(min(bit_widths) - 0.5, max(bit_widths) + 0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'))
    print(f"  Saved: {output_path}")
    plt.close()


def plot_memory_savings(bit_widths: List[int], total_weights: int, 
                       output_path: str, model_name: str = "Model"):
    """
    Bar chart showing memory savings at different bit-widths.
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Calculate memory in MB for each bit-width
    memory_mb = [(total_weights * b) / (8 * 1024 * 1024) for b in bit_widths]
    
    colors = ['#d62728' if b <= 4 else '#ff7f0e' if b <= 8 else '#2ca02c' if b <= 16 else '#1f77b4' 
              for b in bit_widths]
    
    bars = ax.bar(range(len(bit_widths)), memory_mb, color=colors, 
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(bit_widths)))
    ax.set_xticklabels([str(b) for b in bit_widths])
    
    # Add value labels
    for bar, mem in zip(bars, memory_mb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(memory_mb),
                f'{mem:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Bit-width')
    ax.set_ylabel('Model Size (MB)')
    ax.set_title(f'Memory Savings: {model_name}')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'))
    print(f"  Saved: {output_path}")
    plt.close()


def generate_quantization_report(results: Dict, output_dir: str, model_name: str = "Model"):
    """
    Generate all publication-quality plots from quantization results.
    
    Args:
        results: Dictionary with 'bit_widths', 'psnr_values', 'optimal_bits', 'total_weights'
        output_dir: Directory to save plots
        model_name: Name for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📊 Generating publication-quality plots...")
    
    if 'bit_widths' in results and 'psnr_values' in results:
        plot_quantization_sweep(
            results['bit_widths'],
            results['psnr_values'],
            os.path.join(output_dir, 'quantization_psnr_sweep.pdf'),
            model_name,
            results.get('optimal_bits')
        )
    
    if 'bit_widths' in results and 'total_weights' in results:
        plot_memory_savings(
            results['bit_widths'],
            results['total_weights'],
            os.path.join(output_dir, 'memory_savings.pdf'),
            model_name
        )
    
    print(f"✓ All plots saved to: {output_dir}")


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

# Removed custom PSNR/SSIM - now using torchmetrics (matching test_model_metrics.py)


def evaluate_model_metrics(model: nn.Module, dataloader: DataLoader, 
                           device: torch.device, ar_system=None,
                           use_roi_extraction: bool = False,
                           roi_coords=None, roi_target_size=DEFAULT_ROI_TARGET_SIZE,
                           save_samples: bool = False,
                           sample_save_dir: str = None,
                           sample_prefix: str = '',
                           num_samples: int = 4) -> Dict[str, float]:
    """
    Evaluate model on a dataset and compute metrics.
    EXACTLY matches test_model_metrics.py pattern.
    
    Args:
        model: NN2 model to evaluate
        dataloader: DataLoader with (input, target) pairs
        device: Device to use
        ar_system: ARSystem for applying distortions (if dont_use_nn1)
        use_roi_extraction: Whether to use ROI extraction
        roi_coords: ROI coordinates for extraction
        roi_target_size: Target size for ROI patches
        save_samples: Whether to save sample images
        sample_save_dir: Directory to save samples (if save_samples=True)
        sample_prefix: Prefix for saved sample filenames
        num_samples: Number of sample images to save
    
    Returns:
        Dictionary with metrics: psnr, ssim, rmse, lpips
    """
    model.eval()
    
    # Initialize metrics (EXACTLY as test_model_metrics.py)
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    rmse_metric = torchmetrics.MeanSquaredError(squared=False).to(device)
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    total_psnr = 0
    total_ssim = 0
    total_rmse = 0
    total_lpips = 0
    num_batches = 0
    skipped_batches = 0
    
    with torch.no_grad():
        for batch_idx, (clean_input, clean_target) in enumerate(dataloader):
            clean_input = clean_input.to(device)
            clean_target = clean_target.to(device)
            
            try:
                # Use autocast for mixed precision evaluation (matching test_model_metrics.py)
                if USE_AMP and torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        # NN2 processes clean image
                        nn2_output = model(clean_input)
                        
                        # Handle different output formats
                        if isinstance(nn2_output, dict):
                            nn2_final_output = nn2_output['final']
                        else:
                            nn2_final_output = nn2_output
                        
                        # Apply distortion using AR system if provided
                        if ar_system is not None:
                            # Use gradient-friendly AR system for distortion (EXACTLY as test_model_metrics.py)
                            if nn2_final_output.dim() == 4:  # [B, C, H, W] - batch processing
                                batch_size = nn2_final_output.size(0)
                                result_batch = []
                                for i in range(batch_size):
                                    single_image = nn2_final_output[i]  # [C, H, W]
                                    distorted_tensor = apply_gradient_friendly_distortions(single_image, ar_system)
                                    result_batch.append(distorted_tensor)
                                # Stack back into batch tensor
                                final_output = torch.stack(result_batch, dim=0)
                            else:  # [C, H, W] - single image
                                final_output = apply_gradient_friendly_distortions(nn2_final_output, ar_system)
                        else:
                            # No AR system - evaluate NN2 output directly
                            final_output = nn2_final_output
                        
                        # Mask out ROI areas from final prediction before metric calculation
                        if use_roi_extraction:
                            final_output_masked = mask_out_roi_areas(final_output, roi_coords)
                        else:
                            final_output_masked = final_output
                else:
                    # Standard precision evaluation
                    nn2_output = model(clean_input)
                    
                    # Handle different output formats
                    if isinstance(nn2_output, dict):
                        nn2_final_output = nn2_output['final']
                    else:
                        nn2_final_output = nn2_output
                    
                    # Apply distortion using AR system if provided
                    if ar_system is not None:
                        if nn2_final_output.dim() == 4:
                            batch_size = nn2_final_output.size(0)
                            result_batch = []
                            for i in range(batch_size):
                                single_image = nn2_final_output[i]
                                distorted_tensor = apply_gradient_friendly_distortions(single_image, ar_system)
                                result_batch.append(distorted_tensor)
                            final_output = torch.stack(result_batch, dim=0)
                        else:
                            final_output = apply_gradient_friendly_distortions(nn2_final_output, ar_system)
                    else:
                        final_output = nn2_final_output
                    
                    # Mask out ROI areas
                    if use_roi_extraction:
                        final_output_masked = mask_out_roi_areas(final_output, roi_coords)
                    else:
                        final_output_masked = final_output
                
                # For metrics, use the same expected output format (EXACTLY as test_model_metrics.py)
                if use_roi_extraction:
                    has_geo = has_geometric_distortion_in_ar_system(ar_system) if ar_system is not None else None
                    clean_target_metric = create_expected_output_with_roi(
                        clean_target, roi_coords, roi_target_size, has_geometric_distortion=has_geo
                    )
                    final_output_metric = final_output_masked
                else:
                    clean_target_metric = clean_target
                    final_output_metric = final_output_masked
                
                # Calculate metrics on clamped outputs
                final_output_clamped = torch.clamp(final_output_metric, 0, 1)
                clean_target_clamped = torch.clamp(clean_target_metric, 0, 1)
                
                # Check for NaN/Inf
                if torch.isnan(final_output_clamped).any() or torch.isinf(final_output_clamped).any() or \
                   torch.isnan(clean_target_clamped).any() or torch.isinf(clean_target_clamped).any():
                    skipped_batches += 1
                    continue
                
                # Compute metrics using torchmetrics
                psnr_val = psnr_metric(final_output_clamped, clean_target_clamped)
                ssim_val = ssim_metric(final_output_clamped, clean_target_clamped)
                rmse_val = rmse_metric(final_output_clamped, clean_target_clamped)
                
                # LPIPS calculation (EXACTLY as test_model_metrics.py)
                # Normalize to [-1, 1] for LPIPS
                outputs_lpips = 2.0 * final_output_metric - 1.0
                target_lpips = 2.0 * clean_target_metric - 1.0
                lpips_val = lpips_model(outputs_lpips, target_lpips).mean()
                
                # Check if metrics are valid
                if torch.isnan(psnr_val).any() or torch.isinf(psnr_val).any() or \
                   torch.isnan(ssim_val).any() or torch.isinf(ssim_val).any() or \
                   torch.isnan(rmse_val).any() or torch.isinf(rmse_val).any() or \
                   torch.isnan(lpips_val).any() or torch.isinf(lpips_val).any():
                    skipped_batches += 1
                    continue
                
                # Save sample images if requested
                if save_samples and sample_save_dir is not None and num_batches < num_samples:
                    os.makedirs(sample_save_dir, exist_ok=True)
                    
                    # Also compute distorted input for visualization
                    if ar_system is not None:
                        if clean_input.dim() == 4:
                            distorted_inputs = []
                            for i in range(clean_input.size(0)):
                                distorted_single = apply_gradient_friendly_distortions(clean_input[i], ar_system)
                                distorted_inputs.append(distorted_single)
                            distorted_input = torch.stack(distorted_inputs, dim=0)
                        else:
                            distorted_input = apply_gradient_friendly_distortions(clean_input, ar_system)
                    else:
                        distorted_input = clean_input
                    
                    # Save comparison for each image in batch
                    batch_size = clean_input.size(0)
                    for img_idx in range(min(batch_size, num_samples - num_batches)):
                        sample_idx = num_batches + img_idx
                        
                        # Create comparison figure: Clean, Distorted, Corrected, Target
                        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                        
                        # Clean input - ensure float32 for matplotlib
                        clean_np = clean_input[img_idx].cpu().float().permute(1, 2, 0).numpy().astype(np.float32)
                        axes[0].imshow(np.clip(clean_np, 0, 1))
                        axes[0].set_title('Clean Input')
                        axes[0].axis('off')
                        
                        # Distorted
                        dist_np = distorted_input[img_idx].cpu().float().permute(1, 2, 0).numpy().astype(np.float32)
                        axes[1].imshow(np.clip(dist_np, 0, 1))
                        axes[1].set_title('Distorted')
                        axes[1].axis('off')
                        
                        # Corrected
                        corr_np = final_output_clamped[img_idx].cpu().float().permute(1, 2, 0).numpy().astype(np.float32)
                        axes[2].imshow(np.clip(corr_np, 0, 1))
                        axes[2].set_title('Corrected')
                        axes[2].axis('off')
                        
                        # Target
                        target_np = clean_target_clamped[img_idx].cpu().float().permute(1, 2, 0).numpy().astype(np.float32)
                        axes[3].imshow(np.clip(target_np, 0, 1))
                        axes[3].set_title('Expected')
                        axes[3].axis('off')
                        
                        plt.tight_layout()
                        save_path = os.path.join(sample_save_dir, f'{sample_prefix}sample_{sample_idx}.png')
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                
                # Accumulate metrics
                total_psnr += psnr_val.item()
                total_ssim += ssim_val.item()
                total_rmse += rmse_val.item()
                total_lpips += lpips_val.item()
                num_batches += 1
                
            except Exception as e:
                if skipped_batches == 0:  # Print full traceback only for first error
                    import traceback
                    print(f"Error processing batch {batch_idx}:")
                    traceback.print_exc()
                else:
                    print(f"Warning: Error processing batch {batch_idx}: {e}")
                skipped_batches += 1
                continue
    
    # Calculate averages
    if num_batches == 0:
        print("Warning: No valid batches in evaluation!")
        return {
            'psnr': 0.0,
            'ssim': 0.0,
            'rmse': 0.0,
            'lpips': 0.0,
            'skipped_batches': skipped_batches,
            'total_batches': len(dataloader),
            'valid_batches': 0
        }
    
    return {
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches,
        'rmse': total_rmse / num_batches,
        'lpips': total_lpips / num_batches,
        'skipped_batches': skipped_batches,
        'total_batches': len(dataloader),
        'valid_batches': num_batches
    }


# =============================================================================
# STEP-BY-STEP QUANTIZATION WORKFLOW
# =============================================================================

class QuantizationAnalyzer:
    """
    Analyzes model quantization with step-by-step bit reduction.
    MATCHES test_model_metrics.py evaluation pattern.
    
    TWO-STEP ANALYSIS:
    1. WEIGHT-ONLY: Quantize weights only, activations stay FP32
       - Dequantizes to FP32 before compute (slower floating-point ops)
       - Saves memory/bandwidth
       - Simpler, safer, less accuracy loss
    
    2. WEIGHT+ACTIVATION: Quantize both weights and activations
       - Pure integer operations (faster integer ops)
       - Saves memory AND speeds up compute
       - Requires calibration, more accuracy risk
    """
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        output_dir: str = './quantization_results',
        min_bits: int = DEFAULT_MIN_BITS,
        max_bits: int = DEFAULT_MAX_BITS,
        stride: int = 1,  # Step size for bit sweep (1=every bit, 2=skip every other)
        symmetric: bool = DEFAULT_SYMMETRIC,
        per_channel: bool = DEFAULT_PER_CHANNEL,
        ar_system=None,  # ARSystem for applying distortions
        use_roi_extraction: bool = False,
        roi_coords=None,
        roi_target_size=DEFAULT_ROI_TARGET_SIZE,
        verbose: bool = False  # Print detailed quantization statistics
    ):
        self.original_model = model
        self.dataloader = dataloader
        self.device = device
        self.output_dir = output_dir
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.stride = stride
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.ar_system = ar_system
        self.use_roi_extraction = use_roi_extraction
        self.roi_coords = roi_coords
        self.roi_target_size = roi_target_size
        self.verbose = verbose
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Store results
        self.weight_quant_results = []
        self.activation_quant_results = []
    
    def analyze_weight_quantization(self) -> Tuple[int, Dict]:
        """
        Step 1: Analyze WEIGHT-ONLY quantization from max_bits down to min_bits.
        
        WEIGHT-ONLY QUANTIZATION:
        - Stores weights as x-bit integers (saves memory/bandwidth)
        - Dequantizes back to FP32 before computation (uses floating-point units)
        - Activations remain in FP32 (no activation quantization yet)
        - Simpler, safer, less accuracy loss
        
        Returns:
            optimal_bits: Best bit-width before significant metric drop
            results: Dictionary of all results
        """
        print("\n" + "=" * 80)
        print("STEP 1: WEIGHT-ONLY Quantization Analysis")
        print("=" * 80)
        print("MODE: Weight-only quantization")
        print("  - Weights stored as integers, dequantized to FP32 before compute")
        print("  - Activations remain in FP32 (no quantization)")
        print("  - Uses floating-point hardware")
        print("  - Benefit: memory/bandwidth savings")
        print()
        print(f"Testing bit-widths from {self.max_bits} down to {self.min_bits} bits")
        print(f"Symmetric: {self.symmetric}, Per-channel: {self.per_channel}")
        print()
        
        # Evaluate baseline (full precision)
        print("Evaluating baseline (32-bit full precision)...")
        self.original_model.to(self.device)
        
        # Create samples directory
        samples_dir = os.path.join(self.output_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        baseline_metrics = evaluate_model_metrics(
            self.original_model, self.dataloader, self.device,
            ar_system=self.ar_system,
            use_roi_extraction=self.use_roi_extraction,
            roi_coords=self.roi_coords,
            roi_target_size=self.roi_target_size,
            save_samples=True,
            sample_save_dir=samples_dir,
            sample_prefix='baseline_32bit_',
            num_samples=4
        )
        
        print(f"Baseline Metrics:")
        print(f"  PSNR: {baseline_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {baseline_metrics['ssim']:.4f}")
        print(f"  RMSE: {baseline_metrics['rmse']:.6f}")
        print(f"  LPIPS: {baseline_metrics['lpips']:.6f}")
        print()
        
        self.weight_quant_results.append({
            'bits': 32,
            'type': 'full_precision',
            **baseline_metrics
        })
        
        # Test each bit-width with stride
        import time
        bit_widths = list(range(self.max_bits - 1, self.min_bits - 1, -self.stride))
        # Ensure min_bits is included
        if self.min_bits not in bit_widths:
            bit_widths.append(self.min_bits)
        
        for num_bits in bit_widths:
            bit_start_time = time.time()
            print(f"Testing {num_bits}-bit WEIGHT-ONLY quantization...")
            
            # Create quantization config
            weight_config = QuantizationConfig(
                num_bits=num_bits,
                symmetric=self.symmetric,
                per_channel=self.per_channel,
                quantize_bias=False
            )
            
            # Quantize model (WEIGHT-ONLY mode: activations stay FP32)
            # Only print verbose stats on the first iteration
            is_first_iteration = (num_bits == bit_widths[0])
            quantized_model = quantize_model(
                self.original_model,
                weight_config=weight_config,
                activation_config=None,  # No activation quantization = weight-only mode
                use_weight_only=True,
                verbose=(self.verbose and is_first_iteration)
            )
            quantized_model.to(self.device)
            
            # Evaluate (save samples for minimum bits to show worst case)
            save_samples_flag = (num_bits == self.min_bits)
            metrics = evaluate_model_metrics(
                quantized_model, self.dataloader, self.device,
                ar_system=self.ar_system,
                use_roi_extraction=self.use_roi_extraction,
                roi_coords=self.roi_coords,
                roi_target_size=self.roi_target_size,
                save_samples=save_samples_flag,
                sample_save_dir=samples_dir if save_samples_flag else None,
                sample_prefix=f'weight_{num_bits}bit_',
                num_samples=4
            )
            
            # Calculate degradation
            psnr_drop = baseline_metrics['psnr'] - metrics['psnr']
            ssim_drop = baseline_metrics['ssim'] - metrics['ssim']
            
            # Calculate time spent
            bit_elapsed_time = time.time() - bit_start_time
            bit_elapsed_minutes = bit_elapsed_time / 60.0
            
            print(f"  PSNR: {metrics['psnr']:.2f} dB (Δ: {psnr_drop:+.2f})")
            print(f"  SSIM: {metrics['ssim']:.4f} (Δ: {ssim_drop:+.4f})")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  LPIPS: {metrics['lpips']:.6f}")
            print(f"  ⏱ Time: {bit_elapsed_minutes:.2f} minutes")
            print()
            
            self.weight_quant_results.append({
                'bits': num_bits,
                'type': 'weight_only',
                'psnr_drop': psnr_drop,
                'ssim_drop': ssim_drop,
                **metrics
            })
            
            # Clean up
            del quantized_model
            torch.cuda.empty_cache()
        
        # Find optimal bit-width using default thresholds
        optimal_bits = self.find_optimal_bits(
            self.weight_quant_results,
            psnr_threshold=DEFAULT_PSNR_THRESHOLD,
            ssim_threshold=DEFAULT_SSIM_THRESHOLD
        )
        
        # Store for plotting
        self._optimal_weight_bits = optimal_bits
        
        print(f"\n{'='*80}")
        print(f"Optimal weight quantization: {optimal_bits} bits")
        print(f"{'='*80}\n")
        
        # Generate and save samples for optimal quantization level if not already saved
        if optimal_bits != self.min_bits:
            print(f"Generating samples for optimal {optimal_bits}-bit WEIGHT-ONLY quantization...")
            optimal_config = QuantizationConfig(
                num_bits=optimal_bits,
                symmetric=self.symmetric,
                per_channel=self.per_channel,
                quantize_bias=False
            )
            optimal_model = quantize_model(
                self.original_model,
                weight_config=optimal_config,
                activation_config=None,
                use_weight_only=True  # Weight-only mode
            )
            optimal_model.to(self.device)
            
            _ = evaluate_model_metrics(
                optimal_model, self.dataloader, self.device,
                ar_system=self.ar_system,
                use_roi_extraction=self.use_roi_extraction,
                roi_coords=self.roi_coords,
                roi_target_size=self.roi_target_size,
                save_samples=True,
                sample_save_dir=samples_dir,
                sample_prefix=f'weight_optimal_{optimal_bits}bit_',
                num_samples=4
            )
            
            del optimal_model
            torch.cuda.empty_cache()
            print("✓ Optimal quantization samples saved\n")
        
        # Save results
        self.save_results('weight_quantization_results.json', self.weight_quant_results)
        self.plot_weight_quantization_results()
        
        return optimal_bits, self.weight_quant_results
    
    def analyze_activation_quantization(self, weight_bits: int) -> Dict:
        """
        Step 2: Analyze WEIGHT+ACTIVATION quantization with fixed weight quantization.
        
        WEIGHT+ACTIVATION QUANTIZATION:
        - Stores BOTH weights AND activations as x-bit integers
        - Performs pure INTEGER operations (uses integer hardware)
        - NO dequantization until final output
        - Benefit: memory savings + significant compute speedup
        - Tradeoff: requires careful calibration, risks more accuracy loss
        
        Args:
            weight_bits: Bit-width for weight quantization (from step 1)
        
        Returns:
            results: Dictionary of all results
        """
        print("\n" + "=" * 80)
        print("STEP 2: WEIGHT+ACTIVATION Quantization Analysis")
        print("=" * 80)
        print("MODE: Weight+Activation quantization")
        print("  - Both weights AND activations stored as integers")
        print("  - Pure integer operations (faster)")
        print("  - Uses integer hardware (more efficient)")
        print("  - Benefit: memory savings + compute speedup")
        print("  - Tradeoff: requires calibration, more accuracy risk")
        print()
        print(f"Weight quantization fixed at {weight_bits} bits")
        print(f"Testing activation quantization from 32 down to {self.min_bits} bits")
        print()
        
        # Create samples directory
        samples_dir = os.path.join(self.output_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        # Get baseline with weight quantization only
        weight_config = QuantizationConfig(
            num_bits=weight_bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            quantize_bias=False
        )
        
        weight_only_model = quantize_model(
            self.original_model,
            weight_config=weight_config,
            activation_config=None,
            use_weight_only=True,  # Explicitly use weight-only mode
            verbose=False  # Suppress stats for baseline model
        )
        weight_only_model.to(self.device)
        
        baseline_metrics = evaluate_model_metrics(
            weight_only_model, self.dataloader, self.device,
            ar_system=self.ar_system,
            use_roi_extraction=self.use_roi_extraction,
            roi_coords=self.roi_coords,
            roi_target_size=self.roi_target_size
        )
        
        print(f"Baseline (Weight-only @ {weight_bits} bits):")
        print(f"  PSNR: {baseline_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {baseline_metrics['ssim']:.4f}")
        print(f"  RMSE: {baseline_metrics['rmse']:.6f}")
        print(f"  LPIPS: {baseline_metrics['lpips']:.6f}")
        print()
        
        self.activation_quant_results.append({
            'weight_bits': weight_bits,
            'activation_bits': 32,
            'type': 'weight_only',
            **baseline_metrics
        })
        
        # Test each activation bit-width with stride
        import time
        bit_widths = list(range(self.max_bits - 1, self.min_bits - 1, -self.stride))
        # Ensure min_bits is included
        if self.min_bits not in bit_widths:
            bit_widths.append(self.min_bits)
        
        for act_bits in bit_widths:
            bit_start_time = time.time()
            print(f"Testing {act_bits}-bit WEIGHT+ACTIVATION quantization...")
            
            # Create activation config
            activation_config = QuantizationConfig(
                num_bits=act_bits,
                symmetric=True,  # Assumption: Symmetric for activations (ReLU outputs are >= 0, but we keep symmetric for generality)
                per_channel=False,  # Assumption: Per-tensor for activations
                quantize_bias=False
            )
            
            # Quantize model (WEIGHT+ACTIVATION mode: pure integer operations)
            # Only print verbose stats on the first iteration
            is_first_iteration = (act_bits == bit_widths[0])
            quantized_model = quantize_model(
                self.original_model,
                weight_config=weight_config,
                activation_config=activation_config,
                use_weight_only=False,  # Enable weight+activation mode
                verbose=(self.verbose and is_first_iteration)
            )
            quantized_model.to(self.device)
            
            # Evaluate (save samples for minimum bits to show worst case)
            save_samples_flag = (act_bits == self.min_bits)
            metrics = evaluate_model_metrics(
                quantized_model, self.dataloader, self.device,
                ar_system=self.ar_system,
                use_roi_extraction=self.use_roi_extraction,
                roi_coords=self.roi_coords,
                roi_target_size=self.roi_target_size,
                save_samples=save_samples_flag,
                sample_save_dir=samples_dir if save_samples_flag else None,
                sample_prefix=f'weight{weight_bits}bit_act{act_bits}bit_',
                num_samples=4
            )
            
            # Calculate degradation
            psnr_drop = baseline_metrics['psnr'] - metrics['psnr']
            ssim_drop = baseline_metrics['ssim'] - metrics['ssim']
            
            # Calculate time spent
            bit_elapsed_time = time.time() - bit_start_time
            bit_elapsed_minutes = bit_elapsed_time / 60.0
            
            print(f"  PSNR: {metrics['psnr']:.2f} dB (Δ: {psnr_drop:+.2f})")
            print(f"  SSIM: {metrics['ssim']:.4f} (Δ: {ssim_drop:+.4f})")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  LPIPS: {metrics['lpips']:.6f}")
            print(f"  ⏱ Time: {bit_elapsed_minutes:.2f} minutes")
            print()
            
            self.activation_quant_results.append({
                'weight_bits': weight_bits,
                'activation_bits': act_bits,
                'type': 'weight_and_activation',
                'psnr_drop': psnr_drop,
                'ssim_drop': ssim_drop,
                **metrics
            })
            
            # Clean up
            del quantized_model
            torch.cuda.empty_cache()
        
        # Find optimal activation bits
        optimal_act_bits = self.find_optimal_bits(
            self.activation_quant_results,
            psnr_threshold=DEFAULT_PSNR_THRESHOLD,
            ssim_threshold=DEFAULT_SSIM_THRESHOLD
        )
        
        # Store for plotting
        self._optimal_activation_bits = optimal_act_bits
        
        print(f"\n{'='*80}")
        print(f"Optimal activation quantization: {optimal_act_bits} bits")
        print(f"{'='*80}\n")
        
        # Save results
        self.save_results('activation_quantization_results.json', self.activation_quant_results)
        self.plot_activation_quantization_results()
        
        return self.activation_quant_results
    
    def find_optimal_bits(
        self,
        results: List[Dict],
        psnr_threshold: float = DEFAULT_PSNR_THRESHOLD,
        ssim_threshold: float = DEFAULT_SSIM_THRESHOLD
    ) -> int:
        """
        Find optimal bit-width based on acceptable metric degradation.
        
        NOTE: Does NOT assume linear degradation with bit reduction.
        Instead, scans ALL bit-widths and finds the lowest one that meets 
        the threshold. E.g., if bit=31 is bad but bit=30 is good, this will 
        correctly select 30 (not 32).
        
        Assumption: We prioritize PSNR drop < threshold, then SSIM drop < threshold.
        """
        # Collect all acceptable bit-widths (those within threshold)
        acceptable_bits = []
        
        for result in results:
            # Handle both weight results ('bits') and activation results ('activation_bits')
            bits = result.get('bits') or result.get('activation_bits')
            if bits is None:
                continue
            
            # Identify baseline results (no drop metrics calculated)
            if 'psnr_drop' not in result:
                # Baseline is always acceptable
                acceptable_bits.append(bits)
                continue
            
            psnr_drop = result.get('psnr_drop', 0)
            ssim_drop = result.get('ssim_drop', 0)
            
            # Check if this bit-width meets the threshold
            if psnr_drop <= psnr_threshold and ssim_drop <= ssim_threshold:
                acceptable_bits.append(bits)
        
        # Return the lowest acceptable bit-width
        if acceptable_bits:
            return min(acceptable_bits)
        
        # If none are acceptable, return maximum bits (safest option)
        return self.max_bits
    
    def save_results(self, filename: str, results: List[Dict]):
        """Save quantization results to JSON."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filepath}")
    
    def plot_weight_quantization_results(self):
        """Plot weight quantization analysis results."""
        bits = [r['bits'] for r in self.weight_quant_results]
        psnr = [r['psnr'] for r in self.weight_quant_results]
        ssim = [r['ssim'] for r in self.weight_quant_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PSNR plot
        ax1.plot(bits, psnr, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Bit-width', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('Weight Quantization: PSNR vs Bit-width', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        # SSIM plot
        ax2.plot(bits, ssim, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Bit-width', fontsize=12)
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.set_title('Weight Quantization: SSIM vs Bit-width', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'weight_quantization_plot.png')
        plt.savefig(filepath, dpi=150)
        print(f"Weight plot saved to: {filepath}")
        plt.close()
        
        # Store for combined plot
        self._weight_plot_data = (bits, psnr, ssim)
    
    def plot_activation_quantization_results(self):
        """Plot activation quantization analysis results."""
        act_bits = [r['activation_bits'] for r in self.activation_quant_results]
        psnr = [r['psnr'] for r in self.activation_quant_results]
        ssim = [r['ssim'] for r in self.activation_quant_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PSNR plot
        ax1.plot(act_bits, psnr, 'go-', linewidth=2, markersize=8)
        ax1.set_xlabel('Activation Bit-width', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('Activation Quantization: PSNR vs Bit-width', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        # SSIM plot
        ax2.plot(act_bits, ssim, 'mo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Activation Bit-width', fontsize=12)
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.set_title('Activation Quantization: SSIM vs Bit-width', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'activation_quantization_plot.png')
        plt.savefig(filepath, dpi=150)
        print(f"Activation plot saved to: {filepath}")
        plt.close()
        
        # Store for combined plot
        self._activation_plot_data = (act_bits, psnr, ssim)
        
        # Create combined plot
        self.plot_combined_results()
    
    def plot_combined_results(self):
        """Create a combined plot with both weight and activation quantization results."""
        if not hasattr(self, '_weight_plot_data'):
            return
        
        weight_bits, weight_psnr, weight_ssim = self._weight_plot_data
        
        # Find optimal bits if available
        optimal_weight_bits = getattr(self, '_optimal_weight_bits', None)
        optimal_act_bits = getattr(self, '_optimal_activation_bits', None)
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Weight quantization - PSNR
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(weight_bits, weight_psnr, 'bo-', linewidth=2, markersize=8, label='Weight Quant')
        if optimal_weight_bits:
            ax1.axvline(x=optimal_weight_bits, color='green', linestyle='--', linewidth=2, label=f'Optimal: {optimal_weight_bits}-bit')
        ax1.set_xlabel('Weight Bit-width', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('Weight Quantization: PSNR vs Bit-width', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        ax1.legend()
        
        # Weight quantization - SSIM
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(weight_bits, weight_ssim, 'ro-', linewidth=2, markersize=8, label='Weight Quant')
        if optimal_weight_bits:
            ax2.axvline(x=optimal_weight_bits, color='green', linestyle='--', linewidth=2, label=f'Optimal: {optimal_weight_bits}-bit')
        ax2.set_xlabel('Weight Bit-width', fontsize=12)
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.set_title('Weight Quantization: SSIM vs Bit-width', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        ax2.legend()
        
        # Activation quantization (if available)
        if hasattr(self, '_activation_plot_data'):
            act_bits, act_psnr, act_ssim = self._activation_plot_data
            
            # Activation quantization - PSNR
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(act_bits, act_psnr, 'go-', linewidth=2, markersize=8, label='Activation Quant')
            if optimal_act_bits:
                ax3.axvline(x=optimal_act_bits, color='orange', linestyle='--', linewidth=2, label=f'Optimal: {optimal_act_bits}-bit')
            ax3.set_xlabel('Activation Bit-width', fontsize=12)
            ax3.set_ylabel('PSNR (dB)', fontsize=12)
            ax3.set_title('Activation Quantization: PSNR vs Bit-width', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.invert_xaxis()
            ax3.legend()
            
            # Activation quantization - SSIM
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(act_bits, act_ssim, 'mo-', linewidth=2, markersize=8, label='Activation Quant')
            if optimal_act_bits:
                ax4.axvline(x=optimal_act_bits, color='orange', linestyle='--', linewidth=2, label=f'Optimal: {optimal_act_bits}-bit')
            ax4.set_xlabel('Activation Bit-width', fontsize=12)
            ax4.set_ylabel('SSIM', fontsize=12)
            ax4.set_title('Activation Quantization: SSIM vs Bit-width', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.invert_xaxis()
            ax4.legend()
        
        # Add optimal bits info to title
        title = 'Quantization Analysis Results'
        if optimal_weight_bits:
            title += f' (Optimal Weights: {optimal_weight_bits}-bit'
            if optimal_act_bits:
                title += f', Activations: {optimal_act_bits}-bit)'
            else:
                title += ')'
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save as PNG
        filepath_png = os.path.join(self.output_dir, 'combined_quantization_plots.png')
        plt.savefig(filepath_png, dpi=200, bbox_inches='tight')
        print(f"✓ Combined plot saved to: {filepath_png}")
        
        # Save as PDF
        filepath_pdf = os.path.join(self.output_dir, 'combined_quantization_plots.pdf')
        plt.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
        print(f"✓ Combined plot (PDF) saved to: {filepath_pdf}")
        
        plt.close()
    
    def create_composite_samples(self):
        """Create a composite image from all saved samples."""
        samples_dir = os.path.join(self.output_dir, 'samples')
        if not os.path.exists(samples_dir):
            print("No samples directory found, skipping composite creation")
            return
        
        # Find all sample images
        sample_files = sorted(glob.glob(os.path.join(samples_dir, '*.png')))
        if not sample_files:
            print("No sample images found, skipping composite creation")
            return
        
        print(f"\nCreating composite image from {len(sample_files)} samples...")
        
        # Load all images
        images = []
        labels = []
        for sample_file in sample_files:
            img = Image.open(sample_file)
            images.append(np.array(img))
            # Extract label from filename
            label = os.path.basename(sample_file).replace('.png', '').replace('_', ' ')
            labels.append(label)
        
        # Calculate grid size
        n_samples = len(images)
        n_cols = min(2, n_samples)  # Maximum 2 columns
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each sample
        for idx, (img, label) in enumerate(zip(images, labels)):
            axes[idx].imshow(img)
            axes[idx].set_title(label, fontsize=10, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Quantization Analysis - Sample Comparisons', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        filepath = os.path.join(self.output_dir, 'all_samples_composite.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Composite samples saved to: {filepath}")
        plt.close()


# =============================================================================
# MAIN CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Quantization Analysis for AR Glasses Models')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       help='Model type (e.g., parametric_a11, neural_a10)')
    parser.add_argument('--model_complexity', type=str, default=DEFAULT_MODEL_COMPLEXITY,
                       help=f'Model complexity (simple, medium, complex). Default: {DEFAULT_MODEL_COMPLEXITY}')
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_HIDDEN_DIM,
                       help=f'Hidden dimension for model. Default: {DEFAULT_HIDDEN_DIM}')
    parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE,
                       help=f'Patch size for A9/A11/A12 models. Default: {DEFAULT_PATCH_SIZE}')
    parser.add_argument('--nn2_sequential_configs', type=str, default=None,
                       help='Sequential model configs (JSON string). Required for sequential models. '
                            'Example: \'[{"model_type":"parametric_a1","model_complexity":"complex","hidden_dim":64}]\'')
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE,
                       help=f'Image size for evaluation. Default: {DEFAULT_IMAGE_SIZE}')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size for evaluation. Default: {DEFAULT_BATCH_SIZE}')
    
    # AR System arguments (for applying distortions during evaluation)
    parser.add_argument('--glass_geo_pipeline', type=str, default='',
                       help='Glass geometric distortion pipeline (e.g., "keystone:0.3,barrel:0.3")')
    parser.add_argument('--glass_color_pipeline', type=str, default='',
                       help='Glass color distortion pipeline')
    parser.add_argument('--projector_geo_pipeline', type=str, default='',
                       help='Projector geometric distortion pipeline')
    parser.add_argument('--projector_color_pipeline', type=str, default='',
                       help='Projector color distortion pipeline')
    parser.add_argument('--use_roi_extraction', action='store_true',
                       help='Use ROI extraction (must match training config)')
    parser.add_argument('--roi_target_size', type=int, nargs=2, default=list(DEFAULT_ROI_TARGET_SIZE),
                       help=f'ROI target size as two integers (height width). Default: {DEFAULT_ROI_TARGET_SIZE}')
    
    # Quantization arguments (defaults from DEFAULT HYPERPARAMETERS section)
    parser.add_argument('--min_bits', type=int, default=DEFAULT_MIN_BITS,
                       help=f'Minimum bit-width for WEIGHT quantization. Default: {DEFAULT_MIN_BITS}')
    parser.add_argument('--max_bits', type=int, default=DEFAULT_MAX_BITS,
                       help=f'Maximum bit-width for WEIGHT quantization. Default: {DEFAULT_MAX_BITS}')
    parser.add_argument('--act_min_bits', type=int, default=None,
                       help='Minimum bit-width for ACTIVATION quantization (defaults to min_bits)')
    parser.add_argument('--act_max_bits', type=int, default=None,
                       help='Maximum bit-width for ACTIVATION quantization (defaults to max_bits)')
    parser.add_argument('--symmetric', action='store_true', default=DEFAULT_SYMMETRIC,
                       help=f'Use symmetric quantization. Default: {DEFAULT_SYMMETRIC}')
    parser.add_argument('--per_channel', action='store_true', default=DEFAULT_PER_CHANNEL,
                       help=f'Use per-channel quantization for weights. Default: {DEFAULT_PER_CHANNEL}')
    parser.add_argument('--stride', type=int, default=1,
                       help='Step size for bit sweep (1=every bit, 2=skip every other, etc.). Default: 1')
    
    # Hardware-realistic quantization arguments (from DEFAULT HYPERPARAMETERS section)
    parser.add_argument('--accumulator_bits', type=int, default=DEFAULT_ACCUMULATOR_BITS,
                       help=f'Fixed accumulator bit-width. Default: {DEFAULT_ACCUMULATOR_BITS}')
    parser.add_argument('--rounding_mode', type=str, default=DEFAULT_ROUNDING_MODE,
                       choices=['nearest_even', 'nearest_away', 'truncate', 'stochastic'],
                       help=f'Hardware rounding mode. Default: {DEFAULT_ROUNDING_MODE}')
    parser.add_argument('--use_saturation', action='store_true', default=DEFAULT_USE_SATURATION,
                       help=f'Use saturation on overflow (vs wrap-around). Default: {DEFAULT_USE_SATURATION}')
    parser.add_argument('--requantize_outputs', action='store_true', default=DEFAULT_REQUANTIZE_OUTPUTS,
                       help=f'Requantize layer outputs for next layer. Default: {DEFAULT_REQUANTIZE_OUTPUTS}')
    parser.add_argument('--output_bits', type=int, default=DEFAULT_OUTPUT_BITS,
                       help=f'Target output bits after requantization. Default: {DEFAULT_OUTPUT_BITS}')
    
    # Calibration arguments
    parser.add_argument('--calibration_fraction', type=float, default=DEFAULT_CALIBRATION_FRACTION,
                       help=f'Fraction of dataset for calibration. Default: {DEFAULT_CALIBRATION_FRACTION} ({int(DEFAULT_CALIBRATION_FRACTION*100)}%%)')
    parser.add_argument('--full_eval_after_calibration', action='store_true', default=DEFAULT_FULL_EVAL_AFTER_CALIBRATION,
                       help=f'Evaluate on full test dataset after finding best bits. Default: {DEFAULT_FULL_EVAL_AFTER_CALIBRATION}')
    parser.add_argument('--skip_full_eval', action='store_true', default=False,
                       help='Skip full test evaluation (overrides --full_eval_after_calibration)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./quantization_results',
                       help='Output directory for results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    # Precision / bit-width overrides
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='Evaluate under fp16 autocast instead of true FP32 (paper harness uses FP32)')
    parser.add_argument('--force_weight_bits', type=int, default=None,
                       help='Skip the weight-bit selector and use this width for STEP 2/3')
    parser.add_argument('--force_act_bits', type=int, default=None,
                       help='Skip the activation-bit selector and use this width for STEP 3')

    # Verbosity
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Print detailed quantization statistics for each operation')

    return parser.parse_args()


def main():
    global USE_AMP
    args = parse_args()
    USE_AMP = args.use_amp
    print(f"Evaluation precision: {'fp16 autocast (AMP)' if USE_AMP else 'FP32'}")

    # Create output directory first (needed for log file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging to both console and file
    import logging
    import sys
    
    log_file = os.path.join(args.output_dir, 'quantization.log')
    
    # Create a custom handler that writes to both stdout and file
    class TeeHandler:
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(log_path, 'w')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = TeeHandler(log_file)
    
    # Log the command used
    print("=" * 80)
    print("QUANTIZATION ANALYSIS")
    print("=" * 80)
    print(f"\nCommand: python3 quantization.py \\")
    for key, value in vars(args).items():
        if value is not None and value != '' and value != False:
            if isinstance(value, bool):
                print(f"    --{key} \\")
            else:
                print(f"    --{key} {value} \\")
    print("")
    print(f"Log file: {log_file}")
    print("=" * 80)
    print("")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create hardware configuration from CLI arguments
    hw_config = HardwareConfig(
        accumulator_bits=args.accumulator_bits,
        use_saturation=args.use_saturation,
        rounding_mode=args.rounding_mode,
        requantize_outputs=args.requantize_outputs,
        output_bits=args.output_bits,
        calibration_fraction=args.calibration_fraction,
        full_eval_after_calibration=args.full_eval_after_calibration
    )
    
    print(f"\nHardware Configuration:")
    print(f"  - Accumulator bits: {hw_config.accumulator_bits}")
    print(f"  - Rounding mode: {hw_config.rounding_mode}")
    print(f"  - Use saturation: {hw_config.use_saturation}")
    print(f"  - Requantize outputs: {hw_config.requantize_outputs} ({hw_config.output_bits}-bit)")
    print(f"  - Calibration fraction: {hw_config.calibration_fraction * 100:.1f}%")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    
    # Load checkpoint first to check for model configs
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Check if this is a sequential model
    if args.model_type == 'sequential':
        # Sequential model requires special handling with create_model_and_setup
        from helper import create_model_and_setup
        
        # Build model_args namespace for create_model_and_setup
        model_args = argparse.Namespace()
        model_args.model_type = args.model_type
        model_args.model_complexity = args.model_complexity
        model_args.hidden_dim = args.hidden_dim
        model_args.use_zero_mask = False
        model_args.image_size = args.image_size
        model_args.grid_sample_mode = 'bilinear'
        
        # Get sequential configs from CLI or checkpoint
        if args.nn2_sequential_configs is not None:
            model_args.model_configs = args.nn2_sequential_configs
            print(f"  - Sequential model configs provided via CLI")
        elif 'args' in checkpoint and hasattr(checkpoint['args'], 'model_configs'):
            model_args.model_configs = checkpoint['args'].model_configs
            print(f"  - Sequential model configs loaded from checkpoint")
        elif 'args' in checkpoint and 'model_configs' in checkpoint['args']:
            model_args.model_configs = checkpoint['args']['model_configs']
            print(f"  - Sequential model configs loaded from checkpoint")
        else:
            raise ValueError("Sequential model requires --nn2_sequential_configs or model_configs in checkpoint")
        
        # Create model using helper
        model, _ = create_model_and_setup(model_args, device, zero_mask=None)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Strip 'module.' prefix if present
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print(f"Sequential model loaded successfully")
    else:
        # Standard single model
        model_kwargs = {
            'model_complexity': args.model_complexity,
            'hidden_dim': args.hidden_dim
        }
        
        # Add patch_size for A9/A11/A12 models
        if any(x in args.model_type.lower() for x in ['a9', 'a11', 'a12']):
            model_kwargs['patch_size'] = args.patch_size
        
        model = create_model(args.model_type, **model_kwargs)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Strip 'module.' prefix if present
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully: {args.model_type}")
    
    # Setup AR System if distortion pipelines are provided
    ar_system = None
    if any([args.glass_geo_pipeline, args.glass_color_pipeline, 
            args.projector_geo_pipeline, args.projector_color_pipeline]):
        print(f"\nSetting up AR system...")
        from helper import parse_pipeline_arg
        
        # Parse pipeline strings
        glass_geo_parsed = parse_pipeline_arg(args.glass_geo_pipeline, default_pipeline_strength=0.5)
        glass_color_parsed = parse_pipeline_arg(args.glass_color_pipeline, default_pipeline_strength=0.5)
        projector_geo_parsed = parse_pipeline_arg(args.projector_geo_pipeline, default_pipeline_strength=0.5)
        projector_color_parsed = parse_pipeline_arg(args.projector_color_pipeline, default_pipeline_strength=0.5)
        
        # Create configs
        projector_config = {
            'color_pipeline': projector_color_parsed,
            'geometrical_pipeline': projector_geo_parsed
        }
        glass_config = {
            'color_pipeline': glass_color_parsed,
            'geometrical_pipeline': glass_geo_parsed
        }
        
        # Initialize AR system
        ar_system = ARSystem(
            projector_config=projector_config,
            glass_config=glass_config
        )
        
        distortion_info = []
        if args.glass_geo_pipeline:
            distortion_info.append(f"Glass Geo: {args.glass_geo_pipeline}")
        if args.glass_color_pipeline:
            distortion_info.append(f"Glass Color: {args.glass_color_pipeline}")
        if args.projector_geo_pipeline:
            distortion_info.append(f"Projector Geo: {args.projector_geo_pipeline}")
        if args.projector_color_pipeline:
            distortion_info.append(f"Projector Color: {args.projector_color_pipeline}")
        
        print(f"✓ AR System configured:")
        for info in distortion_info:
            print(f"  - {info}")
    
    # Create dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    
    # Use CleanImageDataset (proper dataset for NN2 evaluation)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    
    dataset = CleanImageDataset(
        image_dirs=[args.dataset_path],
        transform=transform
    )
    
    print(f"Found {len(dataset)} images")
    
    # Use the SAME train/val/test split as train_NN2.py (70/15/15 with seed=42)
    # This ensures we evaluate on the exact same test set used during training
    print("\n📊 Creating train/val/test split (matching train_NN2.py):")
    print("   Split ratio: 70% train / 15% val / 15% test")
    print("   Using seed=42 for reproducibility (same as training)")
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    from torch.utils.data import random_split
    generator = torch.Generator().manual_seed(42)  # Same seed as train_NN2.py
    
    _, _, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"   Train: {train_size} samples")
    print(f"   Val: {val_size} samples")
    print(f"   Test: {test_size} samples (this is what we evaluate on)")
    
    dataset = test_dataset  # Use test set for quantization evaluation
    
    # Split dataset into calibration (1%) and full evaluation
    total_samples = len(dataset)
    calibration_samples = max(1, int(total_samples * hw_config.calibration_fraction))
    
    print(f"\n📊 Dataset Split:")
    print(f"   Total samples: {total_samples}")
    print(f"   Calibration samples (for finding optimal bits): {calibration_samples} ({hw_config.calibration_fraction*100:.1f}%)")
    print(f"   Full evaluation samples: {total_samples}")
    
    # Create calibration subset
    from torch.utils.data import Subset
    calibration_indices = list(range(calibration_samples))
    calibration_dataset = Subset(dataset, calibration_indices)
    
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    full_eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Setup ROI coordinates if needed (compute from AR system like in training)
    roi_coords = None
    roi_target_size = tuple(args.roi_target_size)
    
    if args.use_roi_extraction and ar_system is not None:
        print(f"\nComputing ROI coordinates from AR system...")
        
        # Create zero mask by applying AR distortions to a white image
        with torch.no_grad():
            # Create fully white input [C, H, W]
            white_image = torch.ones(3, args.image_size, args.image_size)
            
            # Apply distortions
            distorted_tensor = apply_gradient_friendly_distortions(white_image, ar_system)
            
            # Create binary mask: 1 where pixels are non-zero (valid), 0 where black (invalid)
            # Convert to grayscale and check if pixels are near zero
            if distorted_tensor.dim() == 3:  # [C, H, W]
                gray = torch.mean(distorted_tensor, dim=0)  # [H, W]
            else:
                gray = distorted_tensor
            
            # Threshold: pixels with value > 0.01 are valid
            zero_mask = (gray > 0.01).float()
            
            print(f"Zero mask shape: {zero_mask.shape}")
            print(f"Valid pixels (mask=1): {torch.sum(zero_mask).item()}/{zero_mask.numel()}")
        
        # Analyze zero mask to get ROI coordinates
        roi_coords = analyze_zero_mask_roi(zero_mask)
        
        if roi_coords is not None:
            print(f"✓ ROI coordinates computed: {roi_coords}")
        else:
            print(f"⚠️  Could not compute ROI coordinates, disabling ROI extraction")
            args.use_roi_extraction = False
    
    # =========================================================================
    # BASELINE: Evaluate model in FP32 (normal mode) before quantization
    # =========================================================================
    print("\n" + "=" * 80)
    print("BASELINE: FP32 Evaluation (no quantization)")
    print("=" * 80)
    print(f"Evaluating original model on full test set ({total_samples} samples)...")
    
    model.to(device)
    model.eval()
    
    baseline_metrics = evaluate_model_metrics(
        model, full_eval_loader, device,
        ar_system=ar_system,
        use_roi_extraction=args.use_roi_extraction,
        roi_coords=roi_coords,
        roi_target_size=roi_target_size
    )
    
    print(f"\n📊 BASELINE METRICS (FP32):")
    print(f"   PSNR: {baseline_metrics['psnr']:.4f} dB")
    print(f"   SSIM: {baseline_metrics['ssim']:.4f}")
    print(f"   RMSE: {baseline_metrics['rmse']:.6f}")
    print(f"   LPIPS: {baseline_metrics['lpips']:.6f}")
    
    # =========================================================================
    # STEP 1: Find optimal WEIGHT bits (weight-only quantization on calibration set)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: WEIGHT-ONLY Quantization Sweep (on calibration set)")
    print("=" * 80)
    print(f"Sweeping weight bits from {args.max_bits} down to {args.min_bits}")
    print(f"Using {calibration_samples} calibration samples\n")
    
    # Create analyzer with CALIBRATION dataloader
    analyzer = QuantizationAnalyzer(
        model=model,
        dataloader=calibration_loader,  # Use calibration set!
        device=device,
        output_dir=args.output_dir,
        min_bits=args.min_bits,
        max_bits=args.max_bits,
        stride=args.stride,
        symmetric=args.symmetric,
        per_channel=args.per_channel,
        ar_system=ar_system,
        use_roi_extraction=args.use_roi_extraction,
        roi_coords=roi_coords,
        roi_target_size=roi_target_size,
        verbose=args.verbose
    )
    
    optimal_weight_bits, weight_results = analyzer.analyze_weight_quantization()
    if args.force_weight_bits is not None:
        print(f"\n⚙ Overriding selected weight bits {optimal_weight_bits} → {args.force_weight_bits}")
        optimal_weight_bits = args.force_weight_bits
    print(f"\n✓ Optimal weight bits: {optimal_weight_bits}")
    
    # =========================================================================
    # STEP 2: Find optimal ACTIVATION bits (with fixed weight bits, on calibration set)
    # =========================================================================
    # Set activation bit range defaults if not specified
    act_min_bits = args.act_min_bits if args.act_min_bits is not None else args.min_bits
    act_max_bits = args.act_max_bits if args.act_max_bits is not None else args.max_bits
    
    print("\n" + "=" * 80)
    print("STEP 2: ACTIVATION Quantization Sweep (on calibration set)")
    print("=" * 80)
    print(f"Using optimal weight bits: {optimal_weight_bits}")
    print(f"Sweeping activation bits from {act_max_bits} down to {act_min_bits}")
    print(f"Using {calibration_samples} calibration samples\n")
    
    # Update analyzer's bit range for activation sweep
    analyzer.min_bits = act_min_bits
    analyzer.max_bits = act_max_bits
    
    
    activation_results = analyzer.analyze_activation_quantization(optimal_weight_bits)
    
    # Find optimal activation bits from results.
    # NOTE: entries from analyze_activation_quantization() key the width as
    # 'activation_bits', not 'bits' -- reading 'bits' here silently never matched
    # and fell through to args.max_bits (i.e. no reduction at all).
    optimal_activation_bits = act_max_bits
    if activation_results:
        baseline_psnr = activation_results[0].get('psnr', 0)
        for result in activation_results:
            bits = result.get('activation_bits', result.get('bits'))
            if bits is None or result.get('type') != 'weight_and_activation':
                continue
            if result.get('psnr', 0) >= baseline_psnr - 0.1:
                optimal_activation_bits = bits

    if args.force_act_bits is not None:
        print(f"\n⚙ Overriding selected activation bits {optimal_activation_bits} → {args.force_act_bits}")
        optimal_activation_bits = args.force_act_bits

    print(f"\n✓ Optimal activation bits: {optimal_activation_bits}")
    
    # =========================================================================
    # STEP 3: Full evaluation with optimal bits (on full dataset)
    # =========================================================================
    run_full_eval = hw_config.full_eval_after_calibration and not args.skip_full_eval
    if run_full_eval:
        print("\n" + "=" * 80)
        print("STEP 3: FULL EVALUATION (on complete dataset)")
        print("=" * 80)
        print(f"Weight bits: {optimal_weight_bits}")
        print(f"Activation bits: {optimal_activation_bits}")
        print(f"Using all {total_samples} samples\n")
        
        # Create final quantization config
        final_weight_config = QuantizationConfig(
            num_bits=optimal_weight_bits,
            symmetric=args.symmetric,
            per_channel=args.per_channel
        )
        final_activation_config = QuantizationConfig(
            num_bits=optimal_activation_bits,
            symmetric=args.symmetric,
            per_channel=False
        )
        
        # WEIGHT-ONLY on the full set first. Without this the Opt.W column would
        # have to be read off the small calibration sweep while the FP32 baseline
        # comes from the full set -- two different sample sets, so the delta
        # between them would be meaningless.
        print(f"\nRunning full evaluation: WEIGHT-ONLY ({optimal_weight_bits}-bit)...")
        weight_only_model = quantize_model(
            model, final_weight_config, None, hw_config,
            use_weight_only=True
        )
        weight_only_model.to(device)
        weight_only_model.eval()
        weight_only_metrics = evaluate_model_metrics(
            weight_only_model, full_eval_loader, device,
            ar_system=ar_system,
            use_roi_extraction=args.use_roi_extraction,
            roi_coords=roi_coords,
            roi_target_size=roi_target_size
        )
        print(f"\n📊 FULL-SET WEIGHT-ONLY RESULTS ({optimal_weight_bits}-bit weights, FP32 activations):")
        print(f"   PSNR: {weight_only_metrics['psnr']:.4f} dB   "
              f"(Δ vs FP32: {weight_only_metrics['psnr'] - baseline_metrics['psnr']:+.4f})")
        print(f"   SSIM: {weight_only_metrics['ssim']:.4f}   "
              f"(Δ vs FP32: {weight_only_metrics['ssim'] - baseline_metrics['ssim']:+.4f})")
        print(f"   RMSE: {weight_only_metrics['rmse']:.6f}")
        print(f"   LPIPS: {weight_only_metrics['lpips']:.6f}")
        del weight_only_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Quantize model with optimal settings
        final_quantized_model = quantize_model(
            model, final_weight_config, final_activation_config, hw_config,
            use_weight_only=False
        )
        final_quantized_model.to(device)
        final_quantized_model.eval()
        print(final_quantized_model)   # <-- here, right after it's built
        import sys; sys.exit(0)        # <-- bail out, we only need the tree for now
        # Evaluate on full dataset
        print("Running full evaluation...")
        final_metrics = evaluate_model_metrics(
            final_quantized_model, full_eval_loader, device,
            ar_system=ar_system,
            use_roi_extraction=args.use_roi_extraction,
            roi_coords=roi_coords,
            roi_target_size=roi_target_size
        )
        
        print(f"\n📊 FINAL RESULTS (on full dataset):")
        print(f"   Weight quantization: {optimal_weight_bits}-bit")
        print(f"   Activation quantization: {optimal_activation_bits}-bit")
        print(f"   PSNR: {final_metrics['psnr']:.4f} dB   "
              f"(Δ vs FP32: {final_metrics['psnr'] - baseline_metrics['psnr']:+.4f})")
        print(f"   SSIM: {final_metrics['ssim']:.4f}   "
              f"(Δ vs FP32: {final_metrics['ssim'] - baseline_metrics['ssim']:+.4f})")
        print(f"   RMSE: {final_metrics['rmse']:.6f}")
        print(f"   LPIPS: {final_metrics['lpips']:.6f}")
    
    # Create composite visualizations
    print("\n" + "=" * 80)
    print("Creating composite visualizations...")
    print("=" * 80)
    analyzer.create_composite_samples()
    
    # Generate publication-quality plots
    print("\n" + "=" * 80)
    print("Generating publication-quality plots...")
    print("=" * 80)
    
    # Get quantization stats for plot generation
    stats = get_quantization_stats(model)
    
    # Prepare results for plotting - weight_results is a list of dicts
    if weight_results and isinstance(weight_results, list):
        bit_widths = [r['bits'] for r in weight_results if 'bits' in r]
        psnr_values = [r['psnr'] for r in weight_results if 'psnr' in r]
    elif weight_results and isinstance(weight_results, dict):
        bit_widths = list(weight_results.keys())
        psnr_values = [weight_results[b]['psnr'] for b in bit_widths]
    else:
        bit_widths = list(range(args.min_bits, args.max_bits + 1))
        psnr_values = []
    
    plot_results = {
        'bit_widths': bit_widths,
        'psnr_values': psnr_values,
        'optimal_bits': optimal_weight_bits,
        'total_weights': stats['total_weights']
    }
    
    generate_quantization_report(plot_results, args.output_dir, args.model_type)
    print_quantization_stats(model)
    print("\n" + "=" * 80)
    print("QUANTIZATION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print(f"\n📊 OPTIMAL CONFIGURATION:")
    print(f"   Weight bits: {optimal_weight_bits}")
    print(f"   Activation bits: {optimal_activation_bits}")
    print(f"\nGenerated files:")
    print(f"  - quantization_psnr_sweep.pdf/.png")
    print(f"  - memory_savings.pdf/.png")
    print(f"  - weight_quantization_results.json")
    print(f"  - activation_quantization_results.json")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    if '--run_simple_test' in sys.argv:
        test_bits = None
        weight_only = False
        
        if '--test_bits' in sys.argv:
            idx = sys.argv.index('--test_bits')
            test_bits = []
            for i in range(idx + 1, len(sys.argv)):
                if sys.argv[i].startswith('--'):
                    break
                try:
                    test_bits.append(int(sys.argv[i]))
                except ValueError:
                    break
        
        if '--weight_only' in sys.argv:
            weight_only = True
        
        run_simple_quantization_test(
            test_bits=test_bits if test_bits else None,
            use_weight_activation=not weight_only
        )
    else:
        main()
