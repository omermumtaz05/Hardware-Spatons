#!/usr/bin/env python3
"""
Train NN2 for AR Glasses Project

NN2 Training Phase:
- Goal: Train a neural network that can pre-distort clean images such that when they go through 
  the AR system (NN1), the final result looks like the original clean image
- Pipeline: clean image -> NN2 (trainable) -> NN1 (frozen) -> target (should match clean image)
- NN1 is a pre-trained frozen model from train-NN1.py
- Loss: Compare NN1(NN2(clean)) with clean image

This script requires a trained NN1 model from train-NN1.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import torch.amp as amp
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import logging
import random
import json
import argparse
import pickle
import multiprocessing
# from tqdm import tqdm  # Replaced with custom printing
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Import project modules
from models import create_model, count_parameters, geometry_loss, color_loss, combined_geometry_color_loss

# Grid sample configuration is now integrated in models.py
from AR_System import ARSystem, create_default_ar_system
from Data import ImageDataset, seed_worker
from model_info import ModelProfiler, check_model_analysis_exists

# Import helper functions to reduce code duplication
from helper import (
    setup_device, create_pipeline_configs, initialize_training_history,
    save_sample_images, create_zero_mask_from_loader, denormalize_image,
    create_nn2_training_plots, create_model_and_setup
)
from log import (
    log_step_header, log_experiment_setup, log_training_configuration,
    log_model_info, log_scheduler_info, log_epoch_progress,
    # NN2-specific logging functions
    log_nn1_model_loading, log_nn1_model_loaded, log_nn1_config_validation,
    log_roi_analysis_start, log_roi_analysis_result, log_roi_analysis_fallback,
    log_nn2_training_start, log_nn2_epoch_progress, log_nn2_best_model_saved,
    log_nn2_training_completion, log_nn2_experiment_setup, log_nn2_dataset_info,
    log_nn2_placeholder_created, log_nn2_experiment_id_info,
    log_dataset_mismatch_warning, log_real_dataset_loaded,
    log_nn2_final_test_results, log_nn2_data_loader_info,
    log_nn2_model_parameters, log_nn2_training_parameters, log_nn2_loss_parameters,
    log_nn2_optimizer_parameters, log_nn2_scheduler_parameters,
    log_nn2_training_control_parameters, log_nn2_data_parameters,
    log_nn2_configuration_parameters
)

# Import loss functions and metrics
import lpips
import torchmetrics
import kornia.losses as losses

# =============================================================================
# SEED SETTING FOR REPRODUCIBILITY
# =============================================================================

def set_seed(seed):
    """
    Set random seeds for reproducible results.
    
    Args:
        seed (int): Random seed value
    """
    import numpy as np
    import random
    import torch
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable strict deterministic algorithms where possible
    # We use warn_only=True to avoid crashing on ops that don't have deterministic versions
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        # Older torch versions might not have this
        pass
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for deterministic cuBLAS
    
    logging.info(f"Random seed set to: {seed}")




# Training hyperparameters
EPOCHS = 30
BATCH_SIZE = 16
INITIAL_LR = 5e-4  # Lower learning rate for NN2
WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42  # Default seed for reproducible results 1/42/9999
MODEL_TYPE = 'neural_a1'  # 'unet', 'resnet', 'mobilenet', 'unetpp', 'neural_a1', 'neural_a2', 'neural_a3', 'neural_a4', 'parametric_a1', 'parametric_a2', 'parametric_a3', 'parametric_a4'
LOSS_TYPE = 'combined_perceptual'
SSIM_COEFFICIENT = 0.3
PERCEPTUAL_COEFFICIENT = 0.3
USE_AMP = False
NORM = False
MIN_LR = 1e-6

# Default pipeline configurations (similar to train.py)
# Pipeline syntax: 'name:strength,name:strength' (comma-separated)
# 
# Supported COLOR distortions:
#   - brightness_contrast [0–1], default 0.3 (combined uniform brightness+contrast)
#   - brightness [-1–1], default ±0.3 (non-uniform: positive=brighter, negative=darker)
#   - contrast [-1–1], default ±0.5 (non-uniform: positive=more contrast, negative=less contrast)
#   - hue_saturation [0–1], default 0.2  
#   - color_balance [0–1], default 0.2
#   - gamma [0–1], default 0.2
#   - channel_mixing [0–1], default 0.2
#   - vignetting [0–1], default 0.3
#   - color_tint [0–1], default 0.2
#   - chromatic_aberration [0–1], default 0.3
#   - blur [0–1], default 0.2
#
# Supported GEOMETRIC distortions:
#   - fisheye [0–1], default 0.3
#   - barrel [0–1], default 0.3
#   - pincushion [0–1], default 0.3
#   - perspective [0–1], default 0.3
#   - wave [0–1], default 0.3
#   - radial_wave [0–1], default 0.3
#   - radial_tangential [0–1], default 0.4
#   - keystone [0–1], default 0.3

# Default pipelines for Projector
PROJECTOR_COLOR_PIPELINE = ''
PROJECTOR_GEO_PIPELINE = ''

# Default pipelines for Glass
GLASS_COLOR_PIPELINE = ''  # No color distortions by default
GLASS_GEO_PIPELINE = ''

# Default pipeline strength (used if an operation lacks :strength)
DEFAULT_PIPELINE_STRENGTH = 0.5

# Global variable to store ROI coordinates after analysis
GLOBAL_ROI_COORDS = None


# Data paths
IMAGE_DIRS = ['/home/maheron/meta/dataset/images.cv', '/home/maheron/meta/dataset/places365']

# Device settings
GPU_INDEX = 0

from Data import CleanImageDataset
from Data import RealDataset

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train NN2 for AR Glasses project')
    
    # NN1 Model parameters
    parser.add_argument('--nn1_model_type', type=str, default=MODEL_TYPE,
                       choices=['unet', 'resnet', 'mobilenet', 'unetpp', 'neural_a1', 'neural_a2', 'neural_a3', 'neural_a4', 'neural_a5', 'neural_a6', 'neural_a7', 'neural_a8', 'parametric_a1', 'parametric_a2', 'parametric_a3', 'parametric_a4', 'parametric_a5', 'parametric_a6', 'parametric_a7', 'parametric_a8'],
                       help='Type of model for NN1 (case-insensitive): unet, resnet, mobilenet, unetpp, neural_a1-a8 (Neural approaches), parametric_a1-a8 (Parametric approaches)')
    parser.add_argument('--nn1_model_complexity', type=str, default='medium',
                       choices=['simple', 'medium', 'complex', 'complexer', 'fisheye'],
                       help='Model complexity for NN1 neural/parametric models: simple (matrix), medium (2-layer), complex (3-layer+residual), fisheye (traditional distortion correction)')
    parser.add_argument('--nn1_hidden_dim', type=int, default=256,
                       help='Hidden dimension size for NN1 neural models (default: 256)')
    parser.add_argument('--nn1_use_zero_mask', action='store_true', default=False,
                       help='Use zero mask for NN1 model forward pass during training (default: False)')
    parser.add_argument('--nn1_model_path', type=str, required=False,
                       help='Path to trained NN1 model (.pth file) - required only when not using --dont_use_nn1')
    parser.add_argument('--dont_use_nn1', action='store_true', default=False,
                       help='Use distortion formulas instead of NN1 model for training (default: False)')
    
    # NN2 Model parameters
    parser.add_argument('--nn2_model_type', type=str, default=MODEL_TYPE,
                       choices=['unet', 'resnet', 'mobilenet', 'unetpp', 'neural_a1', 'neural_a2', 'neural_a3', 'neural_a4', 'neural_a5',
                        'neural_a6', 'neural_a7', 'neural_a8', 'neural_a9', 'neural_a10', 'neural_a11', 'neural_a12', 'neural_a13', 'neural_a14', 'parametric_a1', 'parametric_a2', 'parametric_a3', 'parametric_a4',
                         'parametric_a5', 'parametric_a6', 'parametric_a7', 'parametric_a8', 'parametric_a9', 'parametric_a10', 'parametric_a11', 'parametric_a12', 'parametric_a13', 'parametric_a14', 'kan_a1', 'kan_a2', 'kan_a3', 'kan_a4',
                          'kan_a5', 'kan_a6', 'kan_a7', 'kan_a8', 'kan_a9', 'kan_a10', 'kan_a11', 'kan_a12', 'kan_a13', 'kan_a14',
                          'inverse_lsnu', 'inverse_blur', 'inverse_spatiotemporal_dithering', 'inverse_elevated_black_level',
                          'inverse_color_fringes', 'inverse_wgnu', 'inverse_brightness_contrast', 'inverse_color_balance',
                          'inverse_fisheye', 'inverse_keystone', 'compennetplusplus', 'sequential', 'nafnet'],
                       help='Type of model to train for NN2: unet, resnet, mobilenet, unetpp, neural_a1-a14, parametric_a1-a14, kan_a1-a14, compennetplusplus, sequential, nafnet, or inverse_* models (mathematical inverse of distortions)')
    parser.add_argument('--nn2_nafnet_preset', type=str, default='small',
                       choices=['small', 'medium', 'baseline', 'deep32', 'baseline64'],
                       help='NAFNet (Chen et al. ECCV 2022) preset. On a 24 GB GPU at 1024² B=4 FP32, '
                            'only small (~1.14M, 16.9 GB peak) and medium (~2.9M, 19.0 GB peak) fit; '
                            'baseline/deep32/baseline64 are the upstream presets and OOM at this resolution.')
    parser.add_argument('--nn2_model_complexity', type=str, default='medium',
                       choices=['simple', 'medium', 'complex', 'complexer', 'complex_with_hsl', 'fisheye'],
                       help='Model complexity for NN2 neural/parametric models: simple (matrix), medium (2-layer), complex (3-layer+residual), complex_with_hsl (HSL color space for A4), fisheye (traditional distortion correction)')
    parser.add_argument('--nn2_hidden_dim', type=int, default=256,
                       help='Hidden dimension size for NN2 neural models (default: 256, ignored for approaches 5, 6, and 8)')
    parser.add_argument('--nn2_patch_size', type=int, default=3, choices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                       help='Patch size for NN2 approaches A9, A11, A12, and A14 (default: 3). For A9/A11/A12, must be odd. For A14, can be any positive integer. patch_size² pixels will be used (e.g., 3=9, 4=16, 5=25 pixels). For A13, use input_patch_size and output_patch_size instead.')
    parser.add_argument('--nn2_input_patch_size', type=int, default=5, choices=[3, 5, 7, 9, 11],
                       help='Input patch size for NN2 approach A13 (default: 5). Must be odd. input_patch_size² pixels will be used as input.')
    parser.add_argument('--nn2_output_patch_size', type=int, default=3, choices=[1, 3, 5, 7],
                       help='Output patch size for NN2 approach A13 (default: 3). Must be odd and <= input_patch_size. output_patch_size² pixels will be predicted per call.')
    parser.add_argument('--nn2_use_zero_mask', action='store_true', default=False,
                       help='Use zero mask for NN2 model forward pass during training (default: False)')
    
    # UNet++ specific parameters
    parser.add_argument('--nn2_unetpp_decoder_channels', type=str, default=None,
                       help='Decoder channels for UNet++ model as comma-separated values, e.g. "256,128,64,32,16" or "128,64,32,16,8" (default: None, uses default based on encoder_depth)')
    parser.add_argument('--nn2_unetpp_encoder_depth', type=int, default=None, choices=[2, 3, 4, 5],
                       help='Encoder depth for UNet++ model: 2, 3, 4, or 5 (default: 5). Length must match decoder_channels length.')
    
    # UNet specific parameters
    parser.add_argument('--nn1_unet_base_features', type=int, default=64,
                       choices=[16, 32, 48, 64, 96, 128],
                       help='Base features for NN1 UNet model: 16, 32, 48, 64 (default), 96, or 128. Smaller values create smaller models.')
    parser.add_argument('--nn2_unet_base_features', type=int, default=64,
                       choices=[16, 32, 48, 64, 96, 128],
                       help='Base features for NN2 UNet model: 16, 32, 48, 64 (default), 96, or 128. Smaller values create smaller models.')
    
    # Two-Network Approach Parameters (A5, A6, A8)
    parser.add_argument('--nn2_net1_type', type=str, default='neural',
                       choices=['neural', 'parametric', 'unetpp'],
                       help='Model type for net1 in two-network approaches (A5, A6, A8) (default: neural)')
    parser.add_argument('--nn2_net1_complexity', type=str, default='medium',
                       choices=['simple', 'medium', 'complex', 'complexer', 'fisheye'],
                       help='Model complexity for net1 in two-network approaches (A5, A6, A8) (default: medium)')
    parser.add_argument('--nn2_net1_hidden_dim', type=int, default=256,
                       help='Hidden dimension size for net1 in two-network approaches (A5, A6, A8) (default: 256)')
    parser.add_argument('--nn2_net2_type', type=str, default='neural',
                       choices=['neural', 'parametric', 'unetpp'],
                       help='Model type for net2 in two-network approaches (A5, A6, A8) (default: neural)')
    parser.add_argument('--nn2_net2_complexity', type=str, default='medium',
                       choices=['simple', 'medium', 'complex', 'complexer'],
                       help='Model complexity for net2 in two-network approaches (A5, A6, A8) (default: medium, fisheye not supported for net2)')
    parser.add_argument('--nn2_net2_hidden_dim', type=int, default=256,
                       help='Hidden dimension size for net2 in two-network approaches (A5, A6, A8) (default: 256)')
    parser.add_argument('--nn2_loss_variant', type=str, default='5-1',
                       choices=['5-1', '5-2', '6-1', '6-2', '8-1', '8-2'],
                       help='Loss variant for two-network approaches: 5-1/6-1/8-1 (both outputs), 5-2/6-2/8-2 (final output only) (default: 5-1)')
    
    # Sequential model parameters
    parser.add_argument('--nn2_sequential_configs', type=str, default=None,
                       help='JSON string or path to JSON file containing sequential model configurations. Required when --nn2_model_type=sequential. Example: \'[{"model_type":"parametric_a1","model_complexity":"complex","hidden_dim":64},{"model_type":"parametric_a4","model_complexity":"complex","hidden_dim":64}]\'')
    
    # AR System pipeline parameters (similar to train-NN1.py)
    parser.add_argument('--projector_color_pipeline', type=str, default=PROJECTOR_COLOR_PIPELINE,
                       help='Comma-separated projector color distortions, e.g. vignetting:0.3,blur:0.2 (required when using --dont_use_nn1)')
    parser.add_argument('--projector_geo_pipeline', type=str, default=PROJECTOR_GEO_PIPELINE,
                       help='Comma-separated projector geometric distortions, e.g. keystone:0.2,perspective:0.3 (required when using --dont_use_nn1)')
    parser.add_argument('--glass_color_pipeline', type=str, default=GLASS_COLOR_PIPELINE,
                       help='Comma-separated glass color distortions, e.g. brightness_contrast:0.2 (required when using --dont_use_nn1)')
    parser.add_argument('--glass_geo_pipeline', type=str, default=GLASS_GEO_PIPELINE,
                       help='Comma-separated glass geometric distortions, e.g. radial_tangential:0.4,barrel:0.3 (required when using --dont_use_nn1)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=INITIAL_LR,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                       help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'adam', 'sgd', 'rmsprop'],
                       help='Optimizer type (default: adamw)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer (default: 0.9)')
    parser.add_argument('--beta1', type=float, default=0.9,
                       help='Beta1 for Adam optimizer (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Beta2 for Adam optimizer (default: 0.999)')
    
    # Training parameters
    parser.add_argument('--loss_type', type=str, default=LOSS_TYPE,
                       choices=['mse', 'l1', 'ssim', 'perceptual', 'combined_perceptual', 'combined_mse_l1_perceptual', 'geometry_loss'],
                       help='Loss function type')
    parser.add_argument('--ssim_coefficient', type=float, default=SSIM_COEFFICIENT,
                       help='SSIM coefficient for combined loss (default: 0.3)')
    parser.add_argument('--perceptual_coefficient', type=float, default=PERCEPTUAL_COEFFICIENT,
                       help='Perceptual loss coefficient for combined loss (default: 0.1)')
    parser.add_argument('--min_lr', type=float, default=MIN_LR,
                       help='Minimum learning rate for scheduler (default: 1e-6)')
    parser.add_argument('--use_amp', action='store_true', default=USE_AMP,
                       help='Use automatic mixed precision (disabled when using --dont_use_nn1)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help=f'Random seed for reproducible results (default: {DEFAULT_SEED})')
    parser.add_argument('--norm', action='store_true', default=NORM,
                       help='Use ImageNet normalization')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size (height and width, default: 256). Used for preprocessing, ROI, and architecture dimensions.')
    
    # Multi-resolution training for AMP compatibility with large images
    parser.add_argument('--use_multires', action='store_true', default=False,
                       help='Enable multi-resolution training: downsample to nn2_resolution, train NN2, upsample back to image_size (default: False)')
    parser.add_argument('--nn2_resolution', type=int, default=256,
                       help='Resolution for NN2 processing when using --use_multires (default: 256). NN2 runs at this resolution to avoid AMP issues with grid_sample.')
    parser.add_argument('--use_learnable_resampler', action='store_true', default=False,
                       help='Use trainable CNN for downsample/upsample instead of fixed bilinear (default: False). Only applies when --use_multires is enabled.')
    
    parser.add_argument('--use_roi_extraction', action='store_true', default=False,
                       help='Use ROI extraction to handle image size changes from distortions (default: False)')
    parser.add_argument('--roi_target_size', type=int, nargs=2, default=None,
                       metavar=('HEIGHT', 'WIDTH'),
                       help='Target size for ROI extraction (default: [image_size, image_size])')

    parser.add_argument('--gpu_index', type=int, default=GPU_INDEX,
                       help='GPU index to use (0 for first GPU, 1 for second GPU, etc.). Use -1 for CPU only.')
    
    # Additional training hyperparameters
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loader workers (default: 8)')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='Use pinned memory for data loading (default: True)')
    parser.add_argument('--dataset_fraction', type=float, default=1.0,
                       help='Fraction of dataset to use for training (0.0-1.0, default: 1.0 for full dataset)')
    parser.add_argument('--gradient_clip', type=float, default=None,
                       help='Gradient clipping value (default: None)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Number of warmup epochs (default: 0)')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'cosine_warm_restarts', 'step', 'exponential', 'plateau'],
                       help='Learning rate scheduler type (default: cosine)')
    parser.add_argument('--restart_period', type=int, default=0,
                       help='Initial restart period (in epochs) for CosineAnnealingWarmRestarts. 0=auto')
    parser.add_argument('--restart_mult', type=int, default=1,
                       help='T_mult for CosineAnnealingWarmRestarts (default: 1.0)')
    parser.add_argument('--step_size', type=int, default=30,
                       help='Step size for step scheduler (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for step scheduler (default: 0.1)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience for plateau scheduler (default: 5)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate for models that support it (default: None)')
    parser.add_argument('--batch_norm', action='store_true', default=False,
                       help='Use batch normalization (default: False)')
    parser.add_argument('--early_stopping', action='store_true', default=False,
                       help='Enable early stopping (default: False)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    # Three-stage training for two-network approaches (A5, A6, A8)
    parser.add_argument('--freeze_net1', action='store_true', default=False,
                       help='Freeze net1 parameters (A5/A6/A8)')
    parser.add_argument('--freeze_net2', action='store_true', default=False,
                       help='Freeze net2 parameters (A5/A6/A8)')
    parser.add_argument('--net1_pretrained_path', type=str, default=None,
                       help='Path to pretrained net1 model for loading in approach 5')
    parser.add_argument('--net2_pretrained_path', type=str, default=None,
                       help='Path to pretrained net2 model for loading in approach 5')
    parser.add_argument('--save_networks_separately', action='store_true', default=False,
                       help='Save net1 and net2 separately in approach 5')
    parser.add_argument('--augment', action='store_true', default=False,
                       help='Enable data augmentation (default: False)')
    
    # Grid sample configuration hyperparameters
    parser.add_argument('--grid_sample_mode', type=str, default='bilinear', 
                       choices=['nearest', 'bilinear', 'bicubic'],
                       help='Grid sample interpolation mode: nearest (1-point), bilinear (4-point), or bicubic (16-point) (default: bilinear)')
    
    # Data parameters
    parser.add_argument('--image_dirs', type=str, nargs='+', default=IMAGE_DIRS,
                       help='Directories containing training images')
    
    # New: Real dataset mode (clean + distorted pairs)
    parser.add_argument('--use_real_dataset', action='store_true', default=False,
                       help='Use pre-existing real dataset with clean/distorted pairs instead of AR system')
    parser.add_argument('--clean_train_dir', type=str, default='',
                       help='Directory containing clean training images (when using real dataset)')
    parser.add_argument('--clean_test_dir', type=str, default='',
                       help='Directory containing clean test images (when using real dataset)')
    parser.add_argument('--distorted_train_dir', type=str, default='',
                       help='Directory containing distorted training images (when using real dataset)')
    parser.add_argument('--distorted_test_dir', type=str, default='',
                       help='Directory containing distorted test images (when using real dataset)')
    
    # Output parameters
    parser.add_argument('--comment', type=str, default='',
                       help='Optional comment for this training run')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation on existing model')
    parser.add_argument('--pretrained_model', type=str, default='',
                       help='Path to pretrained NN2 model for evaluation')
    parser.add_argument('--nn2_pretrained_path', type=str, default=None,
                       help='Path to pretrained NN2 model checkpoint to load and continue training from')
    
    # Server identification (required)
    parser.add_argument('--server_number', type=str, required=True,
                       help='Server identifier string (e.g. "server9"). Results will be saved to "{server_number}_NN2-results" directory.')
    
    return parser.parse_args()


def parse_pipeline_arg(arg_str):
    """
    Parse pipeline argument string into list of distortion dictionaries.
    Format: 'name:strength,name:strength' -> [{'type': name, 'strength': strength}, ...]
    """
    pipeline = []
    if not arg_str:
        return pipeline
    parts = [p.strip() for p in arg_str.split(',') if p.strip()]
    for p in parts:
        if ':' in p:
            name, val = p.split(':', 1)
            try:
                strength = float(val)
            except ValueError:
                strength = DEFAULT_PIPELINE_STRENGTH
        else:
            name = p
            strength = DEFAULT_PIPELINE_STRENGTH
        pipeline.append({'type': name, 'strength': strength})
    return pipeline


def pipeline_to_str(pipe):
    """Convert pipeline list back to string for logging"""
    if isinstance(pipe, str):
        return pipe
    elif isinstance(pipe, dict):
        # Handle config dictionary with color_pipeline and geometrical_pipeline keys
        if 'color_pipeline' in pipe and 'geometrical_pipeline' in pipe:
            color_str = ','.join([f"{d['type']}:{d['strength']:.2f}" for d in pipe['color_pipeline']]) if pipe['color_pipeline'] else 'none'
            geo_str = ','.join([f"{d['type']}:{d['strength']:.2f}" for d in pipe['geometrical_pipeline']]) if pipe['geometrical_pipeline'] else 'none'
            return f"color:[{color_str}], geo:[{geo_str}]"
        else:
            return str(pipe)
    elif isinstance(pipe, list):
        return ','.join([f"{d['type']}:{d['strength']:.2f}" for d in pipe]) if pipe else 'none'
    else:
        return str(pipe)


def get_next_experiment_id(nn2_results_dir="NN2-results"):
    """Determine the next experiment identifier by checking existing files in the results directory"""
    import re
    import time

    if not os.path.exists(nn2_results_dir):
        logging.warning(f"{nn2_results_dir} directory not found, starting with experiment #1")
        return 1

    max_id = 0
    found_identified = False

    try:
        # Check all subdirectories and files
        for item in os.listdir(nn2_results_dir):
            # Check for #N- pattern first (modern format)
            match = re.match(r"^#(\d+)-", item)
            if match:
                experiment_id = int(match.group(1))
                max_id = max(max_id, experiment_id)
                found_identified = True
                continue
            
            # Check for placeholder files pattern: .experiment_#N_placeholder
            placeholder_match = re.match(r"^\.experiment_#(\d+)_placeholder", item)
            if placeholder_match:
                experiment_id = int(placeholder_match.group(1))
                max_id = max(max_id, experiment_id)
                found_identified = True
                continue

            # Fallback: check for NN2 files without identifier (legacy)
            if item.startswith("NN2_") and "_e" in item and "_lr" in item and "_bs" in item:
                if not found_identified:
                    max_id = max(max_id, 1)  # If we find legacy files but no identified ones, start from 1

    except OSError as e:
        logging.warning(f"Error reading {nn2_results_dir} directory: {e}, starting with experiment #1")
        return 1

    next_id = max_id + 1

    # Create placeholder to prevent race conditions
    placeholder_file = f"{nn2_results_dir}/.experiment_#{next_id}_placeholder"
    try:
        # Try to create placeholder file atomically
        with open(placeholder_file, 'w') as f:
            f.write(f"Experiment #{next_id} placeholder - created at {time.time()}\n")
        log_nn2_placeholder_created(next_id)
    except (OSError, IOError) as e:
        logging.warning(f"Failed to create placeholder for experiment #{next_id}: {e}")
        # If we can't create placeholder, try next ID
        next_id += 1
        placeholder_file = f"{nn2_results_dir}/.experiment_#{next_id}_placeholder"
        try:
            with open(placeholder_file, 'w') as f:
                f.write(f"Experiment #{next_id} placeholder - created at {time.time()}\n")
            log_nn2_placeholder_created(next_id, is_fallback=True)
        except (OSError, IOError) as e2:
            logging.error(f"Failed to create placeholder for fallback experiment #{next_id}: {e2}")

    log_nn2_experiment_id_info(next_id, list(range(1, max_id + 1)), nn2_results_dir=nn2_results_dir)
    return next_id


def create_experiment_name(args, projector_config, glass_config):
    """Create experiment name for saving results"""
    # Abbreviation mapping for distortion names
    abbr_map = {
        'brightness_contrast': 'bc', 'brightness_positive': 'br+', 'brightness_negative': 'br-', 'brightness_both': 'br±', 
        'brightness_positive_fixed': 'br+f', 'brightness_negative_fixed': 'br-f', 'brightness_both_fixed': 'br±f', 
        'contrast': 'co', 'contrast_fixed': 'cof',
        'hue_saturation': 'hs', 'color_balance': 'cb',
        'gamma': 'g', 'channel_mixing': 'cm', 'vignetting': 'v', 'color_tint': 'ct',
        'chromatic_aberration': 'ca', 'blur': 'b',
        'vdp_wgnu_hashed': 'wgnu_h',
        'fisheye': 'f', 'barrel': 'ba', 'pincushion': 'p', 'perspective': 'pe',
        'wave': 'w', 'radial_wave': 'rw', 'radial_tangential': 'rt', 'keystone': 'k'
    }
    
    def encode_pipeline(pipeline):
        if not pipeline:
            return 'none'
        encoded = []
        for dist in pipeline:
            dist_type = dist['type']
            strength = dist['strength']
            abbr = abbr_map.get(dist_type, dist_type[:2])  # fallback to first 2 chars
            encoded.append(f"{abbr}{int(round(strength*100))}")
        return '-'.join(encoded)
    
    # Check if using real dataset mode
    if hasattr(args, 'use_real_dataset') and args.use_real_dataset:
        # Real dataset mode - use dataset paths instead of AR system config
        dataset_str = "real_dataset"
        if args.clean_train_dir:
            # Extract dataset name from path
            clean_path = args.clean_train_dir.rstrip('/')
            dataset_name = os.path.basename(clean_path)
            if dataset_name in ['train', 'test']:
                dataset_name = os.path.basename(os.path.dirname(clean_path))
            dataset_str = f"real_{dataset_name}"
        
        # Include model type, complexity, and approach information for real dataset mode
        model_str = args.nn2_model_type

        # Handle neural models (new naming convention)
        if args.nn2_model_type.startswith('neural_a'):
            approach = args.nn2_model_type.split('_')[-1]
            complexity_map = {'simple': '1', 'medium': '2', 'complex': '3'}
            complexity_num = complexity_map.get(args.nn2_model_complexity, '2')
            # For two-network approaches (A5, A6, A8), include all three networks info
            if approach in ['a5', 'a6', 'a8']:
                # Shortened naming to avoid filename length issues
                nn1_info = f"nn1_{args.nn1_model_type.split('_')[-1]}"  # Just the approach part
                net1_info = f"n1-{args.nn2_net1_type[0]}-{args.nn2_net1_complexity[0]}-{args.nn2_net1_hidden_dim}"
                net2_info = f"n2-{args.nn2_net2_type[0]}-{args.nn2_net2_complexity[0]}-{args.nn2_net2_hidden_dim}"
                
                model_str = f"neural_{approach}_{args.nn2_loss_variant}_{nn1_info}_{net1_info}_{net2_info}"
            else:
                model_str = f"neural_{approach}_{complexity_num}-{args.nn2_hidden_dim}"

        # Handle parametric models (new naming convention)
        elif args.nn2_model_type.startswith('parametric_a'):
            approach = args.nn2_model_type.split('_')[-1]
            if args.nn2_model_complexity == 'fisheye':
                complexity_abbr = 'fish'
            else:
                complexity_map = {'simple': 'simp', 'medium': 'med', 'complex': 'comp', 'complexer': 'comp2'}
                complexity_abbr = complexity_map.get(args.nn2_model_complexity, 'med')
            model_str = f"parametric_{approach}_{complexity_abbr}"

        # Handle other model types (UNet, ResNet, etc.)
        else:
            model_str = args.nn2_model_type

        experiment_name = f"NN2_{dataset_str}_{model_str}_e{args.epochs}_lr{args.lr:.0e}_bs{args.batch_size}_{args.loss_type}"

        # Add configuration flags for real dataset mode
        if hasattr(args, 'nn2_use_zero_mask') and args.nn2_use_zero_mask:
            experiment_name += "_zeromask"
        if hasattr(args, 'norm') and args.norm:
            experiment_name += "_norm"
        if hasattr(args, 'use_amp') and args.use_amp:
            experiment_name += "_amp"
        if hasattr(args, 'dont_use_nn1') and args.dont_use_nn1:
            experiment_name += "_no_nn1"
        if hasattr(args, 'min_lr'):
            experiment_name += f"_minlr{args.min_lr:.0e}"
        if hasattr(args, 'weight_decay') and args.weight_decay != 0:
            experiment_name += f"_wd{args.weight_decay:.0e}"
        if hasattr(args, 'seed') and args.seed != DEFAULT_SEED:
            experiment_name += f"_seed{args.seed}"
        
        # Add grid_sample configuration for real dataset mode
        experiment_name += f"_grid-lib-{args.grid_sample_mode}"
        
        # Add image size to experiment name
        if hasattr(args, 'image_size') and args.image_size != 256:
            experiment_name += f"_img{args.image_size}"

        if args.comment:
            experiment_name += f"_{args.comment}"

    else:
        # AR System mode - use pipeline configurations
        # Create identifiers for projector config
        try:
            proj_color = encode_pipeline(projector_config.get('color_pipeline', []))
            proj_geo = encode_pipeline(projector_config.get('geometrical_pipeline', []))
            proj_str = f"proj_c[{proj_color}]_g[{proj_geo}]"
        except (KeyError, TypeError) as e:
            logging.warning(f"Error processing projector config: {e}")
            proj_str = "proj_c[none]_g[none]"
    
    # Create identifiers for glass config  
        try:
            glass_color = encode_pipeline(glass_config.get('color_pipeline', []))
            glass_geo = encode_pipeline(glass_config.get('geometrical_pipeline', []))
            glass_str = f"glass_c[{glass_color}]_g[{glass_geo}]"
        except (KeyError, TypeError) as e:
            logging.warning(f"Error processing glass config: {e}")
            glass_str = "glass_c[none]_g[none]"
        
        # Include dataset name from image_dirs
        dataset_str = ""
        if hasattr(args, 'image_dirs') and args.image_dirs:
            if isinstance(args.image_dirs, list):
                # Extract dataset names from multiple directories
                dataset_names = []
                for img_dir in args.image_dirs:
                    if img_dir:
                        dataset_name = os.path.basename(img_dir.rstrip('/'))
                        if dataset_name not in dataset_names:
                            dataset_names.append(dataset_name)
                dataset_str = f"_ds{'-'.join(dataset_names)}"
            else:
                # Single directory
                dataset_name = os.path.basename(args.image_dirs.rstrip('/'))
                dataset_str = f"_ds{dataset_name}"
        
        # Include model type, complexity, and approach information
        model_str = args.nn2_model_type

        # Handle neural models (new naming convention)
        print('***'*10)
        print('debug: args.nn2_model_type.startswith(neural_a)')
        if args.nn2_model_type.startswith('neural_a'):
            print('debug: neural_a')
            approach = args.nn2_model_type.split('_')[-1]
            complexity_map = {'simple': '1', 'medium': '2', 'complex': '3', 'complexer': '32'}
            complexity_num = complexity_map.get(args.nn2_model_complexity, '2')
            # For two-network approaches (A5, A6, A8), include all three networks info
            print(f'approach: {approach}')

            if approach in ['a5', 'a6', 'a8']:
                print('debug: approach in [5, 6]')
                # Shortened naming to avoid filename length issues
                nn1_info = f"nn1_{args.nn1_model_type.split('_')[-1]}"  # Just the approach part
                net1_info = f"n1-{args.nn2_net1_type[0]}-{args.nn2_net1_complexity[0]}-{args.nn2_net1_hidden_dim}"
                net2_info = f"n2-{args.nn2_net2_type[0]}-{args.nn2_net2_complexity[0]}-{args.nn2_net2_hidden_dim}"
                
                model_str = f"neural_{approach}_{args.nn2_loss_variant}_{nn1_info}_{net1_info}_{net2_info}"
            else:
                model_str = f"neural_{approach}_{complexity_num}-{args.nn2_hidden_dim}"

        # Handle parametric models (new naming convention)
        elif args.nn2_model_type.startswith('parametric_a'):
            approach = args.nn2_model_type.split('_')[-1]
            if args.nn2_model_complexity == 'fisheye':
                complexity_abbr = 'fish'
            else:
                complexity_map = {'simple': 'simp', 'medium': 'med', 'complex': 'comp', 'complexer': 'comp2'}
                complexity_abbr = complexity_map.get(args.nn2_model_complexity, 'med')
            model_str = f"parametric_{approach}_{complexity_abbr}"

        # Handle other model types (UNet, ResNet, etc.)
        else:
            model_str = args.nn2_model_type
        
        experiment_name = f"NN2_{proj_str}_{glass_str}_{model_str}_e{args.epochs}_lr{args.lr:.0e}_bs{args.batch_size}_{args.loss_type}{dataset_str}"
    
    # Add configuration flags
    if hasattr(args, 'norm') and args.norm:
        experiment_name += "_norm"
    if hasattr(args, 'use_amp') and args.use_amp:
        experiment_name += "_amp"
    if hasattr(args, 'nn2_use_zero_mask') and args.nn2_use_zero_mask:
        experiment_name += "_zeromask"
    if hasattr(args, 'dont_use_nn1') and args.dont_use_nn1:
        experiment_name += "_no_nn1"
    if hasattr(args, 'min_lr'):
        experiment_name += f"_minlr{args.min_lr:.0e}"
    if hasattr(args, 'weight_decay') and args.weight_decay != 0:
        experiment_name += f"_wd{args.weight_decay:.0e}"
    if hasattr(args, 'seed') and args.seed != DEFAULT_SEED:
        experiment_name += f"_seed{args.seed}"
    # Add image size to experiment name
    if hasattr(args, 'image_size') and args.image_size != 256:
        experiment_name += f"_img{args.image_size}"
    if hasattr(args, 'use_roi_extraction') and args.use_roi_extraction:
        experiment_name += f"_roi{args.roi_target_size[0]}x{args.roi_target_size[1]}"
    
    # Add grid_sample configuration
    experiment_name += f"_grid-lib-{args.grid_sample_mode}"
    
    if args.comment:
        experiment_name += f"_{args.comment}"
    
    print('***'*20)
    print('***'*20)
    print('***'*20)
    print(f'experiment_name: {experiment_name}') 
    return experiment_name


# Use PerceptualLoss from metrics.py
from metrics import PerceptualLoss


# IoU and F1 metrics removed as requested

# Use combined_loss from metrics.py
from metrics import combined_loss


def apply_color_distortion(image_tensor, distortion_type, strength, generator=None):
    """
    Apply a single color distortion to an image tensor.
    
    Args:
        image_tensor: [C, H, W] tensor
        distortion_type: Type of distortion to apply
        strength: Strength of the distortion
    
    Returns:
        Distorted image tensor
    """
    import torch
    import torch.nn.functional as F
    
    if distortion_type == 'brightness_contrast':
        # Brightness and contrast adjustment (combined)
        brightness_factor = 1.0 + (strength)  # brightness
        contrast_factor = 1.0 + (strength)    # contrast
        
        # Apply brightness
        image_tensor = image_tensor * brightness_factor
        
        # Apply contrast (center around 0.5)
        image_tensor = (image_tensor - 0.5) * contrast_factor + 0.5
        
    elif distortion_type == 'brightness_positive':
        # Non-uniform brightness increase only (all areas get brighter, but by different amounts)
        # strength range: [0.0, 1.0]
        # Typical range: [0.1, 0.5]
        h, w = image_tensor.shape[-2:]
        
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Create deterministic non-uniform pattern based on image content hash
        seed_val = (image_tensor.mean().item() * 1000) % 1.0
        seed_val2 = (image_tensor.std().item() * 1000) % 1.0
        
        freq_x = 1.0 + (seed_val * 2.0)  # 1-3 cycles
        freq_y = 1.0 + (seed_val2 * 2.0)
        phase_x = seed_val * 2 * torch.pi
        phase_y = seed_val2 * 2 * torch.pi
        
        # Combine sine waves and map to [0, 1] range (all positive)
        spatial_pattern = (torch.sin(2 * torch.pi * freq_x * xx + phase_x) + 
                          torch.sin(2 * torch.pi * freq_y * yy + phase_y)) / 2.0
        spatial_pattern = (spatial_pattern + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        
        brightness_strength = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        brightness_map = brightness_strength * spatial_pattern
        
        # Apply non-uniform brightness increase
        image_tensor = image_tensor + brightness_map
        
    elif distortion_type == 'brightness_negative':
        # Non-uniform brightness decrease only (all areas get darker, but by different amounts)
        # strength range: [0.0, 1.0] (positive values mean more darkening)
        # Typical range: [0.1, 0.5]
        h, w = image_tensor.shape[-2:]
        
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Create deterministic non-uniform pattern based on image content hash
        seed_val = (image_tensor.mean().item() * 1000) % 1.0
        seed_val2 = (image_tensor.std().item() * 1000) % 1.0
        
        freq_x = 1.0 + (seed_val * 2.0)  # 1-3 cycles
        freq_y = 1.0 + (seed_val2 * 2.0)
        phase_x = seed_val * 2 * torch.pi
        phase_y = seed_val2 * 2 * torch.pi
        
        # Combine sine waves and map to [0, 1] range (all positive, but will subtract)
        spatial_pattern = (torch.sin(2 * torch.pi * freq_x * xx + phase_x) + 
                          torch.sin(2 * torch.pi * freq_y * yy + phase_y)) / 2.0
        spatial_pattern = (spatial_pattern + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        
        brightness_strength = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        brightness_map = brightness_strength * spatial_pattern
        
        # Apply non-uniform brightness decrease (subtract)
        image_tensor = image_tensor - brightness_map
        
    elif distortion_type == 'brightness_both':
        # Non-uniform brightness with both increase and decrease
        # Positive strength creates variation (some areas brighter, some darker)
        # strength range: [0.0, 1.0]
        # Typical range: [0.1, 0.5]
        h, w = image_tensor.shape[-2:]
        
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Create deterministic non-uniform pattern based on image content hash
        seed_val = (image_tensor.mean().item() * 1000) % 1.0
        seed_val2 = (image_tensor.std().item() * 1000) % 1.0
        
        freq_x = 1.0 + (seed_val * 2.0)  # 1-3 cycles
        freq_y = 1.0 + (seed_val2 * 2.0)
        phase_x = seed_val * 2 * torch.pi
        phase_y = seed_val2 * 2 * torch.pi
        
        # Combine sine waves for smooth spatial variation in [-1, 1] range
        spatial_pattern = (torch.sin(2 * torch.pi * freq_x * xx + phase_x) + 
                          torch.sin(2 * torch.pi * freq_y * yy + phase_y)) / 2.0
        
        brightness_strength = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        brightness_map = brightness_strength * spatial_pattern  # Range: [-strength, +strength]
        
        # Apply non-uniform brightness (both increase and decrease)
        image_tensor = image_tensor + brightness_map.unsqueeze(0)
        
    elif distortion_type == 'brightness_positive_fixed':
        # Non-uniform brightness increase only with fixed pattern (not seed-based)
        # strength range: [0.0, 1.0]
        # Typical range: [0.1, 0.5]
        h, w = image_tensor.shape[-2:]
        
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Use fixed numbers instead of seed-based values
        freq_x = 2.0  # Fixed frequency
        freq_y = 2.5  # Fixed frequency
        phase_x = 0.5 * torch.pi  # Fixed phase
        phase_y = 1.0 * torch.pi  # Fixed phase
        
        # Combine sine waves and map to [0, 1] range (all positive)
        spatial_pattern = (torch.sin(2 * torch.pi * freq_x * xx + phase_x) + 
                          torch.sin(2 * torch.pi * freq_y * yy + phase_y)) / 2.0
        spatial_pattern = (spatial_pattern + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        
        brightness_strength = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        brightness_map = brightness_strength * spatial_pattern
        
        # Apply non-uniform brightness increase
        image_tensor = image_tensor + brightness_map
        
    elif distortion_type == 'brightness_negative_fixed':
        # Non-uniform brightness decrease only with fixed pattern (not seed-based)
        # strength range: [0.0, 1.0] (positive values mean more darkening)
        # Typical range: [0.1, 0.5]
        h, w = image_tensor.shape[-2:]
        
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Use fixed numbers instead of seed-based values
        freq_x = 2.0  # Fixed frequency
        freq_y = 2.5  # Fixed frequency
        phase_x = 0.5 * torch.pi  # Fixed phase
        phase_y = 1.0 * torch.pi  # Fixed phase
        
        # Combine sine waves and map to [0, 1] range (all positive, but will subtract)
        spatial_pattern = (torch.sin(2 * torch.pi * freq_x * xx + phase_x) + 
                          torch.sin(2 * torch.pi * freq_y * yy + phase_y)) / 2.0
        spatial_pattern = (spatial_pattern + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        
        brightness_strength = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        brightness_map = brightness_strength * spatial_pattern
        
        # Apply non-uniform brightness decrease (subtract)
        image_tensor = image_tensor - brightness_map
        
    elif distortion_type == 'brightness_both_fixed':
        # Non-uniform brightness with both increase and decrease, using fixed pattern (not seed-based)
        # Positive strength creates variation (some areas brighter, some darker)
        # strength range: [0.0, 1.0]
        # Typical range: [0.1, 0.5]
        h, w = image_tensor.shape[-2:]
        
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Use fixed numbers instead of seed-based values
        freq_x = 2.0  # Fixed frequency
        freq_y = 2.5  # Fixed frequency
        phase_x = 0.5 * torch.pi  # Fixed phase
        phase_y = 1.0 * torch.pi  # Fixed phase
        
        # Combine sine waves for smooth spatial variation in [-1, 1] range
        spatial_pattern = (torch.sin(2 * torch.pi * freq_x * xx + phase_x) + 
                          torch.sin(2 * torch.pi * freq_y * yy + phase_y)) / 2.0
        
        brightness_strength = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        brightness_map = brightness_strength * spatial_pattern  # Range: [-strength, +strength]
        
        # Apply non-uniform brightness (both increase and decrease)
        image_tensor = image_tensor + brightness_map.unsqueeze(0)
        
    elif distortion_type == 'brightness_uniform':
        # Uniform brightness increase/decrease over the entire image
        # strength range: [-1.0, 1.0]
        # Positive values increase brightness, negative values decrease brightness
        # Typical range: [-0.5, 0.5]
        brightness_strength = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), -1.0, 1.0)
        
        # Apply uniform brightness change (add to all pixels uniformly)
        image_tensor = image_tensor + brightness_strength
        
    elif distortion_type == 'contrast':
        # Non-uniform contrast adjustment
        # Positive strength increases contrast, negative decreases it
        # strength range: [-1.0, 1.0]
        # Typical range: [-0.5, 0.5]
        h, w = image_tensor.shape[-2:]
        
        # Create spatial variation pattern using smooth random field
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Create deterministic non-uniform pattern based on image content hash
        # Use mean and std of image as seed for reproducibility
        seed_val = (image_tensor.mean().item() * 1000) % 1.0
        seed_val2 = (image_tensor.std().item() * 1000) % 1.0
        
        # Create non-uniform pattern with deterministic phase and frequency
        freq_x = 1.0 + (seed_val * 2.0)  # 1-3 cycles
        freq_y = 1.0 + (seed_val2 * 2.0)
        phase_x = seed_val * 2 * torch.pi
        phase_y = seed_val2 * 2 * torch.pi
        
        # Combine sine waves for smooth spatial variation
        spatial_pattern = (torch.sin(2 * torch.pi * freq_x * xx + phase_x) + 
                          torch.sin(2 * torch.pi * freq_y * yy + phase_y)) / 2.0
        
        # Scale pattern by strength (pattern is in [-1, 1])
        contrast_strength = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), -1.0, 1.0)
        # Contrast factor varies spatially: 1.0 + strength * pattern * 0.5
        contrast_map = 1.0 + contrast_strength * spatial_pattern
        contrast_map = torch.clamp(contrast_map, 0.2, 2.0)  # Reasonable range
        
        # Apply non-uniform contrast (center around 0.5 - mid-gray)
        image_tensor = (image_tensor - 0.5) * contrast_map.unsqueeze(0) + 0.5
        
    elif distortion_type == 'contrast_fixed':
        # Non-uniform contrast adjustment with fixed pattern (not seed-based)
        # Positive strength increases contrast, negative decreases it
        # strength range: [-1.0, 1.0]
        # Typical range: [-0.5, 0.5]
        h, w = image_tensor.shape[-2:]
        
        # Create spatial variation pattern using smooth random field
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Use fixed numbers instead of seed-based values
        freq_x = 2.0  # Fixed frequency
        freq_y = 2.5  # Fixed frequency
        phase_x = 0.5 * torch.pi  # Fixed phase
        phase_y = 1.0 * torch.pi  # Fixed phase
        
        # Combine sine waves for smooth spatial variation
        spatial_pattern = (torch.sin(2 * torch.pi * freq_x * xx + phase_x) + 
                          torch.sin(2 * torch.pi * freq_y * yy + phase_y)) / 2.0
        
        # Scale pattern by strength (pattern is in [-1, 1])
        contrast_strength = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), -1.0, 1.0)
        # Contrast factor varies spatially: 1.0 + strength * pattern * 0.5
        contrast_map = 1.0 + contrast_strength * spatial_pattern
        contrast_map = torch.clamp(contrast_map, 0.2, 2.0)  # Reasonable range
        
        # Apply non-uniform contrast (center around 0.5 - mid-gray)
        image_tensor = (image_tensor - 0.5) * contrast_map.unsqueeze(0) + 0.5
        
    elif distortion_type == 'gamma':
        # Gamma correction
        gamma = 1.0 + (strength)  # 1.0 to 1.8
        image_tensor = torch.pow(image_tensor, 1.0 / gamma)
        
    elif distortion_type == 'vignetting':
        # Simple vignetting effect
        h, w = image_tensor.shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=image_tensor.device),
                             torch.linspace(-1, 1, w, device=image_tensor.device), indexing='ij')
        vignette = 1.0 - (x**2 + y**2) * strength
        vignette = torch.clamp(vignette, 0.3, 1.0)
        image_tensor = image_tensor * vignette.unsqueeze(0)
        
    elif distortion_type == 'blur':
        # Blur: differentiable Gaussian blur via depthwise conv
        h, w = image_tensor.shape[-2:]
        # Map strength -> sigma in pixels (soft cap at ≈ 3.0 px)
        sigma = 0.1 + 2.9 * torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        # Kernel size from sigma (ensure odd)
        ksize = int((sigma * 3.0).item()) * 2 + 1
        ksize = max(3, min(ksize, 31))
        coords = torch.arange(ksize, device=image_tensor.device, dtype=image_tensor.dtype) - (ksize - 1) / 2
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
        gauss_1d = gauss_1d / (gauss_1d.sum() + 1e-8)
        # Separable convolution: first horizontal, then vertical
        gauss_x = gauss_1d.view(1, 1, 1, ksize)
        gauss_y = gauss_1d.view(1, 1, ksize, 1)
        x = image_tensor.unsqueeze(0)
        # Depthwise conv by grouping
        C = x.shape[1]
        kernel_x = gauss_x.expand(C, 1, 1, ksize)
        kernel_y = gauss_y.expand(C, 1, ksize, 1)
        # Apply horizontal blur
        x = F.conv2d(x, kernel_x, groups=C, padding=(0, ksize//2))
        # Apply vertical blur
        x = F.conv2d(x, kernel_y, groups=C, padding=(ksize//2, 0))
        image_tensor = x.squeeze(0)
        
    elif distortion_type in ['lsnu', 'light_source_non_uniformity']:
        # LSNU (Light Source Non-Uniformity): multiplicative radial falloff
        h, w = image_tensor.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(-1, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        r2 = x * x + y * y
        amp = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 0.3)
        falloff = 1.0 - amp * r2
        falloff = torch.clamp(falloff, 0.7, 1.0)
        image_tensor = image_tensor * falloff.unsqueeze(0)
        
    elif distortion_type in ['spatiotemporal_dithering', 'dither']:
        # Spatiotemporal dithering: additive noise
        # Allowed strength range: [0.0, 0.1]
        # Suggested realistic range: 0.005 – 0.02
        amp = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 0.1)
        noise = torch.randn(image_tensor.shape, generator=generator, device=image_tensor.device, dtype=image_tensor.dtype) * amp
        image_tensor = image_tensor + noise
        
    elif distortion_type == 'color_balance':
        # Color balance adjustment
        balance_factor = 1.0 + (strength)
        
        # Apply to each channel differently (avoid in-place operations)
        out_channels = []
        for c in range(image_tensor.shape[0]):
            channel_factor = balance_factor * (1.0 + (c - 1) * 0.1)  # Slight variation per channel
            out_channels.append(image_tensor[c] * channel_factor)
        image_tensor = torch.stack(out_channels, dim=0)
            
    elif distortion_type in ['elevated_black_level', 'black_level']:
        # Elevated black level (pedestal) → contrast loss
        # y = (x + p) / (1 + p), p ≥ 0
        p = 0.4 * torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 0.5) / 2.0
        image_tensor = (image_tensor + p) / (1.0 + p)
        
    elif distortion_type == 'contrast_loss':
        # Global contrast compression around mid-gray
        # y = (x - 0.5) * (1 - c) + 0.5
        # Allowed strength range: [0.0, 1.0] mapped to c ∈ [0, 0.6]
        # Suggested realistic range: 0.1 – 0.3 (c ≈ 0.06 – 0.18)
        c = 0.6 * torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        image_tensor = (image_tensor - 0.5) * (1.0 - c) + 0.5
        
    elif distortion_type == 'contrast_uniform':
        # ============================================================================
        # Uniform global contrast adjustment around mid-gray (0.5)
        # ============================================================================
        # Simulates: Simple global contrast change, without spatial variation.
        #
        # Strength range:
        #   - strength ∈ [-1.0, 1.0]
        #     * strength > 0 → higher contrast
        #     * strength < 0 → lower contrast
        #
        # Mapping:
        #   - contrast_factor = 1 + strength
        #     * strength = 0   → factor = 1.0  (no change)
        #     * strength = 0.5 → factor = 1.5  (moderate increase)
        #     * strength = -0.5→ factor = 0.5  (moderate decrease)
        #
        # Formula:
        #   y = (x - 0.5) * contrast_factor + 0.5
        # ============================================================================
        contrast_strength = torch.clamp(
            torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype),
            -1.0,
            1.0
        )
        contrast_factor = 1.0 + contrast_strength
        image_tensor = (image_tensor - 0.5) * contrast_factor + 0.5
        
    elif distortion_type in ['color_fringes', 'chromatic_fringes']:
        # Color fringes (simple chromatic aberration): per-channel slight radial scaling
        import torch
        import torch.nn.functional as F
        h, w = image_tensor.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(-1, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        r2 = xx * xx + yy * yy
        base = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        # Map to small scale offsets (±0.5%)
        s = 1.0 + 0.050 * base
        scales = torch.tensor([s, 1.0, 2.0 - s], device=image_tensor.device, dtype=image_tensor.dtype)
        out_channels = []
        for c_idx in range(3):
            sx = sy = scales[c_idx]
            x_d = xx * sx
            y_d = yy * sy
            grid = torch.stack([x_d, y_d], dim=-1).unsqueeze(0)
            ch = image_tensor[c_idx:c_idx+1, :, :].unsqueeze(0)
            warped = F.grid_sample(ch, grid, mode='bilinear', padding_mode='border', align_corners=True)
            out_channels.append(warped.squeeze(0))
        image_tensor = torch.cat(out_channels, dim=0)
        
    elif distortion_type in ['wgnu', 'waveguide_non_uniformity']:
        # WGNU (WaveGuide Non-Uniformity): multiplicative stripe/ripple pattern across the guide
        h, w = image_tensor.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        amp = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        # Frequency increases mildly with strength (1–6 cycles)
        freq = 1.0 + 5.0 * amp
        # phase = torch.rand(1, device=image_tensor.device, dtype=image_tensor.dtype) * 2 * torch.pi
        phase = 0.5 * 2 * torch.pi

        stripes = torch.sin(2 * torch.pi * freq * xx + phase)
        stripes = stripes / (stripes.abs().amax() + 1e-8)
        # Increase gain multiplier from 0.2 to 0.4 for stronger effect (was ±0.1, now ±0.2 at max amp)
        gain = 1.0 + 0.1 * amp * stripes  # cap amplitude inside ±0.2 at max amp
        image_tensor = image_tensor * gain.unsqueeze(0)
    
    elif distortion_type == 'vdp_spatiotemporal_dithering':
        # ============================================================================
        # Spatiotemporal Dithering
        # ============================================================================
        # Simulates: Bit depth reduction with blue noise mask dithering per color channel
        # Context: Display systems with constrained bit depths (e.g., reduced color channels)
        #          Dithering is performed spatially and temporally to improve image quality
        # 
        # Paper Specification:
        #   - Level 1: 6-bit color depth (64 levels per channel)
        #   - Level 2: 5-bit color depth (32 levels per channel)
        #   - Level 3: 4-bit color depth (16 levels per channel)
        # 
        # Strength Mapping:
        #   - 0.0-0.33 → 6 bits (Level 1)
        #   - 0.33-0.67 → 5 bits (Level 2)
        #   - 0.67-1.0 → 4 bits (Level 3)
        # 
        # Recommended Values:
        #   - strength=0.2 for Level 1 (subtle quantization)
        #   - strength=0.5 for Level 2 (medium quantization)
        #   - strength=0.85 for Level 3 (strong quantization, visible banding)
        # ============================================================================
        
        # Determine bit depth based on discrete levels
        strength_clamped = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        if strength_clamped < 0.33:
            bit_depth = 6  # Level 1
        elif strength_clamped < 0.67:
            bit_depth = 5  # Level 2
        else:
            bit_depth = 4  # Level 3
        
        levels = 2 ** bit_depth
        
        # Generate blue noise approximation
        # Note: True blue noise requires pre-computed masks; we approximate with high-pass filtered noise
        h, w = image_tensor.shape[-2:]
        white_noise = torch.rand(image_tensor.shape, generator=generator, device=image_tensor.device, dtype=image_tensor.dtype) - 0.5
        
        # Simple blue noise approximation: add spatial high-frequency component
        # Creates better spatial distribution than pure white noise
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.arange(w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        # Add high-frequency modulation to prevent low-frequency clustering
        blue_noise_approx = white_noise + 0.1 * torch.sin(2 * torch.pi * (x_grid + y_grid) / 8.0).unsqueeze(0)
        blue_noise_approx = blue_noise_approx - blue_noise_approx.mean()  # Zero mean
        
        # Apply dithering
        dither = blue_noise_approx / levels
        
        # Quantize with dithering
        image_tensor = image_tensor + dither
        image_tensor = torch.round(image_tensor * (levels - 1)) / (levels - 1)
        
    elif distortion_type == 'vdp_lsnu':
        # ============================================================================
        # LSNU (Light Source Non-Uniformity)
        # ============================================================================
        # Simulates: High-frequency spatial nonuniformity in μOLED and μLED displays
        # Context: Each pixel consists of a separate light source module which may vary
        #          in its light output, creating spatial and color artifacts
        # 
        # Paper Specification:
        #   - Each pixel intensity is randomly modulated per color channel
        #   - Modulation uses Gaussian distribution with specific std dev
        #   - Values are "up to X%" which means 2σ (twice the standard deviation)
        #   - Level 1: ±12% (std = 6%)
        #   - Level 2: ±16% (std = 8%)
        #   - Level 3: ±18% (std = 9%)
        #   - Modulation ratios are CONSTANT for each pixel across all frames
        #   - Simulations done in linear color space
        # 
        # Strength Mapping:
        #   - 0.0-0.33 → std=6% (Level 1)
        #   - 0.33-0.67 → std=8% (Level 2)
        #   - 0.67-1.0 → std=9% (Level 3)
        # 
        # Recommended Values:
        #   - strength=0.2 for Level 1 (subtle grainy texture)
        #   - strength=0.5 for Level 2 (medium pixel variation)
        #   - strength=0.85 for Level 3 (strong grainy appearance)
        # ============================================================================
        
        # Determine std based on discrete levels
        strength_clamped = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        if strength_clamped < 0.33:
            std = 0.06  # Level 1: ±12% at 2σ
        elif strength_clamped < 0.67:
            std = 0.08  # Level 2: ±16% at 2σ
        else:
            std = 0.09  # Level 3: ±18% at 2σ
        
        # Use fixed seed to generate consistent per-pixel noise
        # This ensures the modulation is constant across frames (per paper spec)
        generator = torch.Generator(device=image_tensor.device)
        generator.manual_seed(42)  # Fixed seed for reproducibility
        
        # Generate per-pixel, per-channel random modulation
        modulation = torch.randn(image_tensor.shape, generator=generator, 
                                device=image_tensor.device, dtype=image_tensor.dtype) * std
        
        # Clamp to ±2*std (paper spec: "up to 2*12%, 2*16%, 2*18%")
        modulation = torch.clamp(modulation, -4*std, 4*std)
        
        # Apply multiplicatively: new_intensity = old_intensity * (1 + modulation)
        image_tensor = image_tensor * (1.0 + modulation)
        
    elif distortion_type == 'vdp_blur':
        # ============================================================================
        # Blur (MTF Degradation)
        # ============================================================================
        # Simulates: Optical blur from lenses and waveguides in AR/VR displays
        # Context: Optical components degrade the system Modulation Transfer Function (MTF)
        #          resulting in blurry images
        # 
        # Paper Specification:
        #   - Representative Point Spread Function (PSF) applied with various severities
        #   - Same PSF applied to all color channels
        #   - Lateral chromatic aberration for Levels 2 and 3:
        #     * Green channel frame shrunk by 2 pixels
        #   - See Fig. 13 in paper for MTF curves at different levels
        # 
        # Strength Mapping:
        #   - 0.0 → minimal blur
        #   - 0.5 → medium blur
        #   - 1.0 → strong blur
        #   - Chromatic aberration applies when strength > 0.33
        # 
        # Recommended Values:
        #   - strength=0.3 for Level 1 (slight softening)
        #   - strength=0.6 for Level 2 (noticeable blur + chromatic aberration)
        #   - strength=1.0 for Level 3 (strong blur + chromatic aberration)
        # ============================================================================
        
        h, w = image_tensor.shape[-2:]
        
        # Map strength to sigma in pixels for Gaussian blur
        sigma = 0.5 + 2.5 * torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        
        # Kernel size from sigma (ensure odd for symmetric kernel)
        ksize = int((sigma * 3.0).item()) * 2 + 1
        ksize = max(3, min(ksize, 31))  # Clamp to reasonable range
        coords = torch.arange(ksize, device=image_tensor.device, dtype=image_tensor.dtype) - (ksize - 1) / 2
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
        gauss_1d = gauss_1d / (gauss_1d.sum() + 1e-8)  # Normalize
        
        # Separable convolution (more efficient than 2D)
        gauss_x = gauss_1d.view(1, 1, 1, ksize)
        gauss_y = gauss_1d.view(1, 1, ksize, 1)
        x = image_tensor.unsqueeze(0)
        
        # Depthwise convolution (apply same kernel to each channel independently)
        C = x.shape[1]
        kernel_x = gauss_x.expand(C, 1, 1, ksize)
        kernel_y = gauss_y.expand(C, 1, ksize, 1)
        
        # Apply horizontal then vertical blur
        x = F.conv2d(x, kernel_x, groups=C, padding=(0, ksize//2))
        x = F.conv2d(x, kernel_y, groups=C, padding=(ksize//2, 0))
        image_tensor = x.squeeze(0)
        
        # Lateral chromatic aberration for levels 2 and 3 (strength > 0.33)
        # Paper spec: Shrink green channel by 2 pixels
        if strength > 0.33 and image_tensor.shape[0] >= 3:
            shrink_pixels = 2
            green = image_tensor[1:2, :, :]
            scale_factor = max(0.5, (h - 2 * shrink_pixels) / h)
            green_small = F.interpolate(green.unsqueeze(0), scale_factor=scale_factor, 
                                        mode='bilinear', align_corners=False)
            
            # Pad back to original size
            pad = shrink_pixels
            green_padded = F.pad(green_small, (pad, pad, pad, pad), mode='replicate')
            image_tensor[1:2, :, :] = green_padded.squeeze(0)
            
    elif distortion_type == 'vdp_reduced_contrast':
        # ============================================================================
        # Reduced Contrast
        # ============================================================================
        # Simulates: Poor contrast in optical see-through displays (especially in bright environments)
        # Context: Challenging to achieve good contrast in AR displays
        # 
        # Paper Specification:
        #   - Simulated in linear space
        #   - Contrast reduced by increasing minimum value while maintaining maximum at 1.0
        #   - Contrast defined as max/min
        #   - Level 1: 100:1 contrast (black_level = 0.01)
        #   - Level 2: 50:1 contrast (black_level = 0.02)
        #   - Level 3: 30:1 contrast (black_level = 0.033)
        # 
        # Strength Mapping:
        #   - 0.0-0.33 → 100:1 (Level 1)
        #   - 0.33-0.67 → 50:1 (Level 2)
        #   - 0.67-1.0 → 30:1 (Level 3)
        # 
        # Recommended Values:
        #   - strength=0.2 for Level 1 (slight contrast loss)
        #   - strength=0.5 for Level 2 (medium contrast loss, washed out)
        #   - strength=0.85 for Level 3 (severe contrast loss, very washed out)
        # ============================================================================
        
        # Determine contrast ratio based on discrete levels
        strength_clamped = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        if strength_clamped < 0.33:
            contrast = 100.0  # Level 1
        elif strength_clamped < 0.67:
            contrast = 50.0   # Level 2
        else:
            contrast = 30.0   # Level 3
        
        black_level = 1.0 / contrast
        
        # Map [0, 1] → [black_level, 1]
        # This lifts the blacks while keeping whites at 1.0
        image_tensor = image_tensor * (1.0 - black_level) + black_level
        
    elif distortion_type == 'vdp_wgnu':
        # ============================================================================
        # WGNU (Waveguide Non-Uniformity)
        # ============================================================================
        # Simulates: Spatially-varying color nonuniformity in diffractive waveguide AR displays
        # Context: Microsoft HoloLens, Magic Leap, Snap Spectacles exhibit this artifact
        #          Nonuniformity is low-frequency (<1 cycle per degree)
        #          Heavily dependent on pupil position within the eye box
        #          During VOR (vestibulo-ocular reflex) movements, users notice color shifting
        # 
        # Paper Specification:
        #   - Two nonuniformity patterns obtained from empirical waveguide transmission data
        #   - Base variation maps normalized across channels
        #   - Amplitude multipliers:
        #     * Level 1: 0.15 (transitions 1 time over video)
        #     * Level 2: 0.3 (transitions 4 times over video)
        #     * Level 3: 0.4 (transitions 7 times over video)
        #   - Pattern transitions simulate changes with eye position
        #   - See Fig. 14 in paper showing temporal variation
        # 
        # Strength Mapping:
        #   - 0.0-0.33 → amp=0.15 (Level 1)
        #   - 0.33-0.67 → amp=0.3 (Level 2)
        #   - 0.67-1.0 → amp=0.4 (Level 3)
        # 
        # Recommended Values:
        #   - strength=0.2 for Level 1 (subtle color nonuniformity)
        #   - strength=0.5 for Level 2 (medium color shifts)
        #   - strength=0.85 for Level 3 (strong waveguide artifacts)
        # 
        # Note: Paper uses empirical data; we approximate with synthetic low-frequency patterns
        #       For video: temporal transitions would be implemented
        #       For images: we show a fixed blend of the two patterns
        # ============================================================================
        
        h, w = image_tensor.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Determine amplitude based on discrete levels
        strength_clamped = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        if strength_clamped < 0.33:
            amp = 0.15  # Level 1
        elif strength_clamped < 0.67:
            amp = 0.3   # Level 2
        else:
            amp = 0.4   # Level 3
        
        # Pattern 1: Low-frequency horizontal variation (approximates empirical waveguide pattern)
        # Multiple sine waves at different frequencies create complex spatial variation
        freq1_base = 1.2
        pattern1 = (torch.sin(2 * torch.pi * freq1_base * xx) * 0.5 +
                    torch.sin(2 * torch.pi * freq1_base * 1.7 * yy) * 0.3 +
                    torch.sin(2 * torch.pi * freq1_base * 0.8 * (xx + yy)) * 0.2)
        pattern1 = pattern1 / (pattern1.abs().amax() + 1e-8)
        
        # Pattern 2: Low-frequency diagonal variation (different empirical approximation)
        freq2_base = 0.9
        pattern2 = (torch.sin(2 * torch.pi * freq2_base * (xx - yy)) * 0.4 +
                    torch.sin(2 * torch.pi * freq2_base * 1.5 * xx) * 0.4 +
                    torch.sin(2 * torch.pi * freq2_base * 1.3 * yy) * 0.2)
        pattern2 = pattern2 / (pattern2.abs().amax() + 1e-8)
        
        # Temporal interpolation
        # For video: t = (frame_idx % (total_frames / num_transitions)) / (total_frames / num_transitions)
        # For images: use fixed seed for consistent pattern blend
        # generator = torch.Generator(device=image_tensor.device)
        # generator.manual_seed(123)  # Fixed seed for reproducibility
        # t = torch.rand(1, generator=generator, device=image_tensor.device, dtype=image_tensor.dtype).item()
        t = 0.5
        # Smooth cosine interpolation between patterns (simulates VOR movement)
        blend = (1 - torch.cos(torch.tensor(t * torch.pi))) / 2
        blended_pattern = pattern1 * (1 - blend) + pattern2 * blend
        
        # Apply per-channel with color nonuniformity (characteristic of waveguides)
        out_channels = []
        for c_idx in range(image_tensor.shape[0]):
            # Per-channel phase variation creates color artifacts
            channel_offset = (c_idx - 1) * 0.15
            gain = 1.0 + amp * blended_pattern * (1.0 + channel_offset)
            out_channels.append(image_tensor[c_idx] * gain)
        image_tensor = torch.stack(out_channels, dim=0)
        
    elif distortion_type == 'vdp_wgnu_hashed':
        # ============================================================================
        # WGNU (Waveguide Non-Uniformity) - Hashed Version
        # ============================================================================
        # Same as vdp_wgnu but uses deterministic seeding based on image content hash
        # instead of fixed patterns. This creates image-specific patterns while remaining
        # reproducible for the same image.
        # ============================================================================
        
        h, w = image_tensor.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        # Determine amplitude based on discrete levels (same as vdp_wgnu)
        strength_clamped = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        if strength_clamped < 0.33:
            amp = 0.15  # Level 1
        elif strength_clamped < 0.67:
            amp = 0.3   # Level 2
        else:
            amp = 0.4   # Level 3
        
        # Create deterministic seeds from image content
        seed_val = (image_tensor.mean().item() * 1000) % 1.0
        seed_val2 = (image_tensor.std().item() * 1000) % 1.0
        seed_val3 = (image_tensor.max().item() * 1000) % 1.0
        
        # Pattern 1: Low-frequency horizontal variation with hashed frequencies
        freq1_base = 0.8 + seed_val * 0.8  # 0.8-1.6 cycles
        pattern1 = (torch.sin(2 * torch.pi * freq1_base * xx + seed_val * 2 * torch.pi) * 0.5 +
                    torch.sin(2 * torch.pi * freq1_base * 1.7 * yy + seed_val2 * 2 * torch.pi) * 0.3 +
                    torch.sin(2 * torch.pi * freq1_base * 0.8 * (xx + yy) + seed_val3 * 2 * torch.pi) * 0.2)
        pattern1 = pattern1 / (pattern1.abs().amax() + 1e-8)
        
        # Pattern 2: Low-frequency diagonal variation with hashed frequencies
        freq2_base = 0.7 + seed_val2 * 0.6  # 0.7-1.3 cycles
        pattern2 = (torch.sin(2 * torch.pi * freq2_base * (xx - yy) + seed_val2 * 2 * torch.pi) * 0.4 +
                    torch.sin(2 * torch.pi * freq2_base * 1.5 * xx + seed_val3 * 2 * torch.pi) * 0.4 +
                    torch.sin(2 * torch.pi * freq2_base * 1.3 * yy + seed_val * 2 * torch.pi) * 0.2)
        pattern2 = pattern2 / (pattern2.abs().amax() + 1e-8)
        
        # Temporal interpolation using deterministic blend
        t = seed_val3  # Use third seed for blend factor
        blend = (1 - torch.cos(torch.tensor(t * torch.pi, device=image_tensor.device, dtype=image_tensor.dtype))) / 2
        blended_pattern = pattern1 * (1 - blend) + pattern2 * blend
        
        # Apply per-channel with color nonuniformity (characteristic of waveguides)
        out_channels = []
        for c_idx in range(image_tensor.shape[0]):
            # Per-channel phase variation creates color artifacts
            channel_offset = (c_idx - 1) * 0.15
            gain = 1.0 + amp * blended_pattern * (1.0 + channel_offset)
            out_channels.append(image_tensor[c_idx] * gain)
        image_tensor = torch.stack(out_channels, dim=0)
        
    elif distortion_type == 'vdp_dce':
        # ============================================================================
        # DCE (Dynamic Correction Error)
        # ============================================================================
        # Simulates: Eye tracking inaccuracy causing imperfect waveguide correction
        # Context: Dynamic correction algorithms invert waveguide distortion based on gaze
        #          Inaccurate eye tracking leads to wrong pupil position estimates
        #          Results in temporal color artifacts (imperfect or incorrect correction)
        # 
        # Paper Specification:
        #   - Precision error randomly generated per frame from Gaussian distribution
        #   - Standard deviations:
        #     * Level 1: 0.27° (aligned with existing VR headsets)
        #     * Level 2: 0.39°
        #     * Level 3: 0.58°
        #   - Errors cause temporal color artifacts
        # 
        # Strength Mapping:
        #   - 0.0-0.33 → 0.27° std (Level 1)
        #   - 0.33-0.67 → 0.39° std (Level 2)
        #   - 0.67-1.0 → 0.58° std (Level 3)
        # 
        # Recommended Values:
        #   - strength=0.2 for Level 1 (subtle correction errors)
        #   - strength=0.5 for Level 2 (medium tracking errors)
        #   - strength=0.85 for Level 3 (significant tracking errors, visible color shifts)
        # ============================================================================
        
        # Determine angle std based on discrete levels
        strength_clamped = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        if strength_clamped < 0.33:
            angle_std = 0.27  # Level 1
        elif strength_clamped < 0.67:
            angle_std = 0.39  # Level 2
        else:
            angle_std = 0.58  # Level 3
        
        # Convert angle error to pixel shift (assuming ~60 pixels per degree for VR/AR)
        ppd = 60.0  # pixels per degree (typical for VR/AR displays)
        pixel_std = angle_std * ppd
        
        h, w = image_tensor.shape[-2:]
        
        # Generate random eye tracking error per frame
        shift_x = torch.randn(1, generator=generator, device=image_tensor.device, dtype=image_tensor.dtype) * pixel_std
        shift_y = torch.randn(1, generator=generator, device=image_tensor.device, dtype=image_tensor.dtype) * pixel_std
        
        # Create base waveguide nonuniformity pattern (similar to WGNU)
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
            torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
            indexing='ij'
        )
        
        freq = 2.5
        phase = torch.rand(1, generator=generator, device=image_tensor.device, dtype=image_tensor.dtype) * 2 * torch.pi
        pattern = torch.sin(2 * torch.pi * freq * xx + phase) + 0.5 * torch.sin(2 * torch.pi * freq * 0.7 * yy)
        pattern = pattern / (pattern.abs().amax() + 1e-8)
        
        # Apply tracking error as pattern shift
        shift_x_norm = shift_x / w * 2
        shift_y_norm = shift_y / h * 2
        
        theta = torch.tensor([[1, 0, shift_x_norm.item()], 
                            [0, 1, shift_y_norm.item()]], 
                            device=image_tensor.device, dtype=image_tensor.dtype)
        grid = F.affine_grid(theta.unsqueeze(0), [1, 1, h, w], align_corners=False)
        
        pattern_shifted = F.grid_sample(pattern.unsqueeze(0).unsqueeze(0), grid, 
                                    mode='bilinear', padding_mode='border', align_corners=False)
        pattern_shifted = pattern_shifted.squeeze()
        
        # Correction error = difference between intended and actual correction
        # This represents the imperfect correction due to eye tracking inaccuracy
        correction_error = pattern - pattern_shifted
        amp = 0.35 * (angle_std / 0.58)  # Scale amplitude with error magnitude
        
        # Apply with per-channel color variation
        out_channels = []
        for c_idx in range(image_tensor.shape[0]):
            channel_factor = 1.0 + (c_idx - 1) * 0.18
            gain = 1.0 + amp * correction_error * channel_factor
            out_channels.append(image_tensor[c_idx] * gain)
        image_tensor = torch.stack(out_channels, dim=0)
        
    elif distortion_type == 'vdp_color_fringes':
        # ============================================================================
        # Color Fringes
        # ============================================================================
        # Simulates: Optical misalignment of color channels in head-mounted displays
        # Context: Caused by mechanical shifts of optical components, imperfect fabrication,
        #          and thermal loads
        # 
        # Paper Specification:
        #   - First-order approximation: global 2D shift per color channel
        #   - Green channel is reference (no shift)
        #   - Level 1:
        #     * Red:  [0, 0.5] pixels in [x, y]
        #     * Green: [0, 0] pixels (reference)
        #     * Blue: [-0.5, 0] pixels
        #   - Level 2:
        #     * Red:  [1, 0.5] pixels
        #     * Green: [0, 0] pixels
        #     * Blue: [0.5, -1] pixels
        #   - Level 3:
        #     * Red:  [1.5, 1] pixels
        #     * Green: [0, 0] pixels
        #     * Blue: [1, -1] pixels
        # 
        # Strength Mapping:
        #   - 0.0-0.33 → Level 1 shifts
        #   - 0.33-0.67 → Level 2 shifts
        #   - 0.67-1.0 → Level 3 shifts
        # 
        # Recommended Values:
        #   - strength=0.2 for Level 1 (subtle color fringes on edges)
        #   - strength=0.5 for Level 2 (noticeable color separation)
        #   - strength=0.85 for Level 3 (strong chromatic misalignment)
        # ============================================================================
        
        # Determine shifts based on discrete levels
        strength_clamped = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        if strength_clamped < 0.33:
            # Level 1
            shift_r = torch.tensor([0.0, 0.5], device=image_tensor.device, dtype=image_tensor.dtype)
            shift_g = torch.tensor([0.0, 0.0], device=image_tensor.device, dtype=image_tensor.dtype)
            shift_b = torch.tensor([-0.5, 0.0], device=image_tensor.device, dtype=image_tensor.dtype)
        elif strength_clamped < 0.67:
            # Level 2
            shift_r = torch.tensor([1.0, 0.5], device=image_tensor.device, dtype=image_tensor.dtype)
            shift_g = torch.tensor([0.0, 0.0], device=image_tensor.device, dtype=image_tensor.dtype)
            shift_b = torch.tensor([0.5, -1.0], device=image_tensor.device, dtype=image_tensor.dtype)
        else:
            # Level 3
            shift_r = torch.tensor([1.5, 1.0], device=image_tensor.device, dtype=image_tensor.dtype)
            shift_g = torch.tensor([0.0, 0.0], device=image_tensor.device, dtype=image_tensor.dtype)
            shift_b = torch.tensor([1.0, -1.0], device=image_tensor.device, dtype=image_tensor.dtype)
        
        h, w = image_tensor.shape[-2:]
        out_channels = []
        
        for c_idx, shift in enumerate([shift_r, shift_g, shift_b]):
            # Normalize shifts to [-1, 1] grid space for grid_sample
            shift_x = -2.0 * shift[0] / w
            shift_y = -2.0 * shift[1] / h
            
            # Create affine transformation matrix
            theta = torch.tensor([[1, 0, shift_x], 
                                [0, 1, shift_y]], 
                                device=image_tensor.device, dtype=image_tensor.dtype)
            grid = F.affine_grid(theta.unsqueeze(0), [1, 1, h, w], align_corners=False)
            
            # Apply translation to channel
            ch = image_tensor[c_idx:c_idx+1, :, :].unsqueeze(0)
            shifted = F.grid_sample(ch, grid, mode='bilinear', padding_mode='border', align_corners=False)
            out_channels.append(shifted.squeeze(0))
        
        image_tensor = torch.cat(out_channels, dim=0)
        
    elif distortion_type == 'vdp_chroma_subsampling':
        # ============================================================================
        # Chroma Subsampling
        # ============================================================================
        # Simulates: Reduced spatial resolution of chroma channels
        # Context: Common in video compression and signal transmission
        #          Takes advantage of reduced visual sensitivity to high-frequency chromatic content
        # 
        # Paper Specification:
        #   - Spatial resolutions of chroma channels reduced
        #   - Maintains overall image quality by preserving luma
        #   - Level 1: 1/4 reduction (4:1 subsampling)
        #   - Level 2: 1/8 reduction (8:1 subsampling)
        #   - Level 3: 1/12 reduction (12:1 subsampling)
        # 
        # Strength Mapping:
        #   - 0.0-0.33 → 1/4 reduction (Level 1)
        #   - 0.33-0.67 → 1/8 reduction (Level 2)
        #   - 0.67-1.0 → 1/12 reduction (Level 3)
        # 
        # Recommended Values:
        #   - strength=0.2 for Level 1 (slight chroma blockiness)
        #   - strength=0.5 for Level 2 (medium chroma artifacts)
        #   - strength=0.85 for Level 3 (strong chroma blockiness)
        # 
        # Note: In RGB, we treat R as luma (full resolution) and G, B as chroma (subsampled)
        #       In true YCbCr, luma would be Y and chroma would be Cb, Cr
        # ============================================================================
        
        # Determine downsample factor based on discrete levels
        strength_clamped = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
        if strength_clamped < 0.33:
            downsample_factor = 4   # Level 1: 1/4 resolution
        elif strength_clamped < 0.67:
            downsample_factor = 8   # Level 2: 1/8 resolution
        else:
            downsample_factor = 12  # Level 3: 1/12 resolution
        
        h, w = image_tensor.shape[-2:]
        
        # Subsample chroma channels (G and B), keep luma (R) at full resolution
        out_channels = []
        for c_idx in range(image_tensor.shape[0]):
            ch = image_tensor[c_idx:c_idx+1, :, :]
            
            if c_idx == 0:
                # Keep luma (R channel) at full resolution
                out_channels.append(ch)
            else:
                # Downsample chroma channels
                h_down = max(1, h // downsample_factor)
                w_down = max(1, w // downsample_factor)
                
                # Downsample with bilinear averaging
                ch_small = F.interpolate(ch.unsqueeze(0), size=(h_down, w_down), 
                                        mode='bilinear', align_corners=False)
                
                # Upsample back with nearest neighbor (creates characteristic blocky artifacts)
                ch_restored = F.interpolate(ch_small, size=(h, w), mode='nearest')
                
                out_channels.append(ch_restored.squeeze(0))
        
        image_tensor = torch.stack(out_channels, dim=0)

    return image_tensor


def apply_geometric_distortion(image_tensor, distortion_type, strength, generator=None):
    """
    Apply a single geometric distortion to an image tensor.
    
    Args:
        image_tensor: [C, H, W] tensor
        distortion_type: Type of distortion to apply
        strength: Strength of the distortion
    
    Returns:
        Distorted image tensor
    """
    import torch
    import torch.nn.functional as F
    import kornia.geometry.transform as KTF

    if distortion_type == 'keystone':
        # Keystone distortion (trapezoidal)
        h, w = image_tensor.shape[-2:]
        # Create keystone grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device),
            torch.linspace(-1, 1, w, device=image_tensor.device),
            indexing='ij'
        )
        keystone_factor = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 0.45)
        # Build perspective transform where the top edge is squeezed horizontally
        top_margin = keystone_factor * (w * 0.5)
        src = torch.tensor(
            [[[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]]],
            device=image_tensor.device, dtype=image_tensor.dtype
        )
        dst = src.clone()
        dst[:, 0, 0] = top_margin
        dst[:, 1, 0] = (w - 1.0) - top_margin
        transform = KTF.get_perspective_transform(src, dst)
        image_tensor = KTF.warp_perspective(
            image_tensor.unsqueeze(0), transform, dsize=(h, w),
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0)
        
    elif distortion_type == 'fisheye':
        # Differentiable, center-focused fisheye
        h, w = image_tensor.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device),
            torch.linspace(-1, 1, w, device=image_tensor.device),
            indexing='ij'
        )
        r2 = x * x + y * y
        denom = 1.0 - strength * r2
        mask_inside = (denom > 0.1)
        safe_denom = torch.where(mask_inside, denom, torch.ones_like(denom))
        x_distorted = x / safe_denom
        y_distorted = y / safe_denom
        grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)
        image_tensor = F.grid_sample(
            image_tensor.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0)
        # Zero-out pixels outside fisheye part
        image_tensor = image_tensor * mask_inside.unsqueeze(0)
        
    # elif distortion_type == 'barrel':
    #     # Barrel matching Data.py idea in normalized coords:
    #     # r_new = r * (1 - s * (r/r_max)); with r_max≈1 in [-1,1] grid → factor = 1 - s * r
    #     h, w = image_tensor.shape[-2:]
    #     y, x = torch.meshgrid(
    #         torch.linspace(-1, 1, h, device=image_tensor.device),
    #         torch.linspace(-1, 1, w, device=image_tensor.device),
    #         indexing='ij'
    #     )
    #     r = torch.sqrt(x * x + y * y)
    #     s = torch.clamp(torch.tensor(strength, device=image_tensor.device), 0.0, 0.5)
    #     factor = 1.0 - s * r
    #     factor = torch.clamp(factor, 0.1, 1.0)
    #     x_distorted = x * factor
    #     y_distorted = y * factor
    #     grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)
    #     image_tensor = F.grid_sample(
    #         image_tensor.unsqueeze(0), grid,
    #         mode='bilinear', padding_mode='zeros', align_corners=True
    #     ).squeeze(0)

    elif distortion_type == 'barrel':
        """
        Barrel distortion (negative radial distortion)
        Common in wide-angle lenses
        
        Reference:
        Brown, D. C. (1966). "Decentering distortion of lenses". 
        Photogrammetric Engineering, 32(3), 444-462.
        
        Mathematical Model (Polynomial Radial Distortion):
        ────────────────────────────────────────────────────
        Let (x, y) be normalized undistorted coordinates
        Let (x_d, y_d) be distorted coordinates
        Let r² = x² + y²
        
        Forward distortion model:
            x_d = x · (1 + k₁r² + k₂r⁴ + k₃r⁶)
            y_d = y · (1 + k₁r² + k₂r⁴ + k₃r⁶)
        
        For barrel distortion: k₁ < 0 (negative coefficient)
        
        Typical parameter ranges:
            k₁ ∈ [-0.5, -0.1]  (primary barrel effect)
            k₂ ∈ [-0.1, 0.0]   (optional higher-order term)
        
        Physical interpretation:
            - Negative k₁: Image magnification decreases with radius
            - Points farther from center are compressed inward
            - Straight lines bow outward (barrel shape)
        
        Strength mapping:
            strength ∈ [0, 1]
            k₁ = -strength × 0.5  → k₁ ∈ [-0.5, 0]
            k₂ = -strength × 0.05  → k₂ ∈ [-0.05, 0]
        """
        h, w = image_tensor.shape[-2:]
        
        # Create normalized coordinate grid [-1, 1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device),
            torch.linspace(-1, 1, w, device=image_tensor.device),
            indexing='ij'
        )
        
        # Calculate radius from center: r² = x² + y²
        r2 = x * x + y * y
        
        # Barrel distortion: k₁ < 0 (use polynomial model)
        k1 = -strength * 0.5  # Negative for barrel
        k2 = -strength * 0.05  # Optional second-order term
        
        # Apply radial distortion: (1 + k₁r² + k₂r⁴)
        radial_distortion = 1.0 + k1 * r2 + k2 * r2 * r2
        
        # Transform coordinates
        x_distorted = x * radial_distortion
        y_distorted = y * radial_distortion
        
        # Create mask for valid regions (coordinates stay within [-1, 1] bounds)
        # mask_inside = (x_distorted.abs() <= 1.0) & (y_distorted.abs() <= 1.0)
        
        # Create sampling grid
        grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)
        
        # Apply distortion via grid sampling with zero padding
        image_tensor = F.grid_sample(
            image_tensor.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0)
        
        # Zero-out pixels outside valid region
        # image_tensor = image_tensor * mask_inside.unsqueeze(0)
        
    elif distortion_type == 'pincushion':
        """
        Pincushion distortion (positive radial distortion)
        Common in telephoto lenses
        
        Primary Reference (Division Model):
        Fitzgibbon, A. W. (2001). "Simultaneous linear estimation of multiple 
        view geometry and lens distortion". Proceedings of the 2001 IEEE Computer 
        Society Conference on Computer Vision and Pattern Recognition (CVPR), 
        Vol. 1, pp. I-I.
        
        Alternative Reference (Polynomial Model):
        Brown, D. C. (1966). "Decentering distortion of lenses". 
        Photogrammetric Engineering, 32(3), 444-462.
        
        Mathematical Model (Polynomial Radial Distortion):
        ────────────────────────────────────────────────────
        Let (x, y) be normalized undistorted coordinates
        Let (x_d, y_d) be distorted coordinates
        Let r² = x² + y²
        
        Forward distortion model:
            x_d = x · (1 + k₁r² + k₂r⁴ + k₃r⁶)
            y_d = y · (1 + k₁r² + k₂r⁴ + k₃r⁶)
        
        For pincushion distortion: k₁ > 0 (positive coefficient)
        
        Typical parameter ranges:
            k₁ ∈ [0.1, 0.5]  (primary pincushion effect)
            k₂ ∈ [0.0, 0.1]   (optional higher-order term)
        
        Physical interpretation:
            - Positive k₁: Image magnification increases with radius
            - Points farther from center are stretched outward
            - Straight lines bow inward (pincushion shape)
        
        Strength mapping:
            strength ∈ [0, 1]
            k₁ = strength × 0.5  → k₁ ∈ [0, 0.5]
            k₂ = strength × 0.05  → k₂ ∈ [0, 0.05]
        """
        h, w = image_tensor.shape[-2:]
        
        # Create normalized coordinate grid [-1, 1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device),
            torch.linspace(-1, 1, w, device=image_tensor.device),
            indexing='ij'
        )
        
        # Calculate radius from center: r² = x² + y²
        r2 = x * x + y * y
        
        # Pincushion distortion: k₁ > 0 (use polynomial model, consistent with barrel)
        k1 = strength * 0.5  # Positive for pincushion
        k2 = strength * 0.05  # Optional second-order term
        
        # Apply radial distortion: (1 + k₁r² + k₂r⁴)
        radial_distortion = 1.0 + k1 * r2 + k2 * r2 * r2
        
        # Transform coordinates
        x_distorted = x * radial_distortion
        y_distorted = y * radial_distortion
        
        # Create mask for valid regions (coordinates stay within [-1, 1] bounds)
        # mask_inside = (x_distorted.abs() <= 1.0) & (y_distorted.abs() <= 1.0)
        
        # Create sampling grid
        grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)
        
        # Apply distortion via grid sampling with zero padding
        image_tensor = F.grid_sample(
            image_tensor.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0)
        
        # Zero-out pixels outside valid region
        # image_tensor = image_tensor * mask_inside.unsqueeze(0)
        
    elif distortion_type == 'tangential':
        """
        Tangential (decentering) distortion
        Caused by lens misalignment or non-perpendicular sensor plane
        
        Primary References:
        - Conrady, A. E. (1919). "Decentred Lens-Systems". Monthly Notices of 
          the Royal Astronomical Society, 79, 384-390.
        - Brown, D. C. (1971). "Close-range camera calibration". Photogrammetric 
          Engineering, 37(8), 855-866.
        
        Alternative Reference:
        - Tsai, R. Y. (1987). "A versatile camera calibration technique for 
          high-accuracy 3D machine vision metrology using off-the-shelf TV cameras 
          and lenses". IEEE Journal on Robotics and Automation, 3(4), 323-344.
        
        Mathematical Model (Brown-Conrady Tangential Distortion):
        ────────────────────────────────────────────────────────────
        Let (x, y) be normalized undistorted coordinates
        Let (x_d, y_d) be distorted coordinates
        Let r² = x² + y²
        
        Tangential distortion components:
            Δx_tan = 2p₁xy + p₂(r² + 2x²)
            Δy_tan = p₁(r² + 2y²) + 2p₂xy
        
        Complete distorted coordinates:
            x_d = x + Δx_tan
            y_d = y + Δy_tan
        
        Where:
            p₁, p₂ = tangential distortion coefficients
        
        Typical parameter ranges:
            p₁, p₂ ∈ [-0.01, 0.01]  (usually much smaller than radial)
        
        Physical interpretation:
            - Models lens decentering (optical elements misaligned)
            - Models sensor plane not perpendicular to optical axis
            - Creates asymmetric distortion (not rotationally symmetric)
            - Generally smaller effect than radial distortion
            - Also called "decentering distortion" or "thin prism distortion"
        
        Strength mapping:
            strength ∈ [0, 1]
            p₁ = strength × 0.02  → p₁ ∈ [0, 0.02]
            p₂ = strength × 0.02  → p₂ ∈ [0, 0.02]
        """
        h, w = image_tensor.shape[-2:]
        
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device),
            torch.linspace(-1, 1, w, device=image_tensor.device),
            indexing='ij'
        )
        
        # Calculate radius: r² = x² + y²
        r2 = x * x + y * y
        
        # Tangential distortion coefficients
        p1 = strength * 0.02
        p2 = strength * 0.02
        
        # Tangential distortion (Brown-Conrady model)
        # Δx = 2p₁xy + p₂(r² + 2x²)
        # Δy = p₁(r² + 2y²) + 2p₂xy
        x_tangential = 2*p1*x*y + p2*(r2 + 2*x*x)
        y_tangential = p1*(r2 + 2*y*y) + 2*p2*x*y
        
        # Apply tangential displacement
        x_distorted = x + x_tangential
        y_distorted = y + y_tangential
        
        # Create mask for valid regions (coordinates stay within [-1, 1] bounds)
        # mask_inside = (x_distorted.abs() <= 1.0) & (y_distorted.abs() <= 1.0)
        
        grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)
        
        # Apply distortion via grid sampling with zero padding
        image_tensor = F.grid_sample(
            image_tensor.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0)
        
        # Zero-out pixels outside valid region
        # image_tensor = image_tensor * mask_inside.unsqueeze(0)
        
    elif distortion_type == 'mustache':
        """
        Mustache (wave) distortion
        Combination of barrel near center and pincushion at edges
        Common in wide-range zoom lenses and some retrofocus designs
        
        Primary References:
        - Brown, D. C. (1966). "Decentering distortion of lenses". Photogrammetric 
          Engineering, 32(3), 444-462.
        - Santana-Cedrés, D., Gomez, L., Alemán-Flores, M., Salgado, A., Esclarín, J., 
          Mazorra, L., & Alvarez, L. (2015). "Invertibility and estimation of 
          two-parameter polynomial and division lens distortion models". SIAM Journal 
          on Imaging Sciences, 8(3), 1574-1606.
        
        Practical Examples:
        - Nikon 18-200mm (mentioned in literature as classic mustache case)
        - Many wide-range consumer zoom lenses
        
        Mathematical Model (Mustache/Wave Distortion):
        ──────────────────────────────────────────────────
        Let (x, y) be normalized undistorted coordinates
        Let (x_d, y_d) be distorted coordinates
        Let r² = x² + y²
        
        Mustache distortion model:
            x_d = x · (1 + k₁r² + k₂r⁴)
            y_d = y · (1 + k₁r² + k₂r⁴)
        
        Characteristic sign pattern for mustache:
            k₁ < 0  (barrel distortion dominates near center)
            k₂ > 0  (pincushion takes over at edges)
        
        Distortion behavior vs. radius:
            Small r: (1 + k₁r²) dominates → barrel effect (k₁ < 0)
            Large r: k₂r⁴ term dominates → pincushion effect (k₂ > 0)
            Transition: Zero-crossing at r_transition = √(-k₁/k₂)
        
        Physical interpretation:
            - Lens designers minimize distortion at two radius values
            - Creates S-shaped distortion curve vs. radius
            - Center: barrel (lines bow outward)
            - Edges: pincushion (lines bow inward)
            - Creates characteristic "mustache" or "wave" appearance
        
        Strength mapping:
            strength ∈ [0, 1]
            k₁ = -strength × 0.5  → k₁ ∈ [-0.5, 0]
            k₂ = strength × 0.6   → k₂ ∈ [0, 0.6]
        """
        h, w = image_tensor.shape[-2:]
        
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device),
            torch.linspace(-1, 1, w, device=image_tensor.device),
            indexing='ij'
        )
        
        # Calculate radius: r² and r⁴
        r2 = x * x + y * y
        r4 = r2 * r2
        
        # Mustache distortion: L(r) = 1 + k₁r² + k₂r⁴
        k1 = -strength * 0.5  # Negative for barrel near center
        k2 = strength * 0.6   # Positive for pincushion at edges
        
        radial_distortion = 1.0 + k1*r2 + k2*r4
        
        x_distorted = x * radial_distortion
        y_distorted = y * radial_distortion
        
        # Create mask for valid regions (coordinates stay within [-1, 1] bounds)
        # mask_inside = (x_distorted.abs() <= 1.0) & (y_distorted.abs() <= 1.0)
        
        grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)
        
        # Apply distortion via grid sampling with zero padding
        image_tensor = F.grid_sample(
            image_tensor.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0)
        
        # Zero-out pixels outside valid region
        # image_tensor = image_tensor * mask_inside.unsqueeze(0)
        
    elif distortion_type == 'brown_conrady':
        """
        Complete Brown-Conrady model with radial + tangential distortion
        The most comprehensive and widely-used lens distortion model
        
        Primary References:
        - Conrady, A. E. (1919). "Decentred Lens-Systems". Monthly Notices of 
          the Royal Astronomical Society, 79, 384-390.
        - Brown, D. C. (1966). "Decentering distortion of lenses". Photogrammetric 
          Engineering, 32(3), 444-462.
        - Brown, D. C. (1971). "Close-range camera calibration". Photogrammetric 
          Engineering, 37(8), 855-866.
        
        Modern Implementations:
        - Zhang, Z. (2000). "A flexible new technique for camera calibration". 
          IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11), 
          1330-1334.
        - OpenCV documentation (Bradski, G., & Kaehler, A. (2008). "Learning 
          OpenCV: Computer vision with the OpenCV library". O'Reilly Media, Inc.)
        
        Mathematical Model (Complete Brown-Conrady):
        ───────────────────────────────────────────────────
        Let (x, y) be normalized undistorted coordinates (principal point removed)
        Let (x_d, y_d) be distorted coordinates
        Let r² = x² + y²
        
        Radial distortion component:
            L_r(x,y) = (1 + k₁r² + k₂r⁴ + k₃r⁶)
        
        Tangential distortion components:
            Δx_tan = 2p₁xy + p₂(r² + 2x²)
            Δy_tan = p₁(r² + 2y²) + 2p₂xy
        
        Complete distortion model:
            x_d = x · L_r(x,y) + Δx_tan
            y_d = y · L_r(x,y) + Δy_tan
        
        Expanded form:
            x_d = x(1 + k₁r² + k₂r⁴ + k₃r⁶) + 2p₁xy + p₂(r² + 2x²)
            y_d = y(1 + k₁r² + k₂r⁴ + k₃r⁶) + p₁(r² + 2y²) + 2p₂xy
        
        Coefficient interpretations:
            k₁, k₂, k₃: Radial distortion coefficients (2nd, 4th, 6th order)
                k₁ < 0 → barrel distortion
                k₁ > 0 → pincushion distortion
                k₁ < 0, k₂ > 0 → mustache distortion
            
            p₁, p₂: Tangential/decentering distortion coefficients
                Model lens misalignment and sensor tilt
        
        Typical parameter ranges:
            k₁ ∈ [-0.5, 0.5]    (primary radial effect)
            k₂ ∈ [-0.1, 0.1]    (secondary radial effect)
            k₃ ∈ [-0.05, 0.05]  (tertiary, often negligible)
            p₁, p₂ ∈ [-0.01, 0.01]  (tangential, smaller magnitude)
        
        Strength mapping (creates mixed barrel + tangential):
            strength ∈ [0, 1]
            k₁ = -strength × 0.3  (barrel component)
            k₂ = strength × 0.1   (slight pincushion at edges)
            k₃ = 0.0              (not used here)
            p₁ = strength × 0.01  (tangential)
            p₂ = strength × 0.01  (tangential)
        """
        h, w = image_tensor.shape[-2:]
        
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device),
            torch.linspace(-1, 1, w, device=image_tensor.device),
            indexing='ij'
        )
        
        # Calculate radius and its powers: r², r⁴, r⁶
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        
        # Map strength to distortion coefficients
        k1 = -strength * 0.3  # Barrel component
        k2 = strength * 0.1   # Slight pincushion at edges
        k3 = 0.0              # Higher-order (often negligible)
        p1 = strength * 0.01  # Tangential
        p2 = strength * 0.01  # Tangential
        
        # Radial distortion component: L_r = (1 + k₁r² + k₂r⁴ + k₃r⁶)
        radial_distortion = 1.0 + k1*r2 + k2*r4 + k3*r6
        
        # Tangential distortion components
        # Δx_tan = 2p₁xy + p₂(r² + 2x²)
        # Δy_tan = p₁(r² + 2y²) + 2p₂xy
        x_tangential = 2*p1*x*y + p2*(r2 + 2*x*x)
        y_tangential = p1*(r2 + 2*y*y) + 2*p2*x*y
        
        # Combined distortion: x_d = x·L_r + Δx_tan, y_d = y·L_r + Δy_tan
        x_distorted = x * radial_distortion + x_tangential
        y_distorted = y * radial_distortion + y_tangential
        
        # Create mask for valid regions (coordinates stay within [-1, 1] bounds)
        # mask_inside = (x_distorted.abs() <= 1.0) & (y_distorted.abs() <= 1.0)
        
        grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)
        
        # Apply distortion via grid sampling with zero padding
        image_tensor = F.grid_sample(
            image_tensor.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0)
        
        # Zero-out pixels outside valid region
        # image_tensor = image_tensor * mask_inside.unsqueeze(0)
        
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")
    
    
    return image_tensor


def apply_gradient_friendly_distortions(image_tensor, ar_system):
    """
    Apply AR system distortions using gradient-friendly PyTorch operations.
    This preserves gradients for backpropagation to NN2.
    
    Args:
        image_tensor: [C, H, W] or [B, C, H, W] tensor on GPU
        ar_system: AR system with projector and glass configurations
        generator: Optional isolated torch.Generator for deterministic random events
    
    Returns:
        Tensor with same rank as input, distortions applied and gradients preserved
    """
    import torch
    import torch.nn.functional as F
    import kornia.augmentation as K

    # Handle batched input
    if isinstance(image_tensor, torch.Tensor) and image_tensor.dim() == 4:
        batch_size = image_tensor.size(0)
        outputs = []
        for i in range(batch_size):
            single = image_tensor[i]
            single = torch.clamp(single, 0, 1)
            if hasattr(ar_system, 'projector') and ar_system.projector:
                single = apply_projector_distortions(single, ar_system.projector, generator=getattr(ar_system, 'generator', None))
            if hasattr(ar_system, 'glass') and ar_system.glass:
                single = apply_glass_distortions(single, ar_system.glass, generator=getattr(ar_system, 'generator', None))
            outputs.append(torch.clamp(single, 0, 1))
        return torch.stack(outputs, dim=0)

    # Single image path [C, H, W]
    # Ensure image is in [0, 1] range
    image_tensor = torch.clamp(image_tensor, 0, 1)

    # Apply projector distortions first
    if hasattr(ar_system, 'projector') and ar_system.projector:
        image_tensor = apply_projector_distortions(image_tensor, ar_system.projector, generator=getattr(ar_system, 'generator', None))

    # Apply glass distortions second
    if hasattr(ar_system, 'glass') and ar_system.glass:
        image_tensor = apply_glass_distortions(image_tensor, ar_system.glass, generator=getattr(ar_system, 'generator', None))

    return torch.clamp(image_tensor, 0, 1)


def apply_projector_distortions(image_tensor, projector, generator=None):
    """Apply projector distortions using PyTorch operations"""
    # Apply color pipeline distortions
    for distortion in projector.color_pipeline:
        distortion_type = distortion.get('type', 'none')
        strength = distortion.get('strength', 0.3)
        image_tensor = apply_color_distortion(image_tensor, distortion_type, strength, generator=generator)
    
    # Apply geometric pipeline distortions
    for distortion in projector.geometrical_pipeline:
        distortion_type = distortion.get('type', 'none')
        strength = distortion.get('strength', 0.3)
        image_tensor = apply_geometric_distortion(image_tensor, distortion_type, strength, generator=generator)
    
    return image_tensor


def apply_glass_distortions(image_tensor, glass, generator=None):
    """Apply glass distortions using PyTorch operations"""
    # Apply color pipeline distortions
    for distortion in glass.color_pipeline:
        distortion_type = distortion.get('type', 'none')
        strength = distortion.get('strength', 0.3)
        image_tensor = apply_color_distortion(image_tensor, distortion_type, strength, generator=generator)
    
    # Apply geometric pipeline distortions
    for distortion in glass.geometrical_pipeline:
        distortion_type = distortion.get('type', 'none')
        strength = distortion.get('strength', 0.3)
        image_tensor = apply_geometric_distortion(image_tensor, distortion_type, strength, generator=generator)
    
    return image_tensor


def _gradient_consistency_loss(img_a, img_b):
    """
    Encourage matching image gradients between two tensors (L1 on dx, dy).
    Shapes: [B, C, H, W].
    """
    # Horizontal gradients
    grad_ax = img_a[:, :, :, 1:] - img_a[:, :, :, :-1]
    grad_bx = img_b[:, :, :, 1:] - img_b[:, :, :, :-1]
    # Vertical gradients
    grad_ay = img_a[:, :, 1:, :] - img_a[:, :, :-1, :]
    grad_by = img_b[:, :, 1:, :] - img_b[:, :, :-1, :]
    # L1 difference of gradients
    loss_x = (grad_ax - grad_bx).abs().mean()
    loss_y = (grad_ay - grad_by).abs().mean()
    return loss_x + loss_y


class LearnableResampler(nn.Module):
    """
    Lightweight CNN that learns to resample (downsample or upsample).
    Uses 2-layer conv + ReLU for efficient gradient flow.
    """
    def __init__(self, in_channels=3, out_channels=3, target_size=None):
        super(LearnableResampler, self).__init__()
        self.target_size = target_size  # (H, W) - explicit target resolution
        hidden_dim = 16
        
        # Simple 2-layer conv: preserves spatial info with padding
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        import torch.nn.functional as F
        
        # Apply learned transformation
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        # Resize to explicit target size if specified (gradient-friendly)
        if self.target_size is not None:
            if self.target_size != (x.size(-2), x.size(-1)):
                x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        return x


class MultiResAdapter(nn.Module):
    """
    Adapter that adds learnable downsample and upsample to any model.
    """
    def __init__(self, base_model, nn2_resolution=256, use_learnable_resampler=False):
        super().__init__()
        self.base_model = base_model
        self.nn2_resolution = nn2_resolution
        self.use_learnable_resampler = use_learnable_resampler
        
        if use_learnable_resampler:
            # Learnable resamplers (trainable) - target sizes set in forward pass
            self.downsampler = LearnableResampler(3, 3, target_size=None)  # Target set dynamically
            self.upsampler = LearnableResampler(3, 3, target_size=None)   # Target set dynamically
        else:
            # Use fixed bilinear interpolation
            self.downsampler = None
            self.upsampler = None
    
    def forward(self, x):
        import torch.nn.functional as F
        orig_h, orig_w = x.shape[-2:]
        target_low = (self.nn2_resolution, self.nn2_resolution)
        
        # Downsample
        if self.use_learnable_resampler and self.downsampler is not None:
            # Set explicit target size for downsampler
            self.downsampler.target_size = target_low
            x_low = self.downsampler(x)
        else:
            x_low = F.interpolate(x, size=target_low, 
                                  mode='bilinear', align_corners=False)
        
        # Run base model at low resolution
        output_low = self.base_model(x_low)
        
        # Handle dict output
        if isinstance(output_low, dict):
            output_high = {}
            for key, value in output_low.items():
                if isinstance(value, torch.Tensor) and value.dim() == 4:
                    if self.use_learnable_resampler and self.upsampler is not None:
                        # Set explicit target size for upsampler (original resolution)
                        self.upsampler.target_size = (orig_h, orig_w)
                        output_high[key] = self.upsampler(value)
                    else:
                        output_high[key] = F.interpolate(value, size=(orig_h, orig_w),
                                                         mode='bilinear', align_corners=False)
                else:
                    output_high[key] = value
            return output_high
        else:
            # Standard tensor output
            if self.use_learnable_resampler and self.upsampler is not None:
                # Set explicit target size for upsampler (original resolution)
                self.upsampler.target_size = (orig_h, orig_w)
                return self.upsampler(output_low)
            else:
                return F.interpolate(output_low, size=(orig_h, orig_w),
                                   mode='bilinear', align_corners=False)


def multires_nn2_forward(nn2_model, clean_input, nn2_resolution, use_multires, 
                         use_learnable_resampler=False, resampler_adapter=None):
    """
    Multi-resolution forward pass for NN2.
    
    When use_multires=True:
    1. Downsample input to nn2_resolution
    2. Run NN2 at lower resolution (where AMP works with grid_sample)
    3. Upsample output back to original resolution
    
    Args:
        nn2_model: NN2 model
        clean_input: Input tensor [B, C, H, W] at original resolution
        nn2_resolution: Target resolution for NN2 processing
        use_multires: Whether to use multi-resolution mode
        use_learnable_resampler: Whether to use trainable resamplers
        resampler_adapter: Optional MultiResAdapter wrapper
    
    Returns:
        nn2_output: Output at original resolution
        metadata: Dict with resolution info for logging
    """
    if not use_multires:
        # Standard single-resolution mode
        return nn2_model(clean_input), {'mode': 'standard'}
    
    # If using learnable resampler with adapter
    if use_learnable_resampler and resampler_adapter is not None:
        nn2_output = resampler_adapter(clean_input)
        orig_h, orig_w = clean_input.shape[-2:]
        metadata = {
            'mode': 'multires_learnable',
            'input_res': (orig_h, orig_w),
            'nn2_res': nn2_resolution,
            'upsampled_res': (orig_h, orig_w),
            'learnable': True
        }
        return nn2_output, metadata
    
    # Fixed bilinear interpolation (original implementation)
    import torch.nn.functional as F
    
    orig_h, orig_w = clean_input.shape[-2:]
    
    # Step 1: Downsample input to nn2_resolution
    if orig_h != nn2_resolution or orig_w != nn2_resolution:
        input_lowres = F.interpolate(
            clean_input, 
            size=(nn2_resolution, nn2_resolution), 
            mode='bilinear', 
            align_corners=False
        )
    else:
        input_lowres = clean_input
    
    # Step 2: Run NN2 at lower resolution (AMP-safe)
    output_lowres = nn2_model(input_lowres)
    
    # Step 3: Upsample output back to original resolution
    if isinstance(output_lowres, dict):
        # Handle dict output (A5, A6, A8 approaches)
        nn2_output = {}
        for key, value in output_lowres.items():
            if isinstance(value, torch.Tensor) and value.dim() == 4:
                # Upsample tensor outputs
                nn2_output[key] = F.interpolate(
                    value,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                # Keep non-tensor values as-is
                nn2_output[key] = value
    else:
        # Standard tensor output
        nn2_output = F.interpolate(
            output_lowres,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
    
    metadata = {
        'mode': 'multires_fixed',
        'input_res': (orig_h, orig_w),
        'nn2_res': nn2_resolution,
        'upsampled_res': (orig_h, orig_w),
        'learnable': False
    }
    
    return nn2_output, metadata


def compute_nn2_loss(nn2_output, clean_target, nn1_model, perceptual_loss_fn, args, zero_mask=None, ar_system=None):
    """
    Compute loss for NN2 training, handling different output formats for new approaches.
    
    Args:
        nn2_output: Output from NN2 model (can be tensor or dict for A5/A6)
        clean_target: Target clean image
        nn1_model: NN1 model (frozen) or None if using formulas
        perceptual_loss_fn: Perceptual loss function
        args: Training arguments
        zero_mask: Zero mask for masking
        ar_system: AR system for distortion formulas (used when nn1_model is None)
    
    Returns:
        loss: Computed loss
        nn1_final_output: Final output after NN1 or AR system (for logging)
    """
    
    # Helper function to apply distortion (either NN1 or AR system)
    def apply_distortion(image):
        if nn1_model is not None:
            return nn1_model(image)
        elif ar_system is not None:
            # Gradient-friendly AR system processing using PyTorch operations
            import torch
            import torch.nn.functional as F
            
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:  # [B, C, H, W] - batch processing
                    batch_size = image.size(0)
                    device = image.device
                    
                    # Process entire batch with gradient-friendly operations
                    result_batch = []
                    for i in range(batch_size):
                        single_image = image[i]  # [C, H, W]
                        distorted_tensor = apply_gradient_friendly_distortions(single_image, ar_system)
                        result_batch.append(distorted_tensor)
                    
                    # Stack back into batch tensor on GPU
                    return torch.stack(result_batch, dim=0)
                    
                elif image.dim() == 3:  # [C, H, W] - single image
                    return apply_gradient_friendly_distortions(image, ar_system)
                else:
                    raise ValueError(f"Expected 3D or 4D tensor for image, got {image.dim()}D")
            else:
                # PIL Image input - convert to tensor first
                from torchvision import transforms
                image_tensor = transforms.ToTensor()(image)
                if hasattr(image, 'device'):
                    image_tensor = image_tensor.to(image.device)
                return apply_gradient_friendly_distortions(image_tensor, ar_system)
        else:
            raise ValueError("Either nn1_model or ar_system must be provided")
    
    # Handle different output formats
    if isinstance(nn2_output, dict):
        # Two-network approaches (A5, A6, A8) - return dictionary with multiple outputs
        loss_variant = args.nn2_loss_variant
        
        # Get intermediate and final outputs
        im_prime = nn2_output['im_prime']
        im_double_prime = nn2_output['im_double_prime']
        
        # Pass final output through distortion (NN1 or AR system)
        nn1_final_output = apply_distortion(im_double_prime)
        
        # Compute loss based on variant
        if loss_variant in ['5-1', '6-1']:
            # Loss on both intermediate and final outputs
            # Distortion(im') loss
            nn1_im_prime_output = apply_distortion(im_prime)
            # For A6 keep original behavior
            loss_im_prime = combined_loss(
                nn1_im_prime_output, clean_target, perceptual_loss_fn, args.loss_type,
                perceptual_coeff=args.perceptual_coefficient
            )
            loss_im_double_prime = combined_loss(
                nn1_final_output, clean_target, perceptual_loss_fn, args.loss_type,
                perceptual_coeff=args.perceptual_coefficient
            )
            loss = loss_im_prime + loss_im_double_prime
            
        elif loss_variant in ['5-2', '6-2']:
            # Loss only on final output
            loss = combined_loss(nn1_final_output, clean_target, perceptual_loss_fn, args.loss_type, perceptual_coeff=args.perceptual_coefficient)
        else:
            raise ValueError(f"Unknown loss variant: {loss_variant}")
            
    else:
        # Single-network approaches (A1-A4, A7) - standard tensor output
        nn1_final_output = apply_distortion(nn2_output)
        if args.loss_type in ['geometry_loss', 'combined_perceptual', 'combined_mse_l1_perceptual']:
            loss, loss_breakdown = combined_loss(nn1_final_output, clean_target, perceptual_loss_fn, args.loss_type, perceptual_coeff=args.perceptual_coefficient, return_breakdown=True)
        else:
            loss = combined_loss(nn1_final_output, clean_target, perceptual_loss_fn, args.loss_type, perceptual_coeff=args.perceptual_coefficient)
    
    # Return loss breakdown if available
    if 'loss_breakdown' in locals():
        return loss, nn1_final_output, loss_breakdown
    else:
        return loss, nn1_final_output, None


def load_nn1_model(args, device):
    """Load pre-trained NN1 model and freeze it with comprehensive error checking"""
    
    # If dont_use_nn1 is True, return None for all NN1-related components
    if args.dont_use_nn1:
        logging.info("🚫 Skipping NN1 model loading - using distortion formulas instead")
        return None, None, None, None, 0
    
    model_path = args.nn1_model_path
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"NN1 model file not found: {model_path}")
    
    # Check if model file is readable
    if not os.access(model_path, os.R_OK):
        raise PermissionError(f"NN1 model file is not readable: {model_path}")
    
    # Check file size (should not be empty)
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        raise ValueError(f"NN1 model file is empty: {model_path}")
    
    log_nn1_model_loading(model_path, file_size)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load NN1 checkpoint from {model_path}: {str(e)}")
    
    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")
    
    required_keys = ['model_state_dict', 'args']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
    
    # Get model configuration from checkpoint
    args_dict = checkpoint.get('args', {})
    if not isinstance(args_dict, dict):
        raise ValueError(f"Invalid args format in checkpoint: expected dict, got {type(args_dict)}")
    
    model_type = args_dict.get('model_type', 'unet')
    if not isinstance(model_type, str):
        raise ValueError(f"Invalid model_type in checkpoint: expected string, got {type(model_type)}")
    
    # Extract zero_mask from checkpoint first (before creating model)
    checkpoint_zero_mask = checkpoint['model_state_dict'].get('zero_mask', None)
    if checkpoint_zero_mask is not None:
        checkpoint_zero_mask = checkpoint_zero_mask.to(device)
        logging.info(f"✅ Found zero_mask in checkpoint with shape: {checkpoint_zero_mask.shape}")
    else:
        logging.info("ℹ️ No zero_mask found in checkpoint model state_dict")
    
    logging.info(f"Creating NN1 model of type: {model_type}")
    
    # Create NN1-specific args for model creation
    nn1_args = argparse.Namespace()
    for attr in dir(args):
        if not attr.startswith('_'):
            setattr(nn1_args, attr, getattr(args, attr))
    
    # Override with NN1-specific parameters (map nn1_* to standard names)
    nn1_args.model_type = args.nn1_model_type
    nn1_args.model_complexity = args.nn1_model_complexity
    nn1_args.hidden_dim = args.nn1_hidden_dim
    nn1_args.use_zero_mask = args.nn1_use_zero_mask
    
    # Map UNet specific parameters for NN1
    if hasattr(args, 'nn1_unet_base_features'):
        nn1_args.base_features = args.nn1_unet_base_features
    
    try:
        # Pass the extracted zero_mask to the model creation
        nn1_model = create_model_and_setup(nn1_args, device, checkpoint_zero_mask)[0]
    except Exception as e:
        raise RuntimeError(f"Failed to create NN1 model of type {model_type}: {str(e)}")
    
    # Validate model was created
    if nn1_model is None:
        raise RuntimeError(f"create_model returned None for model type: {model_type}")
    
    # Count NN1 parameters
    nn1_total_params = count_parameters(nn1_model)
    logging.info(f"NN1 parameters: {nn1_total_params:,}")
    
    try:
        # Load state dict (should now include zero_mask if it exists)
        nn1_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict into NN1 model: {str(e)}")
    
    # Verify model is on correct device
    if next(nn1_model.parameters()).device != device:
        raise RuntimeError(f"NN1 model is not on expected device {device}")
    
    # Freeze NN1 model
    for param in nn1_model.parameters():
        param.requires_grad = False
    nn1_model.eval()
    
    # Verify model is in eval mode
    if nn1_model.training:
        raise RuntimeError("Failed to set NN1 model to eval mode")
    
    # Get AR system configuration if available
    projector_config = checkpoint.get('projector_config', None)
    glass_config = checkpoint.get('glass_config', None)
    
    # Use the zero_mask we already extracted from model_state_dict
    zero_mask = checkpoint_zero_mask
    
    # Validate AR system configs if present
    if projector_config is not None and not isinstance(projector_config, dict):
        logging.warning(f"Projector config has unexpected type: {type(projector_config)}")
        projector_config = None
    
    if glass_config is not None and not isinstance(glass_config, dict):
        logging.warning(f"Glass config has unexpected type: {type(glass_config)}")
        glass_config = None
    
    # Test forward pass with dummy input to ensure model works
    try:
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            dummy_output = nn1_model(dummy_input)
        
        if dummy_output is None:
            raise RuntimeError("NN1 model forward pass returned None")
        
        expected_shape = (1, 3, 256, 256)
        if dummy_output.shape != expected_shape:
            logging.warning(f"NN1 model output shape {dummy_output.shape} differs from expected {expected_shape}")
        
        logging.info(f"✅ NN1 model forward pass test successful")
        
    except Exception as e:
        raise RuntimeError(f"NN1 model forward pass test failed: {str(e)}")
    
    # Log successful loading
    try:
        # Count total parameters (not just trainable ones)
        total_param_count = sum(p.numel() for p in nn1_model.parameters())
        trainable_param_count = count_parameters(nn1_model)  # This will be 0 for frozen model
    except Exception as e:
        logging.warning(f"count_parameters failed: {e}, using manual count")
        # Manual parameter counting as fallback
        if hasattr(nn1_model, 'parameters'):
            param_list = list(nn1_model.parameters())
            total_param_count = sum(p.numel() for p in param_list)
            trainable_param_count = sum(p.numel() for p in param_list if p.requires_grad)
        else:
            total_param_count = 0
            trainable_param_count = 0
            logging.warning("Model has no parameters attribute")
    
    log_nn1_model_loaded(model_type, total_param_count, device)
    logging.info(f"✅ NN1 trainable parameters: {trainable_param_count:,} (should be 0 for frozen model)")
    logging.info(f"✅ NN1 training mode: {nn1_model.training}")
    
    if total_param_count == 0:
        raise RuntimeError("NN1 model has 0 total parameters - this indicates a serious loading error")
    
    if trainable_param_count > 0:
        logging.warning(f"NN1 model has {trainable_param_count} trainable parameters - it should be frozen")
    
    return nn1_model, projector_config, glass_config, zero_mask, nn1_total_params


def evaluate(nn2_model, nn1_model, data_loader, perceptual_loss_fn, device, loss_type, ar_system=None, use_roi_extraction=False, roi_target_size=(256, 256), perceptual_coefficient: float = 0.1):
    """Evaluate NN2 model with NN1 in the pipeline or using distortion formulas"""
    nn2_model.eval()
    
    # Handle NN1 model state (only if using NN1)
    if nn1_model is not None:
        nn1_model.eval()
        
        # Validate NN1 model state
        if nn1_model.training:
            raise RuntimeError("NN1 model is in training mode during evaluation - it should be frozen")
    
    # Validate NN1 model state (only if using NN1)
    if nn1_model is not None:
        trainable_params = sum(p.numel() for p in nn1_model.parameters() if p.requires_grad)
        if trainable_params > 0:
            raise RuntimeError(f"NN1 model has {trainable_params} trainable parameters during evaluation - it should be frozen")
    
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_rmse = 0
    total_lpips = 0
    num_batches = 0
    skipped_batches = 0
    
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    rmse_metric = torchmetrics.MeanSquaredError(squared=False).to(device)
    
    with torch.no_grad():
        total_eval_batches = len(data_loader)
        eval_batch_idx = 0
        eval_start_time = time.time()
        
        # print(f"\n{'='*60}")
        # print(f"EVALUATION START - {total_eval_batches} batches")
        # print(f"{'='*60}")
        
        for clean_input, clean_target in data_loader:
            clean_input, clean_target = clean_input.to(device), clean_target.to(device)
            
            # Use autocast for mixed precision evaluation
            if USE_AMP and torch.cuda.is_available():
                with amp.autocast(device_type='cuda', enabled=True):
                # NN2 processes clean image
                    nn2_output = nn2_model(clean_input)
                    
                    # Handle different output formats
                    if isinstance(nn2_output, dict):
                        # Two-network approaches - use final output
                        nn2_final_output = nn2_output['final']
                    else:
                        # Single-network approaches
                        nn2_final_output = nn2_output
                    
                    # Apply distortion using NN1 model or AR system
                    if nn1_model is not None:
                        final_output = nn1_model(nn2_final_output)
                    elif ar_system is not None:
                        # Use gradient-friendly AR system for distortion (same as in training)
                        if nn2_final_output.dim() == 4:  # [B, C, H, W] - batch processing
                            batch_size = nn2_final_output.size(0)
                            device = nn2_final_output.device
                            
                            # Process entire batch with gradient-friendly operations
                            result_batch = []
                            for i in range(batch_size):
                                single_image = nn2_final_output[i]  # [C, H, W]
                                distorted_tensor = apply_gradient_friendly_distortions(single_image, ar_system)
                                result_batch.append(distorted_tensor)
                            
                            # Stack back into batch tensor on GPU
                            final_output = torch.stack(result_batch, dim=0)
                        else:  # [C, H, W] - single image
                            final_output = apply_gradient_friendly_distortions(nn2_final_output, ar_system)
                    else:
                        raise ValueError("Either nn1_model or ar_system must be provided for evaluation")
                    
                    # Mask out ROI areas from final prediction before loss calculation
                    if use_roi_extraction:
                        from helper import mask_out_roi_areas
                        final_output_masked = mask_out_roi_areas(final_output, GLOBAL_ROI_COORDS)
                    else:
                        final_output_masked = final_output
                    
                    # Loss between final output and expected output (ROI-based)
                    if use_roi_extraction:
                        from helper import create_expected_output_with_roi, has_geometric_distortion_in_ar_system
                        has_geo = has_geometric_distortion_in_ar_system(ar_system) if ar_system is not None else None
                        expected_output = create_expected_output_with_roi(
                            clean_target, GLOBAL_ROI_COORDS, roi_target_size, has_geometric_distortion=has_geo
                        )
                        loss = combined_loss(final_output_masked, expected_output, perceptual_loss_fn, loss_type, perceptual_coeff=perceptual_coefficient)
                    else:
                        loss = combined_loss(final_output_masked, clean_target, perceptual_loss_fn, loss_type, perceptual_coeff=perceptual_coefficient)
            else:
                # Standard precision evaluation
                # NN2 processes clean image
                nn2_output = nn2_model(clean_input)
                
                # Handle different output formats
                if isinstance(nn2_output, dict):
                    # Two-network approaches - use final output
                    nn2_final_output = nn2_output['final']
                else:
                    # Single-network approaches
                    nn2_final_output = nn2_output
                
                # Apply distortion using NN1 model or AR system
                if nn1_model is not None:
                    final_output = nn1_model(nn2_final_output)
                elif ar_system is not None:
                    # Use gradient-friendly AR system for distortion (same as in training)
                    if nn2_final_output.dim() == 4:  # [B, C, H, W] - batch processing
                        batch_size = nn2_final_output.size(0)
                        device = nn2_final_output.device
                        
                        # Process entire batch with gradient-friendly operations
                        result_batch = []
                        for i in range(batch_size):
                            single_image = nn2_final_output[i]  # [C, H, W]
                            distorted_tensor = apply_gradient_friendly_distortions(single_image, ar_system)
                            result_batch.append(distorted_tensor)
                        
                        # Stack back into batch tensor on GPU
                        final_output = torch.stack(result_batch, dim=0)
                    else:  # [C, H, W] - single image
                        final_output = apply_gradient_friendly_distortions(nn2_final_output, ar_system)
                else:
                    raise ValueError("Either nn1_model or ar_system must be provided for evaluation")
                
                # Mask out ROI areas from final prediction before loss calculation
                if use_roi_extraction:
                    from helper import mask_out_roi_areas
                    final_output_masked = mask_out_roi_areas(final_output, GLOBAL_ROI_COORDS)
                else:
                    final_output_masked = final_output
                
                # Loss between final output and expected output (ROI-based)
                if use_roi_extraction:
                    from helper import create_expected_output_with_roi, has_geometric_distortion_in_ar_system
                    has_geo = has_geometric_distortion_in_ar_system(ar_system) if ar_system is not None else None
                    expected_output = create_expected_output_with_roi(
                        clean_target, GLOBAL_ROI_COORDS, roi_target_size, has_geometric_distortion=has_geo
                    )
                    loss = combined_loss(final_output_masked, expected_output, perceptual_loss_fn, loss_type, perceptual_coeff=perceptual_coefficient)
                else:
                    loss = combined_loss(final_output_masked, clean_target, perceptual_loss_fn, loss_type, perceptual_coeff=perceptual_coefficient)
            
            # For metrics, use the same expected output format as loss calculation
            if use_roi_extraction:
                from helper import create_expected_output_with_roi, has_geometric_distortion_in_ar_system
                has_geo = has_geometric_distortion_in_ar_system(ar_system) if ar_system is not None else None
                clean_target_metric = create_expected_output_with_roi(
                    clean_target, GLOBAL_ROI_COORDS, roi_target_size, has_geometric_distortion=has_geo
                )
                final_output_metric = final_output_masked
            else:
                clean_target_metric = clean_target
                final_output_metric = final_output_masked

            # Calculate metrics on clamped outputs for proper SSIM calculation
            final_output_clamped = torch.clamp(final_output_metric, 0, 1)
            clean_target_clamped = torch.clamp(clean_target_metric, 0, 1)
            
            # Check for NaN/Inf before computing metrics
            if torch.isnan(final_output_clamped).any() or torch.isinf(final_output_clamped).any() or \
               torch.isnan(clean_target_clamped).any() or torch.isinf(clean_target_clamped).any():
                skipped_batches += 1
                eval_batch_idx += 1
                continue
            
            psnr_val = psnr_metric(final_output_clamped, clean_target_clamped)
            ssim_val = ssim_metric(final_output_clamped, clean_target_clamped)
            rmse_val = rmse_metric(final_output_clamped, clean_target_clamped)

            # LPIPS calculation
            if not NORM:
                outputs_lpips = 2.0 * final_output_metric - 1.0
                target_lpips = 2.0 * clean_target_metric - 1.0
            else:
                outputs_lpips = final_output_metric
                target_lpips = clean_target_metric
            lpips_val = perceptual_loss_fn.lpips_model(outputs_lpips, target_lpips).mean()
            
            # Check if metrics are valid
            if torch.isnan(loss).any() or torch.isinf(loss).any() or \
               torch.isnan(psnr_val).any() or torch.isinf(psnr_val).any() or \
               torch.isnan(ssim_val).any() or torch.isinf(ssim_val).any() or \
               torch.isnan(rmse_val).any() or torch.isinf(rmse_val).any() or \
               torch.isnan(lpips_val).any() or torch.isinf(lpips_val).any():
                skipped_batches += 1
                eval_batch_idx += 1
                continue
            
            total_loss += loss.item()
            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()
            total_rmse += rmse_val.item()
            total_lpips += lpips_val.item()
            num_batches += 1
            eval_batch_idx += 1
            
            # Calculate running averages for evaluation
            avg_eval_loss = total_loss / num_batches
            avg_eval_psnr = total_psnr / num_batches
            avg_eval_ssim = total_ssim / num_batches
            avg_eval_rmse = total_rmse / num_batches
            avg_eval_lpips = total_lpips / num_batches
            
            # Evaluation batch progress printing removed as requested
    
    # Avoid division by zero
    if num_batches == 0:
        return {
            'loss': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'rmse': 0.0,
            'lpips': 0.0,
            'skipped_batches': skipped_batches,
            'total_batches': total_eval_batches
        }
    
    return {
        'loss': total_loss / num_batches,
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches,
        'rmse': total_rmse / num_batches,
        'lpips': total_lpips / num_batches,
        'skipped_batches': skipped_batches,
        'total_batches': total_eval_batches
    }


def save_sample_images(nn2_model, nn1_model, data_loader, device, output_dir, ar_system=None, num_samples=10, phase='train', experiment_name=None, use_real_dataset=False, use_roi_extraction=False, roi_target_size=(256, 256)):
    """
    Save sample images showing the key pipeline stages.
    
    Includes suprathreshold difference maps that highlight perceptually significant differences
    between expected output and AR system output(s). Suprathreshold masking accounts for 
    contrast perception across luminance, frequencies, and color modulation directions.
    """
    nn2_model.eval()
    
    # Handle NN1 model state (only if using NN1)
    if nn1_model is not None:
        nn1_model.eval()
        
        # Validate NN1 model state
        if nn1_model.training:
            raise RuntimeError("NN1 model is in training mode during sample saving - it should be frozen")
        
        trainable_params = sum(p.numel() for p in nn1_model.parameters() if p.requires_grad)
        if trainable_params > 0:
            raise RuntimeError(f"NN1 model has {trainable_params} trainable parameters during sample saving - it should be frozen")
    
    # Use the zero_mask from NN1 model or AR system
    zero_mask = None
    if nn1_model is not None and hasattr(nn1_model, 'zero_mask') and nn1_model.zero_mask is not None:
        zero_mask = nn1_model.zero_mask
        logging.info(f"Using zero_mask from NN1 model with shape: {zero_mask.shape}")
    else:
        logging.info("No zero_mask available for visualization")
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Saving {num_samples} {phase} samples with simplified pipeline visualization (including suprathreshold difference maps) to: {output_dir}")
    
    sample_count = 0
    with torch.no_grad():
        for clean_input, clean_target in data_loader:
            clean_input, clean_target = clean_input.to(device), clean_target.to(device)
            
            # 1. Clean image (input & target)
            clean_img = clean_input
            
            # 2. Pre-distorted image (output of NN2)
            nn2_output = nn2_model(clean_input)
            
            # Handle different output formats for NN2
            if isinstance(nn2_output, dict):
                # Two-network approaches - use final output for NN1
                nn2_final_output = nn2_output['final']
                # Save intermediate outputs if available
                nn2_intermediate = nn2_output.get('im_prime', None)
            else:
                # Single-network approaches
                nn2_final_output = nn2_output
                nn2_intermediate = None
            
            # 3. Distorted pre-distorted image (output of NN1 or AR system) = NN1(NN2(clean)) or AR(NN2(clean))
            if nn1_model is not None:
                # Feed NN2 output into NN1 to match early-sample behavior
                nn1_nn2_output_raw = nn1_model(nn2_final_output)
            elif ar_system is not None:
                # Apply AR formulas to NN2 output (not the clean input) to match early-sample behavior
                if nn2_final_output.dim() == 4:  # [B, C, H, W]
                    batch_size = nn2_final_output.size(0)
                    device = nn2_final_output.device
                    result_batch = []
                    for i in range(batch_size):
                        single_image = nn2_final_output[i]  # [C, H, W]
                        # If normalized, denormalize to [0,1] before AR and clamp
                        if NORM:
                            mean_t = torch.tensor([0.485, 0.456, 0.406], device=single_image.device).view(3, 1, 1)
                            std_t = torch.tensor([0.229, 0.224, 0.225], device=single_image.device).view(3, 1, 1)
                            single_image = torch.clamp(single_image * std_t + mean_t, 0, 1)
                        else:
                            single_image = torch.clamp(single_image, 0, 1)
                        distorted_tensor = apply_gradient_friendly_distortions(single_image, ar_system)
                        result_batch.append(distorted_tensor)
                    nn1_nn2_output_raw = torch.stack(result_batch, dim=0)
                else:  # [C, H, W]
                    single_image = nn2_final_output
                    if NORM:
                        mean_t = torch.tensor([0.485, 0.456, 0.406], device=single_image.device).view(3, 1, 1)
                        std_t = torch.tensor([0.229, 0.224, 0.225], device=single_image.device).view(3, 1, 1)
                        single_image = torch.clamp(single_image * std_t + mean_t, 0, 1)
                    else:
                        single_image = torch.clamp(single_image, 0, 1)
                    nn1_nn2_output_raw = apply_gradient_friendly_distortions(single_image, ar_system)
            else:
                raise ValueError("Either nn1_model or ar_system must be provided for visualization")
            
            # Apply ROI masking to NN1 output if ROI extraction is enabled
            nn1_nn2_output = nn1_nn2_output_raw  # Start with raw output
            if use_roi_extraction and GLOBAL_ROI_COORDS is not None:
                from helper import mask_out_roi_areas
                nn1_nn2_output = mask_out_roi_areas(nn1_nn2_output_raw, GLOBAL_ROI_COORDS)
            
            # 4. Real output of NN1 = Glass(Projector(NN2(clean))
            # Note: We only use gradient-friendly distortions now (no PIL-based forward method)
            # The gradient-friendly version is already computed as nn1_nn2_output_raw
            real_ar_nn2_batch = None
            
            # 5. Ablation: bypass NN2, pass clean input through environment (what final user sees)
            ablation_batch = None
            if ar_system is not None:
                ablation_results = []
                for i in range(clean_input.size(0)):
                    img_tensor = clean_input[i]
                    if NORM:
                        mean_t = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
                        std_t = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
                        img_tensor = img_tensor * std_t + mean_t
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    ablation = apply_gradient_friendly_distortions(img_tensor, ar_system)
                    if NORM:
                        ablation = (ablation - mean_t) / std_t
                    ablation_results.append(ablation)
                ablation_batch = torch.stack(ablation_results, dim=0)
            
            # 6. AR(clean) for visualization in final samples - separate from ablation for display
            ar_clean_batch = ablation_batch  # Reuse ablation computation for display
            
            for i in range(clean_input.size(0)):
                if sample_count >= num_samples:
                    return
                
                # Convert all tensors to numpy for visualization
                clean_img_np = clean_input[i].cpu().numpy().transpose(1, 2, 0)
                nn2_img_np = nn2_final_output[i].cpu().numpy().transpose(1, 2, 0)
                nn1_nn2_img_np_raw = nn1_nn2_output_raw[i].cpu().numpy().transpose(1, 2, 0)  # Raw NN1 output (before ROI masking)
                nn1_nn2_img_np = nn1_nn2_output[i].cpu().numpy().transpose(1, 2, 0)  # NN1 output (after ROI masking)
                
                # Handle intermediate outputs for two-network approaches
                if nn2_intermediate is not None:
                    nn2_intermediate_np = nn2_intermediate[i].cpu().numpy().transpose(1, 2, 0)
                    # Denormalize intermediate output if needed
                    if NORM:
                        nn2_intermediate_np = nn2_intermediate_np * std + mean
                    nn2_intermediate_np = np.clip(nn2_intermediate_np, 0, 1)
                else:
                    nn2_intermediate_np = None
                
                # Denormalize if needed
                if NORM:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    clean_img_np = clean_img_np * std + mean
                    nn2_img_np = nn2_img_np * std + mean
                    nn1_nn2_img_np_raw = nn1_nn2_img_np_raw * std + mean
                    nn1_nn2_img_np = nn1_nn2_img_np * std + mean

                # Always clamp to [0, 1] for matplotlib compatibility
                clean_img_np = np.clip(clean_img_np, 0, 1)
                nn2_img_np = np.clip(nn2_img_np, 0, 1)
                nn1_nn2_img_np_raw = np.clip(nn1_nn2_img_np_raw, 0, 1)
                nn1_nn2_img_np = np.clip(nn1_nn2_img_np, 0, 1)
                
                # Build visualization tensors in the same format as early epoch samples and save with vutils.save_image
                import torchvision.utils as vutils
                from PIL import Image, ImageDraw, ImageFont
                
                # Helper to add label to tensor image (borrowed from early sample saving)
                def add_label_to_tensor(tensor_img, label_text):
                    img_np = (tensor_img.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    draw = ImageDraw.Draw(pil_img)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    x, y = 5, 5
                    draw.rectangle([x-2, y-2, x+text_width+2, y+text_height+2], fill=(0, 0, 0))
                    draw.text((x, y), label_text, fill=(255, 255, 255), font=font)
                    labeled_np = np.array(pil_img).astype(np.float32) / 255.0
                    labeled_tensor = torch.from_numpy(labeled_np).permute(2, 0, 1).unsqueeze(0).to(clean_input.device)
                    return labeled_tensor
                
                # Helper to compute suprathreshold difference map
                def compute_suprathreshold_diff(img1, img2):
                    """
                    Compute suprathreshold difference map accounting for perceptual contrast.
                    Highlights differences above the threshold of human perception.
                    
                    Args:
                        img1, img2: torch tensors [1, 3, H, W] in range [0, 1]
                    Returns:
                        diff_vis: torch tensor [1, 3, H, W] showing perceptually-weighted differences
                    """
                    # Compute absolute difference
                    diff = torch.abs(img1 - img2)
                    
                    # Convert to luminance for perceptual weighting (Rec. 709)
                    # Y = 0.2126*R + 0.7152*G + 0.0722*B
                    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=diff.device).view(1, 3, 1, 1)
                    luma_diff = (diff * weights).sum(dim=1, keepdim=True)
                    
                    # Apply suprathreshold function: emphasize differences above perceptual threshold
                    # Using a power function to model contrast sensitivity (gamma ~ 2.2-2.4)
                    threshold = 0.01  # JND threshold (Just Noticeable Difference)
                    suprathreshold = torch.clamp(luma_diff - threshold, min=0.0)
                    suprathreshold = torch.pow(suprathreshold * 10.0, 0.5)  # Enhance visibility
                    suprathreshold = torch.clamp(suprathreshold, 0, 1)
                    
                    # Create colored difference map: red for differences
                    # Red channel = suprathreshold difference
                    # Green/Blue = show texture of difference
                    diff_colored = torch.cat([
                        suprathreshold,  # Red: suprathreshold magnitude
                        diff[:, 1:2, :, :] * 0.5,  # Green: attenuated color diff
                        diff[:, 2:3, :, :] * 0.5   # Blue: attenuated color diff
                    ], dim=1)
                    
                    return torch.clamp(diff_colored, 0, 1)
                
                # Helper to compute absolute difference map
                def compute_absolute_diff(img1, img2):
                    """
                    Compute simple absolute difference map between two images.
                    
                    Args:
                        img1, img2: torch tensors [1, 3, H, W] in range [0, 1]
                    Returns:
                        diff_vis: torch tensor [1, 3, H, W] showing absolute differences
                    """
                    # Compute absolute difference
                    diff = torch.abs(img1 - img2)
                    
                    # Normalize to [0, 1] range for visualization
                    # Scale by a factor to make differences more visible
                    diff_scaled = torch.clamp(diff * 5.0, 0, 1)  # Scale by 5x for visibility
                    
                    return diff_scaled

                # Prepare tensors (clamped 0..1)
                original_vis = torch.clamp(clean_input[i:i+1], 0, 1)
                nn2_out_vis = torch.clamp(nn2_final_output[i:i+1], 0, 1)
                nn1_out_raw_vis = torch.clamp(nn1_nn2_output_raw[i:i+1], 0, 1)
                nn1_out_vis = torch.clamp(nn1_nn2_output[i:i+1], 0, 1)

                # Expected output/ROI visualization
                if use_roi_extraction and GLOBAL_ROI_COORDS is not None:
                    from helper import create_expected_output_with_roi, has_geometric_distortion_in_ar_system
                    has_geo = has_geometric_distortion_in_ar_system(ar_system) if ar_system is not None else None
                    expected_out = create_expected_output_with_roi(clean_target[i:i+1], GLOBAL_ROI_COORDS, roi_target_size, has_geometric_distortion=has_geo)
                    mask_label = "Zero mask + ROI"
                else:
                    expected_out = clean_target[i:i+1]
                    mask_label = "Zero mask"
                expected_vis = torch.clamp(expected_out, 0, 1)

                # Zero mask visualization if available
                if zero_mask is not None:
                    zero_mask_vis = zero_mask.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)
                    if zero_mask_vis.shape[-2:] != original_vis.shape[-2:]:
                        zero_mask_vis = F.interpolate(zero_mask_vis, size=original_vis.shape[-2:], mode='nearest')
                else:
                    zero_mask_vis = torch.zeros_like(original_vis)

                # Label tensors
                input_labeled = add_label_to_tensor(original_vis, "Input")
                if nn2_intermediate is not None:
                    nn2_intermediate_vis = torch.clamp(nn2_intermediate[i:i+1], 0, 1)
                    nn2_intermediate_labeled = add_label_to_tensor(nn2_intermediate_vis, "Net1 Output (Intermediate)")
                    nn2_out_labeled = add_label_to_tensor(nn2_out_vis, "Net2 Output (Final)")
                    nn1_out_raw_labeled = add_label_to_tensor(nn1_out_raw_vis, "AR system output(Raw - No ROI)" if nn1_model is None else "NN1 output(Raw - No ROI)")
                    nn1_out_labeled = add_label_to_tensor(nn1_out_vis, "Output of AR system (with ROI)" if nn1_model is None else "Output of NN1 (with ROI)")
                    expected_labeled = add_label_to_tensor(expected_vis, "Expected output")
                    zero_mask_labeled = add_label_to_tensor(zero_mask_vis, mask_label)
                    
                    # Compute suprathreshold difference map
                    diff_output = compute_suprathreshold_diff(expected_vis, nn1_out_vis)
                    diff_output_labeled = add_label_to_tensor(diff_output, "Diff: Expected vs Output")
                    
                    # Compute absolute difference map
                    abs_diff_output = compute_absolute_diff(expected_vis, nn1_out_vis)
                    abs_diff_output_labeled = add_label_to_tensor(abs_diff_output, "Abs Diff: Expected vs Output")
                    
                    sample_grid = torch.cat([input_labeled, nn2_intermediate_labeled, nn2_out_labeled, nn1_out_raw_labeled, nn1_out_labeled, expected_labeled, diff_output_labeled, abs_diff_output_labeled, zero_mask_labeled], dim=0)
                    nrow = 3  # 3 rows of 3 images each
                else:
                    nn2_out_labeled = add_label_to_tensor(nn2_out_vis, "Output of NN2")
                    expected_labeled = add_label_to_tensor(expected_vis, "Expected output")
                    zero_mask_labeled = add_label_to_tensor(zero_mask_vis, mask_label)
                    if nn1_model is None and ar_clean_batch is not None:
                        ar_clean_vis = torch.clamp(ar_clean_batch[i:i+1], 0, 1)
                        ar_clean_labeled = add_label_to_tensor(ar_clean_vis, "AR(clean) - No NN2")
                        ar_nn2_labeled = add_label_to_tensor(nn1_out_raw_vis, "AR(NN2(clean)) - With NN2")
                        
                        # Compute suprathreshold difference maps
                        diff_ar_clean = compute_suprathreshold_diff(expected_vis, ar_clean_vis)
                        diff_ar_nn2 = compute_suprathreshold_diff(expected_vis, nn1_out_raw_vis)
                        diff_ar_clean_labeled = add_label_to_tensor(diff_ar_clean, "Diff: Expected vs AR(clean)")
                        diff_ar_nn2_labeled = add_label_to_tensor(diff_ar_nn2, "Diff: Expected vs AR(NN2)")
                        
                        # Compute absolute difference maps
                        abs_diff_ar_clean = compute_absolute_diff(expected_vis, ar_clean_vis)
                        abs_diff_ar_nn2 = compute_absolute_diff(expected_vis, nn1_out_raw_vis)
                        abs_diff_ar_clean_labeled = add_label_to_tensor(abs_diff_ar_clean, "Abs Diff: Expected vs AR(clean)")
                        abs_diff_ar_nn2_labeled = add_label_to_tensor(abs_diff_ar_nn2, "Abs Diff: Expected vs AR(NN2)")
                        
                        sample_grid = torch.cat([input_labeled, nn2_out_labeled, ar_clean_labeled, ar_nn2_labeled, expected_labeled, diff_ar_clean_labeled, diff_ar_nn2_labeled, abs_diff_ar_clean_labeled, abs_diff_ar_nn2_labeled, zero_mask_labeled], dim=0)
                        nrow = 5  # 2 rows: [Input, NN2, AR(clean), AR(NN2), Expected] and [Diff1, Diff2, AbsDiff1, AbsDiff2, Mask]
                    else:
                        nn1_out_raw_labeled = add_label_to_tensor(nn1_out_raw_vis, "NN1 output(Raw - No ROI)" if nn1_model is not None else "AR system output(Raw - No ROI)")
                        nn1_out_labeled = add_label_to_tensor(nn1_out_vis, "Output of NN1 (with ROI)" if nn1_model is not None else "Output of AR system (with ROI)")
                        
                        # Compute suprathreshold difference map
                        diff_output = compute_suprathreshold_diff(expected_vis, nn1_out_vis)
                        diff_output_labeled = add_label_to_tensor(diff_output, "Diff: Expected vs Output")
                        
                        # Compute absolute difference map
                        abs_diff_output = compute_absolute_diff(expected_vis, nn1_out_vis)
                        abs_diff_output_labeled = add_label_to_tensor(abs_diff_output, "Abs Diff: Expected vs Output")
                        
                        sample_grid = torch.cat([input_labeled, nn2_out_labeled, nn1_out_raw_labeled, nn1_out_labeled, expected_labeled, diff_output_labeled, abs_diff_output_labeled, zero_mask_labeled], dim=0)
                        nrow = 4  # 2 rows of 4 images each

                # Save grid with torchvision like early samples
                filename = f"NN2_simplified_{phase}_sample_{sample_count:03d}.png"
                vutils.save_image(sample_grid, os.path.join(output_dir, filename), nrow=nrow, padding=4, normalize=False)
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    logging.info(f"✅ Saved {sample_count} {phase} samples with simplified pipeline visualization (including suprathreshold difference maps)")





def main():
    """Main training function"""
    # Declare global variables at the beginning
    global GLOBAL_ROI_COORDS

    args = parse_arguments()

    # Set the results directory based on server_number
    NN2_RESULTS_DIR = f"{args.server_number}_NN2-results"
    print(f"📁 Results directory: {NN2_RESULTS_DIR}")

    # Set roi_target_size default to image_size if not specified
    if args.roi_target_size is None:
        args.roi_target_size = [args.image_size, args.image_size]

    # Set random seed for reproducible results
    set_seed(args.seed)

    # Log the grid sample configuration for clarity
    print(f"🔧 Grid Sample Configuration: LIBRARY (F.grid_sample) with {args.grid_sample_mode} interpolation")

    # Normalize model_type to lowercase for case-insensitive input
    args.nn1_model_type = args.nn1_model_type.lower()
    args.nn2_model_type = args.nn2_model_type.lower()

    # Validate GPU index
    if args.gpu_index < -1:
        raise ValueError(f"Invalid GPU index: {args.gpu_index}. Must be -1 (CPU) or >= 0 (GPU index)")
    if args.gpu_index >= 0 and torch.cuda.is_available() and args.gpu_index >= torch.cuda.device_count():
        raise ValueError(f"GPU index {args.gpu_index} is not available. Only {torch.cuda.device_count()} GPU(s) available (indices 0-{torch.cuda.device_count()-1})")
    if args.gpu_index >= 0 and not torch.cuda.is_available():
        print(f"Warning: GPU index {args.gpu_index} specified but CUDA is not available. Falling back to CPU.")
        args.gpu_index = -1

    # Validate NN1 model path only if not using formulas
    if not args.dont_use_nn1:
        if not args.nn1_model_path:
            raise ValueError("--nn1_model_path is required when not using --dont_use_nn1")
        if not os.path.exists(args.nn1_model_path):
            raise FileNotFoundError(f"NN1 model not found: {args.nn1_model_path}")

    # Setup device using helper function
    device = setup_device(args.gpu_index)
    
    # Create pipeline configurations using helper function
    projector_config, glass_config = create_pipeline_configs(args, DEFAULT_PIPELINE_STRENGTH)
    
    # Load NN1 model (frozen) or skip if using formulas
    nn1_model, loaded_projector_config, loaded_glass_config, loaded_zero_mask, nn1_total_params = load_nn1_model(args, device)
    
    # Additional validation after loading (only if using NN1)
    if not args.dont_use_nn1:
        if nn1_model is None:
            raise RuntimeError("NN1 model is None after loading - this should not happen")
        
        # Verify model is still in eval mode and frozen
        if nn1_model.training:
            raise RuntimeError("NN1 model is in training mode - it should be frozen")
        
        # Check if any parameters are trainable (they shouldn't be)
        trainable_params = sum(p.numel() for p in nn1_model.parameters() if p.requires_grad)
        if trainable_params > 0:
            raise RuntimeError(f"NN1 model has {trainable_params} trainable parameters - it should be completely frozen")
        
        logging.info(f"✅ NN1 model validation passed - model is properly frozen and in eval mode")
    else:
        logging.info(f"✅ Using distortion formulas instead of NN1 model")
    
    # Use loaded configs if available and using NN1, otherwise use command line args
    if not args.dont_use_nn1 and loaded_projector_config and loaded_glass_config:
        # Ensure the loaded configs have the correct structure
        if isinstance(loaded_projector_config, dict) and 'color_pipeline' in loaded_projector_config and 'geometrical_pipeline' in loaded_projector_config:
            projector_config = loaded_projector_config
            logging.info("Using projector configuration from loaded NN1 model")
        else:
            logging.warning(f"Loaded projector config has unexpected structure: {type(loaded_projector_config)}")
            logging.info("Using AR system configuration from command line arguments")

        if isinstance(loaded_glass_config, dict) and 'color_pipeline' in loaded_glass_config and 'geometrical_pipeline' in loaded_glass_config:
            glass_config = loaded_glass_config
            logging.info("Using glass configuration from loaded NN1 model")
        else:
            logging.warning(f"Loaded glass config has unexpected structure: {type(loaded_glass_config)}")
            logging.info("Using AR system configuration from command line arguments")
    else:
        if args.dont_use_nn1:
            logging.info("Using AR system configuration from command line arguments (--dont_use_nn1 mode)")
            
            # Validate that meaningful pipeline configurations are provided when using formulas
            total_projector_distortions = len(projector_config.get('color_pipeline', [])) + len(projector_config.get('geometrical_pipeline', []))
            total_glass_distortions = len(glass_config.get('color_pipeline', [])) + len(glass_config.get('geometrical_pipeline', []))
            
            if total_projector_distortions == 0 and total_glass_distortions == 0:
                logging.warning("⚠️  No distortions specified in pipeline configurations!")
                logging.warning("   When using --dont_use_nn1, you should specify distortion pipelines.")
                logging.warning("   Example: --projector_color_pipeline 'vignetting:0.3,blur:0.2' --glass_geo_pipeline 'barrel:0.3'")
            else:
                logging.info(f"✅ Pipeline configurations: {total_projector_distortions} projector distortions, {total_glass_distortions} glass distortions")
        else:
            logging.info("Using AR system configuration from command line arguments")
    
    # Create AR system
    ar_system = ARSystem(projector_config, glass_config)
    
    # Create isolated generator for Mathematical Distortions on GPU
    # This guarantees that batch_size changes or Model Dropout do NOT alter the 
    # exact random values generated by AR-specific stochastic operations (like tracker noise).
    ar_generator = torch.Generator(device=device)
    ar_generator.manual_seed(args.seed)
    ar_system.generator = ar_generator
    logging.info("AR system created with configuration")
    
    # Print all 4 distortion pipelines
    logging.info("=" * 70)
    logging.info("AR SYSTEM DISTORTION PIPELINES")
    logging.info("=" * 70)
    logging.info(f"Projector Color Pipeline ({len(ar_system.projector.color_pipeline)} distortions):")
    if ar_system.projector.color_pipeline:
        for dist in ar_system.projector.color_pipeline:
            logging.info(f"  • {dist['type']}: {dist['strength']}")
    else:
        logging.info("  (empty)")
    
    logging.info(f"Projector Geometric Pipeline ({len(ar_system.projector.geometrical_pipeline)} distortions):")
    if ar_system.projector.geometrical_pipeline:
        for dist in ar_system.projector.geometrical_pipeline:
            logging.info(f"  • {dist['type']}: {dist['strength']}")
    else:
        logging.info("  (empty)")
    
    logging.info(f"Glass Color Pipeline ({len(ar_system.glass.color_pipeline)} distortions):")
    if ar_system.glass.color_pipeline:
        for dist in ar_system.glass.color_pipeline:
            logging.info(f"  • {dist['type']}: {dist['strength']}")
    else:
        logging.info("  (empty)")
    
    logging.info(f"Glass Geometric Pipeline ({len(ar_system.glass.geometrical_pipeline)} distortions):")
    if ar_system.glass.geometrical_pipeline:
        for dist in ar_system.glass.geometrical_pipeline:
            logging.info(f"  • {dist['type']}: {dist['strength']}")
    else:
        logging.info("  (empty)")
    logging.info("=" * 70)

    # Create experiment name
    try:
        base_experiment_name = create_experiment_name(args, projector_config, glass_config)
    except Exception as e:
        logging.error(f"Error creating experiment name: {e}")
        logging.error(f"Projector config: {projector_config}")
        logging.error(f"Glass config: {glass_config}")
        # Fallback to a simple experiment name
        base_experiment_name = f"NN2_{args.nn2_model_type}_e{args.epochs}_lr{args.lr:.0e}_bs{args.batch_size}_{args.loss_type}"

        # Add configuration flags
        if args.norm:
            base_experiment_name += "_norm"
        if args.use_amp:
            base_experiment_name += "_amp"
        base_experiment_name += f"_minlr{args.min_lr:.0e}"
        if args.seed != DEFAULT_SEED:
            base_experiment_name += f"_seed{args.seed}"
        
        # Add grid_sample configuration for fallback
        base_experiment_name += f"_grid-lib-{args.grid_sample_mode}"

        if args.comment:
            base_experiment_name += f"_{args.comment}"
    
    # Add freezing info to experiment name
    if args.freeze_net1 and args.freeze_net2:
        base_experiment_name += "_frozen_both"
    elif args.freeze_net1:
        base_experiment_name += "_frozen_net1"
    elif args.freeze_net2:
        base_experiment_name += "_frozen_net2"
    else:
        base_experiment_name += "_trainable_both"

    # Get experiment identifier
    experiment_id = get_next_experiment_id(NN2_RESULTS_DIR)
    experiment_identifier = f"#{experiment_id}-"

    # Create full experiment name with identifier for logging
    experiment_name = f"{experiment_identifier}{base_experiment_name}"

    log_nn2_experiment_setup(experiment_identifier, experiment_name, args.nn1_model_path)
    
    # Setup logging - use short filename
    os.makedirs(f'{NN2_RESULTS_DIR}/logs', exist_ok=True)
    log_file = f"{NN2_RESULTS_DIR}/logs/{experiment_identifier}NN2.log"
    
    # Create a custom class to tee output to both console and file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Write command to log file first (before any logging setup)
    command_str = " ".join(sys.argv)
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMMAND\n")
        f.write("=" * 80 + "\n")
        f.write(command_str + "\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
    
    # Now open file in append mode for the rest of the logging
    log_file_handle = open(log_file, 'a')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(original_stdout, log_file_handle)
    sys.stderr = TeeOutput(original_stderr, log_file_handle)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # Append mode
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting NN2 training: {experiment_name}")
    logging.info("=" * 80)
    logging.info("HYPERPARAMETER SUMMARY")
    logging.info("=" * 80)
    
    # Log grid_sample configuration
    logging.info(f"Grid Sample Configuration: LIBRARY (F.grid_sample) with {args.grid_sample_mode} interpolation")
    logging.info("=" * 80)
    
    # Log all hyperparameters using dedicated functions
    log_nn2_model_parameters(args)
    log_nn2_training_parameters(args)
    log_nn2_loss_parameters(args)
    log_nn2_optimizer_parameters(args)
    log_nn2_scheduler_parameters(args)
    log_nn2_training_control_parameters(args)
    log_nn2_data_parameters(args, GLOBAL_ROI_COORDS)
    
    # Extract pipeline lists from config dictionaries for logging
    # pipeline_to_str expects lists, not config dictionaries
    projector_pipelines = []
    if projector_config.get('color_pipeline'):
        projector_pipelines.extend(projector_config['color_pipeline'])
    if projector_config.get('geometrical_pipeline'):
        projector_pipelines.extend(projector_config['geometrical_pipeline'])
    
    glass_pipelines = []
    if glass_config.get('color_pipeline'):
        glass_pipelines.extend(glass_config['color_pipeline'])
    if glass_config.get('geometrical_pipeline'):
        glass_pipelines.extend(glass_config['geometrical_pipeline'])
    
    # Log configuration parameters using dedicated function
    log_nn2_configuration_parameters(args, projector_pipelines, glass_pipelines, device)

    logging.info("=" * 80)
    
    # Create datasets and data loaders
    logging.info("=" * 60)
    logging.info("DATA TRANSFORMATION SETUP")
    logging.info("=" * 60)

    # Basic transforms for ROI analysis (if needed)
    basic_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])


    transform_list = [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ]
    
    # Add data augmentation if specified
    if hasattr(args, 'augment') and args.augment:
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        transform_list.insert(1, transforms.RandomRotation(degrees=10))
        transform_list.insert(1, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        logging.info("✅ Data augmentation enabled:")
        logging.info("  - Random horizontal flip (p=0.5)")
        logging.info("  - Random rotation (±10 degrees)")
        logging.info("  - Color jitter (brightness, contrast, saturation, hue)")
    else:
        logging.info("Data augmentation disabled")
    
    # Add normalization if specified
    if args.norm:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        logging.info("✅ ImageNet normalization enabled")
        logging.info(f"  - Mean: [0.485, 0.456, 0.406]")
        logging.info(f"  - Std: [0.229, 0.224, 0.225]")
    else:
        logging.info("ImageNet normalization disabled")
    
    transform = transforms.Compose(transform_list)
    logging.info(f"Final transform pipeline: {len(transform_list)} transforms")
    
    # Check if using real dataset mode
    if args.use_real_dataset:
        logging.info("=" * 60)
        logging.info("REAL DATASET MODE: Using pre-existing clean/distorted pairs")
        logging.info("=" * 60)
        
        # Validate real dataset directories
        if not args.clean_train_dir:
            raise ValueError("Real dataset mode requires --clean_train_dir and --distorted_train_dir")
        
        logging.info(f"Clean train directory: {args.clean_train_dir}")
        logging.info(f"Clean test directory: {args.clean_test_dir}")

        # Create real datasets
        train_dataset = RealDataset(args.clean_train_dir, args.distorted_train_dir, transform, 'train', base_seed=args.seed)
        test_dataset = RealDataset(args.clean_test_dir, args.distorted_test_dir, transform, 'test', base_seed=args.seed)
        
        # Create validation split from training data
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        # Use explicit generator for reproducibility
        g_split = torch.Generator()
        g_split.manual_seed(args.seed)
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=g_split
        )
        
        logging.info(f"Real dataset loaded:")
        logging.info(f"  Training samples: {len(train_dataset)}")
        logging.info(f"  Validation samples: {len(val_dataset)}")
        logging.info(f"  Test samples: {len(test_dataset)}")
        logging.info(f"  💡 Training uses: INPUT(clean) → TARGET(real_distorted)")
        
        # Skip the original dataset creation logic
        full_dataset = None
    else:
        # Original AR system mode
        # Create clean image datasets
        logging.info("Creating clean image datasets...")
        if isinstance(args.image_dirs, list):
            datasets = [CleanImageDataset(d, transform=transform, base_seed=args.seed) for d in args.image_dirs]
            from torch.utils.data import ConcatDataset
            full_dataset = ConcatDataset(datasets)
            logging.info(f"Created concatenated dataset from {len(args.image_dirs)} directories")
        else:
            full_dataset = CleanImageDataset(args.image_dirs, transform=transform, base_seed=args.seed)
            logging.info(f"Created single dataset from directory: {args.image_dirs}")
    
    # Train-validation-test split
    # Apply dataset fraction if specified (deterministically use first portion)
    if args.dataset_fraction < 1.0:
        if args.dataset_fraction <= 0.0:
            raise ValueError(f"Dataset fraction must be > 0.0, got {args.dataset_fraction}")
        if args.dataset_fraction > 1.0:
            raise ValueError(f"Dataset fraction must be <= 1.0, got {args.dataset_fraction}")
        
        logging.info(f"Using {args.dataset_fraction*100:.1f}% of the dataset ({args.dataset_fraction:.2f} fraction) - deterministically selecting first portion")
        subset_size = int(args.dataset_fraction * len(full_dataset))
        if subset_size == 0:
            raise ValueError(f"Dataset fraction {args.dataset_fraction} results in 0 samples. Increase the fraction or use a larger dataset.")
        
        # Deterministically select first portion (not random)
        from torch.utils.data import Subset
        indices = list(range(subset_size))
        full_dataset = Subset(full_dataset, indices)
        logging.info(f"Dataset reduced to {len(full_dataset)} samples (using first {subset_size} samples)")
    
    logging.info("Creating train/validation/test splits...")
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Create a dedicated generator for reproducibility in data splitting and loading
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=g
    )

    logging.info(f"Dataset split created:")
    logging.info(f"  - Training: {train_size} samples ({train_size/len(full_dataset)*100:.1f}%)")
    logging.info(f"  - Validation: {val_size} samples ({val_size/len(full_dataset)*100:.1f}%)")
    logging.info(f"  - Test: {test_size} samples ({test_size/len(full_dataset)*100:.1f}%)")
    
    if args.dataset_fraction < 1.0:
        logging.info(f"  - Dataset fraction used: {args.dataset_fraction:.2f} ({args.dataset_fraction*100:.1f}%)")
    
    # Create data loaders
    logging.info("=" * 60)
    logging.info("DATA LOADER SETUP")
    logging.info("=" * 60)
    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    log_nn2_data_loader_info(train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, args, full_dataset)
    
    # Use zero mask from NN1 model (already loaded from checkpoint)
    zero_mask = None
    if hasattr(nn1_model, 'zero_mask') and nn1_model.zero_mask is not None:
        zero_mask = nn1_model.zero_mask
        logging.info("=" * 60)
        logging.info("ZERO MASK SETUP")
        logging.info("=" * 60)
        logging.info(f"✅ Using zero_mask from NN1 model with shape: {zero_mask.shape}")
        logging.info(f"Zero mask non-zero pixels: {torch.sum(zero_mask > 0).item()}")
        logging.info(f"Zero mask zero pixels: {torch.sum(zero_mask == 0).item()}")
    else:
        logging.info("No zero_mask found in NN1 model")
    
    # Analyze ROI if enabled (after zero mask is available)
    if args.use_roi_extraction:
        logging.info("=" * 60)
        logging.info("ROI ANALYSIS")
        logging.info("=" * 60)

        # Extract ROI from zero mask - REQUIRED, no fallbacks allowed
        if zero_mask is not None:
            logging.info("Finding largest white square inside zero mask (valid = 1/white, invalid = 0/black)...")
            
            def analyze_zero_mask_roi(zero_mask_tensor):
                """Extract ROI coordinates from zero mask"""
                import torch
                import numpy as np
                
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
                    raise RuntimeError("Zero mask contains no valid pixels (all pixels are 1)! Cannot proceed with ROI extraction.")
                
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
                    # print(f"DEBUG: Finding largest square in {h}x{w} mask")
                    # print(f"DEBUG: Valid pixels count: {np.sum(binary_mask)}")
                    
                    # Show a sample of the mask for debugging
                    # print(f"DEBUG: Mask sample (top-left 10x10):")
                    sample = binary_mask[:min(10, h), :min(10, w)]
                    for row in sample:
                        print(''.join(['1' if x else '0' for x in row]))
                    
                    # Dynamic programming approach
                    # dp[i][j] = side length of largest square with bottom-right corner at (i,j)
                    dp = np.zeros((h, w), dtype=int)
                    max_size = 0
                    best_bottom_right = (0, 0)
                    
                    # Fill the dp table
                    for i in range(h):
                        for j in range(w):
                            if binary_mask[i][j]:  # Current cell is valid (part of black region)
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
                                    print(f"DEBUG: New max square size {max_size} at bottom-right ({i}, {j})")
                            else:
                                dp[i][j] = 0
                    
                    # print(f"DEBUG: Final maximum square size: {max_size}")
                    # print(f"DEBUG: Best bottom-right corner: {best_bottom_right}")
                    
                    if max_size == 0:
                        raise RuntimeError("Could not find any valid square in zero mask! Cannot proceed with ROI extraction.")
                    
                    # Calculate top-left corner from bottom-right corner and size
                    bottom_right_y, bottom_right_x = best_bottom_right
                    top_left_y = bottom_right_y - max_size + 1
                    top_left_x = bottom_right_x - max_size + 1
                    
                    # print(f"DEBUG: Calculated top-left: ({top_left_y}, {top_left_x})")
                    # print(f"DEBUG: Square covers: ({top_left_y},{top_left_x}) to ({bottom_right_y},{bottom_right_x})")
                    
                    # Verify the square is entirely within valid region
                    square_region = binary_mask[top_left_y:bottom_right_y+1, top_left_x:bottom_right_x+1]
                    all_valid = np.all(square_region)
                    # print(f"DEBUG: Square region shape: {square_region.shape}")
                    # print(f"DEBUG: All pixels in square are valid: {all_valid}")
                    
                    if not all_valid:
                        print(f"DEBUG: ERROR - Square contains some invalid pixels!")
                        print(f"DEBUG: Invalid pixels in square: {np.sum(~square_region)}")
                        # Show the problematic region
                        print(f"DEBUG: Square region:")
                        for row in square_region:
                            print(''.join(['1' if x else '0' for x in row]))
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
                
                logging.info(f"Largest square analysis:")
                logging.info(f"  - Mask shape: {mask_np.shape}")
                logging.info(f"  - Valid pixels (mask=1): {np.sum(valid_pixels)}")
                logging.info(f"  - Invalid pixels (mask=0): {np.sum(mask_np == 0)}")
                logging.info(f"  - Largest square size: {size}x{size} pixels")
                logging.info(f"  - Square top-left: ({top_left_y}, {top_left_x})")
                logging.info(f"  - Square bottom-right: ({y_max}, {x_max})")
                logging.info(f"  - ROI bounds: y({y_min}-{y_max}), x({x_min}-{x_max})")
                
                # Return in format (x_min, y_min, x_max, y_max) as expected by visualization code
                # print(f"DEBUG: Returning coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                return (x_min, y_min, x_max, y_max)
            
            try:
                GLOBAL_ROI_COORDS = analyze_zero_mask_roi(zero_mask)
                if GLOBAL_ROI_COORDS is not None:
                    logging.info(f"✅ ROI analysis complete. Optimal coordinates: {GLOBAL_ROI_COORDS}")
                else:
                    raise RuntimeError("ROI analysis returned None! Cannot proceed with ROI extraction.")
            except Exception as e:
                logging.error(f"❌ ROI analysis failed: {e}")
                raise RuntimeError(f"ROI extraction failed and is required. Cannot proceed: {e}")
        else:
            logging.warning("❌ No zero mask available for ROI analysis!")
            logging.info("🔍 Creating zero mask by analyzing AR system outputs on 100 sample images...")
            
            # Create zero mask by analyzing AR system outputs
            def create_zero_mask_from_ar_system_outputs(ar_system, data_loader, device, num_samples=1, image_size=256):
                """Create zero mask from a single AR system sample: non-zero = valid (white)."""
                import torch
                import numpy as np
                from torchvision import transforms
                from PIL import Image
                import os
                
                invalid_pixel_counts = None
                total_samples = 0
                saved_vis_count = 0
                vis_dir = os.path.join(NN2_RESULTS_DIR, 'zero_mask_samples')
                os.makedirs(vis_dir, exist_ok=True)
                
                logging.info(f"Creating zero mask using a fully white {image_size}x{image_size} image...")
                
                with torch.no_grad():
                    # Create fully white input [C, H, W]
                    H, W = image_size, image_size
                    single_image_tensor = torch.ones(3, H, W)
                    
                    # Apply distortions stage-by-stage (projector color -> projector geo -> glass color -> glass geo)
                    distorted_tensor = apply_gradient_friendly_distortions(single_image_tensor, ar_system)

                            # Save a few visual samples as a single composite with all stages
                    if saved_vis_count < 1:
                                try:
                                    from types import SimpleNamespace
                                    # Stage 0: input
                                    stage0 = single_image_tensor
                                    # Stage 1: after projector color only
                                    proj_color_only = SimpleNamespace(color_pipeline=getattr(ar_system.projector, 'color_pipeline', []),
                                                                      geometrical_pipeline=[])
                                    stage1 = apply_projector_distortions(stage0.clone(), proj_color_only)
                                    # Stage 2: after projector geometrical
                                    proj_geo_only = SimpleNamespace(color_pipeline=[],
                                                                    geometrical_pipeline=getattr(ar_system.projector, 'geometrical_pipeline', []))
                                    stage2 = apply_projector_distortions(stage1.clone(), proj_geo_only)
                                    # Stage 3: after glass color
                                    glass_color_only = SimpleNamespace(color_pipeline=getattr(ar_system.glass, 'color_pipeline', []),
                                                                       geometrical_pipeline=[])
                                    stage3 = apply_glass_distortions(stage2.clone(), glass_color_only)
                                    # Stage 4: after glass geometrical (final)
                                    glass_geo_only = SimpleNamespace(color_pipeline=[],
                                                                     geometrical_pipeline=getattr(ar_system.glass, 'geometrical_pipeline', []))
                                    stage4 = apply_glass_distortions(stage3.clone(), glass_geo_only)

                                    # Convert stages to numpy images
                                    def to_uint8_np(t):
                                        t = torch.clamp(t, 0, 1).detach().cpu().numpy()  # [C, H, W]
                                        t = (t.transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, C]
                                        return t

                                    imgs = [to_uint8_np(x) for x in [stage0, stage1, stage2, stage3, stage4]]
                                    titles = [
                                        "Input (White)",
                                        "Proj Color",
                                        "Proj Geo",
                                        "Glass Color",
                                        "Glass Geo"
                                    ]

                                    # Build a labeled composite: add a title strip above each stage
                                    import numpy as np
                                    from PIL import ImageDraw, ImageFont

                                    h, w, c = imgs[0].shape
                                    title_height = 28
                                    spacing = 0
                                    total_width = w * len(imgs) + spacing * (len(imgs) - 1)
                                    total_height = title_height + h

                                    # Create white background composite
                                    composite = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
                                    draw = ImageDraw.Draw(composite)
                                    try:
                                        font = ImageFont.load_default()
                                    except:
                                        font = None

                                    # Helper to measure text size robustly across PIL versions
                                    def _measure_text(draw_ctx, text_str, font_obj):
                                        try:
                                            bbox = draw_ctx.textbbox((0, 0), text_str, font=font_obj)
                                            return bbox[2] - bbox[0], bbox[3] - bbox[1]
                                        except Exception:
                                            try:
                                                return font_obj.getsize(text_str)
                                            except Exception:
                                                return (max(1, len(text_str) * 6), 12)

                                    # Paste each image and draw its title
                                    for idx, (img_np, title) in enumerate(zip(imgs, titles)):
                                        x_offset = idx * (w + spacing)
                                        # Paste image below the title strip
                                        composite.paste(Image.fromarray(img_np), (x_offset, title_height))
                                        # Draw centered title
                                        text_w, text_h = _measure_text(draw, title, font)
                                        text_x = x_offset + (w - text_w) // 2
                                        text_y = (title_height - text_h) // 2
                                        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)

                                    # Save single composite file
                                    composite_path = os.path.join(vis_dir, f"stages_{total_samples:04d}.png")
                                    composite.save(composite_path)
                                    saved_vis_count += 1
                                    logging.info(f"Saved zero mask composite #{saved_vis_count}: {composite_path}")
                                except Exception as e:
                                    logging.warning(f"Failed to save zero mask composite image: {e}")
                    
                    # Convert back to numpy for analysis
                    ar_output_np = distorted_tensor.numpy()  # [C, H, W]

                    # Convert to grayscale
                    if ar_output_np.shape[0] == 3:
                        gray = np.mean(ar_output_np, axis=0)
                    else:
                        gray = ar_output_np[0]

                    # Valid (white) where output is non-zero; use small epsilon
                    zero_mask = (gray > 0.001).astype(np.float32)

                    logging.info("✅ Created zero mask from white image (white=valid non-zero regions)")
                    logging.info(f"  - Valid pixels (mask=1): {np.sum(zero_mask == 1)}")
                    logging.info(f"  - Invalid pixels (mask=0): {np.sum(zero_mask == 0)}")
                    logging.info(f"  - Mask shape: {zero_mask.shape}")

                    zero_mask_tensor = torch.from_numpy(zero_mask).float()
                    return zero_mask_tensor
            
            try:
                # Create zero mask from AR system outputs (optimized with fewer samples)
                zero_mask = create_zero_mask_from_ar_system_outputs(ar_system, train_loader, device, num_samples=50, image_size=args.image_size)
                
                # Now analyze the created zero mask
                logging.info("🔍 Analyzing created zero mask...")
                
                def analyze_zero_mask_roi(zero_mask_tensor):
                    """Extract ROI coordinates from zero mask"""
                    import torch
                    import numpy as np
                    
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
                        raise RuntimeError("Created zero mask contains no valid pixels (all pixels are 1)! Cannot proceed with ROI extraction.")
                    
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
                        # print(f"DEBUG: Finding largest square in {h}x{w} mask")
                        # print(f"DEBUG: Valid pixels count: {np.sum(binary_mask)}")
                        
                        # Show a sample of the mask for debugging
                        # print(f"DEBUG: Mask sample (top-left 10x10):")
                        sample = binary_mask[:min(10, h), :min(10, w)]
                        for row in sample:
                            print(''.join(['1' if x else '0' for x in row]))
                        
                        # Dynamic programming approach
                        # dp[i][j] = side length of largest square with bottom-right corner at (i,j)
                        dp = np.zeros((h, w), dtype=int)
                        max_size = 0
                        best_bottom_right = (0, 0)
                        
                        # Fill the dp table
                        for i in range(h):
                            for j in range(w):
                                if binary_mask[i][j]:  # Current cell is valid (part of black region)
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
                                        # print(f"DEBUG: New max square size {max_size} at bottom-right ({i}, {j})")
                                else:
                                    dp[i][j] = 0
                        
                        print(f"DEBUG: Final maximum square size: {max_size}")
                        print(f"DEBUG: Best bottom-right corner: {best_bottom_right}")
                        
                        if max_size == 0:
                            raise RuntimeError("Could not find any valid square in zero mask! Cannot proceed with ROI extraction.")
                        
                        # Calculate top-left corner from bottom-right corner and size
                        bottom_right_y, bottom_right_x = best_bottom_right
                        top_left_y = bottom_right_y - max_size + 1
                        top_left_x = bottom_right_x - max_size + 1
                        
                        print(f"DEBUG: Calculated top-left: ({top_left_y}, {top_left_x})")
                        print(f"DEBUG: Square covers: ({top_left_y},{top_left_x}) to ({bottom_right_y},{bottom_right_x})")
                        
                        # Verify the square is entirely within valid region
                        square_region = binary_mask[top_left_y:bottom_right_y+1, top_left_x:bottom_right_x+1]
                        all_valid = np.all(square_region)
                        print(f"DEBUG: Square region shape: {square_region.shape}")
                        print(f"DEBUG: All pixels in square are valid: {all_valid}")
                        
                        if not all_valid:
                            print(f"DEBUG: ERROR - Square contains some invalid pixels!")
                            print(f"DEBUG: Invalid pixels in square: {np.sum(~square_region)}")
                            # Show the problematic region
                            print(f"DEBUG: Square region:")
                            for row in square_region:
                                print(''.join(['1' if x else '0' for x in row]))
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






import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample
import segmentation_models_pytorch as smp
import os
import sys
from pytorch_msssim import ssim
import lpips
from kornia.color import rgb_to_lab

# Import CompenNetPlusplus

from compennetplusplus import CompenNetPlusplus
COMPENNETPLUSPLUS_AVAILABLE = True

# Import vendored NAFNet (megvii-research/NAFNet, ECCV 2022) baseline.
try:
    from nafnet_arch import NAFNet, build_nafnet_from_preset, NAFNET_PRESETS
    NAFNET_AVAILABLE = True
except Exception:
    NAFNET_AVAILABLE = False

# =============================================================================
# COLOR SPACE CONVERSION UTILITIES
# =============================================================================

def rgb_to_hsl(rgb):
    """
    Convert RGB to HSL color space.
    
    Args:
        rgb: Tensor of shape [..., 3] with values in [0, 1]
        
    Returns:
        hsl: Tensor of shape [..., 3] with H in [0, 1], S in [0, 1], L in [0, 1]
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    max_c = torch.max(torch.max(r, g), b)
    min_c = torch.min(torch.min(r, g), b)
    delta = max_c - min_c
    
    # Lightness
    l = (max_c + min_c) / 2.0
    
    # Saturation
    s = torch.zeros_like(l)
    mask = delta != 0
    s[mask] = delta[mask] / (1 - torch.abs(2 * l[mask] - 1) + 1e-7)
    
    # Hue
    h = torch.zeros_like(l)
    
    # Red is max
    mask_r = (max_c == r) & (delta != 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-7)) % 6
    
    # Green is max
    mask_g = (max_c == g) & (delta != 0)
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-7)) + 2
    
    # Blue is max
    mask_b = (max_c == b) & (delta != 0)
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-7)) + 4
    
    h = h / 6.0  # Normalize to [0, 1]
    h = h % 1.0  # Ensure it's in [0, 1]
    
    return torch.stack([h, s, l], dim=-1)


def hsl_to_rgb(hsl):
    """
    Convert HSL to RGB color space.
    
    Args:
        hsl: Tensor of shape [..., 3] with H in [0, 1], S in [0, 1], L in [0, 1]
        
    Returns:
        rgb: Tensor of shape [..., 3] with values in [0, 1]
    """
    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    
    c = (1 - torch.abs(2 * l - 1)) * s
    h_prime = h * 6.0
    x = c * (1 - torch.abs((h_prime % 2) - 1))
    
    m = l - c / 2.0
    
    rgb = torch.zeros_like(hsl)
    
    # 0 <= h < 1
    mask0 = (h_prime >= 0) & (h_prime < 1)
    rgb[mask0] = torch.stack([c[mask0], x[mask0], torch.zeros_like(c[mask0])], dim=-1)
    
    # 1 <= h < 2
    mask1 = (h_prime >= 1) & (h_prime < 2)
    rgb[mask1] = torch.stack([x[mask1], c[mask1], torch.zeros_like(c[mask1])], dim=-1)
    
    # 2 <= h < 3
    mask2 = (h_prime >= 2) & (h_prime < 3)
    rgb[mask2] = torch.stack([torch.zeros_like(c[mask2]), c[mask2], x[mask2]], dim=-1)
    
    # 3 <= h < 4
    mask3 = (h_prime >= 3) & (h_prime < 4)
    rgb[mask3] = torch.stack([torch.zeros_like(c[mask3]), x[mask3], c[mask3]], dim=-1)
    
    # 4 <= h < 5
    mask4 = (h_prime >= 4) & (h_prime < 5)
    rgb[mask4] = torch.stack([x[mask4], torch.zeros_like(c[mask4]), c[mask4]], dim=-1)
    
    # 5 <= h < 6
    mask5 = (h_prime >= 5) & (h_prime < 6)
    rgb[mask5] = torch.stack([c[mask5], torch.zeros_like(c[mask5]), x[mask5]], dim=-1)
    
    rgb = rgb + m.unsqueeze(-1)
    
    return rgb

# from external.DRSformer.basicsr.models.archs.DRSformer_arch import DRSformer as DRSFormerBackbone

# =============================================================================
# GRID SAMPLE CONFIGURATION
# =============================================================================





# =============================================================================
# BASE CLASSES - Extensible Architectureivate env venv
# =============================================================================

class BaseModel(nn.Module):
    """Base class for all transformation models"""
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        """Transform input data - to be implemented by subclasses"""
        raise NotImplementedError


class BaseApproach(nn.Module):
    """Base class for all transformation approaches"""
    def __init__(self, model, zero_mask=None, grid_sample_mode='bilinear'):
        super(BaseApproach, self).__init__()
        self.model = model
        # Register zero mask as buffer to preserve zero areas
        self.register_buffer('zero_mask', zero_mask)
        # Grid sample configuration
        self.grid_sample_mode = grid_sample_mode
        # Initialize debug counter
        # self.debug_image_count = 0

    def create_coordinate_grid(self, batch_size, height, width, device):
        """Create normalized coordinate grid [0, 1]"""
        y_coords = torch.linspace(0, 1, height, device=device)
        x_coords = torch.linspace(0, 1, width, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return grid

    def apply_zero_mask(self, output):
        """Apply zero mask to preserve zero areas in output"""
        if self.zero_mask is not None:
            batch_size, channels, height, width = output.shape
            # Expand mask to match output dimensions [B, C, H, W]
            expanded_mask = self.zero_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, -1, -1)
            # Apply mask: keep valid areas (where mask=0), zero out invalid areas (where mask=1)
            output = output * (1 - expanded_mask)
        return output
    
    def save_debug_images(self, output_before_mask, output_after_mask, expanded_mask, batch_size):
        """Save debug images showing before/after masking and mask itself"""
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        # Create debug directory if it doesn't exist
        debug_dir = "debug_zero_mask"
        os.makedirs(debug_dir, exist_ok=True)

        # Only save first 10 samples to avoid too many files
        max_debug_images = 10
        if self.debug_image_count >= max_debug_images:
            return

        # Process each sample in the batch
        for batch_idx in range(min(batch_size, max_debug_images - self.debug_image_count)):
            sample_idx = self.debug_image_count

            # Convert tensors to numpy for visualization
            # Take first channel for RGB images and detach to avoid grad issues
            before_img = output_before_mask[batch_idx, 0].detach().cpu().numpy()
            after_img = output_after_mask[batch_idx, 0].detach().cpu().numpy()
            mask_img = expanded_mask[batch_idx, 0].detach().cpu().numpy()

            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Prediction before mask
            im1 = axes[0].imshow(before_img, cmap='viridis', vmin=before_img.min(), vmax=before_img.max())
            axes[0].set_title(f'Prediction Before Mask\nSample {sample_idx}')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], shrink=0.8)

            # Prediction after mask
            im2 = axes[1].imshow(after_img, cmap='viridis', vmin=after_img.min(), vmax=after_img.max())
            axes[1].set_title(f'Prediction After Mask\nSample {sample_idx}')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], shrink=0.8)

            # Mask itself
            im3 = axes[2].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f'Zero Mask\nSample {sample_idx}')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], shrink=0.8)

            plt.tight_layout()

            # Save the figure
            filename = f"{debug_dir}/zero_mask_debug_sample_{sample_idx:02d}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"DEBUG: Saved zero mask debug images for sample {sample_idx} to {filename}")

            self.debug_image_count += 1

            # Stop if we've reached the limit
            if self.debug_image_count >= max_debug_images:
                break

    def forward(self, x):
        """Apply transformation approach - to be implemented by subclasses"""
        raise NotImplementedError


# =============================================================================
# MODEL IMPLEMENTATIONS
# =============================================================================

class NeuralModel(BaseModel):
    """Neural network-based transformation model"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, complexity='medium'):
        super(NeuralModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if complexity == 'simple':
            self.transform_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )
        elif complexity == 'medium':
            self.transform_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )
        elif complexity == 'complex':
            self.transform_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown complexity level: {complexity}. Neural models support: simple, medium, complex")
        
        # Initialize for identity mapping (x,y → x,y)
        # self._initialize_identity()
    
    # def _initialize_identity(self):
    #     """Initialize weights so that the network outputs the same coordinates as input (x,y → x,y)"""
    #     if self.input_dim == 2 and self.output_dim == 2:
    #         # We want the network to output the same coordinates as input
    #         # This means: f(x,y) = (x,y)
    #         # For a network with sigmoid output, we need to ensure that when input is [0,1],
    #         # the output is also [0,1]. Since sigmoid maps to [0,1], we need the network
    #         # to learn the identity function within the sigmoid activation.
            
    #         # Initialize all layers with small weights so the network starts close to identity
    #         for module in self.transform_net.modules():
    #             if isinstance(module, nn.Linear):
    #                 with torch.no_grad():
    #                     # Initialize with very small weights so the network starts close to identity
    #                     nn.init.normal_(module.weight, mean=0.0, std=0.01)
    #                     nn.init.zeros_(module.bias)
                        
    #                     # For the final layer, we want it to output the input coordinates
    #                     # Since we have sigmoid activation, we need to account for that
    #                     if module.out_features == 2:  # Final layer
    #                         # Initialize to pass through the input with minimal transformation
    #                         # The sigmoid will map the output to [0,1] range
    #                         module.weight.data = torch.eye(2) * 0.1  # Small identity-like weights
    #                         module.bias.data = torch.zeros(2)
    
    def forward(self, x):
        """Transform input data using neural network"""
        batch_size, height, width, channels = x.shape
        # Use contiguous().view() to preserve gradients instead of reshape
        reshaped_input = x.contiguous().view(-1, channels)
        transformed_output = self.transform_net(reshaped_input)
        # Use contiguous().view() to preserve gradients instead of reshape
        transformed_output = transformed_output.contiguous().view(batch_size, height, width, self.output_dim)
        return transformed_output


class ParametricModel(BaseModel):
    """Parametric transformation with configurable complexity levels

    Complexity levels:
    - simple: Direct matrix multiplication (output = input @ matrix)
    - medium: Two-layer transformation with ReLU activation
    - complex: Three-layer transformation with residual connections
    - complex_with_hsl: Three-layer with HSL color space conversion (for color processing)
    - fisheye: Traditional fisheye distortion correction (coordinates only)
    """
    def __init__(self, input_dim=2, output_dim=2, complexity='simple', *args, **kwargs):
        super(ParametricModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.complexity = complexity
        # Optional configurable hidden dimensions for medium/complex variants
        # Accept either a single hidden_dim (applied to all intermediate layers)
        # or separate hidden_dim1/hidden_dim2 for complex variant.
        hidden_dim_arg = kwargs.pop('hidden_dim', None)
        hidden_dim1_arg = kwargs.pop('hidden_dim1', None)
        hidden_dim2_arg = kwargs.pop('hidden_dim2', None)

        if complexity == 'simple':
            # Simple matrix transformation: output = input @ matrix
            self.transformation_matrix = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

        elif complexity == 'medium':
            # Medium complexity: Two-layer transformation with non-linearity
            hidden_dim = hidden_dim_arg if hidden_dim_arg is not None else max(input_dim, output_dim)
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.ReLU()

        elif complexity == 'complex':
            # Complex: 2-hidden-layer MLP with residual connection (2 -> H -> H -> 2)
            hidden_dim = hidden_dim_arg if hidden_dim_arg is not None else max(input_dim, 16)
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.ReLU()

            # Residual connection (map input directly to output)
            if input_dim == output_dim:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Linear(input_dim, output_dim)

        elif complexity == 'complexer':
            # Complexer: Complex MLP + small CNN for spatial feature extraction
            # This combines parametric transformation with convolutional spatial processing
            hidden_dim = hidden_dim_arg if hidden_dim_arg is not None else max(input_dim, 16)
            
            # MLP branch (same as complex)
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.ReLU()
            
            # Small CNN branch for spatial feature extraction
            # Note: This will process spatial neighborhoods in the forward pass
            # CNN processes input_dim channels
            cnn_hidden = max(8, hidden_dim // 4)  # Small CNN hidden dim
            self.conv1 = nn.Conv2d(input_dim, cnn_hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(cnn_hidden, output_dim, kernel_size=1)  # 1x1 conv for output
            self.conv_activation = nn.ReLU()
            
            # Fusion layer to combine MLP and CNN outputs
            self.fusion = nn.Linear(output_dim * 2, output_dim)
            
            # Residual connection (map input directly to output)
            if input_dim == output_dim:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Linear(input_dim, output_dim)

        elif complexity == 'complex_with_hsl':
            # Complex with HSL: For color processing tasks (Approach 4)
            # Expects input_dim=5 (x, y, R, G, B) and output_dim=3 (R, G, B)
            if input_dim != 5 or output_dim != 3:
                raise ValueError(f"complex_with_hsl requires input_dim=5 and output_dim=3, got {input_dim} -> {output_dim}")
            
            hidden_dim = hidden_dim_arg if hidden_dim_arg is not None else 32
            
            # Process coordinates (x, y) separately
            self.coord_layer = nn.Linear(2, hidden_dim // 2)
            
            # Process HSL colors
            self.hsl_layer1 = nn.Linear(3, hidden_dim // 2)
            
            # Combined processing
            self.combined_layer1 = nn.Linear(hidden_dim, hidden_dim)
            self.combined_layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, 3)  # Output HSL
            
            self.activation = nn.ReLU()

        elif complexity == 'fisheye':
            # Traditional fisheye distortion correction (coordinates only)
            if input_dim != 2 or output_dim != 2:
                raise ValueError(f"Fisheye complexity requires 2D input/output, got {input_dim}D -> {output_dim}D")
            self.k1 = nn.Parameter(torch.tensor(0.1))
            self.k2 = nn.Parameter(torch.tensor(0.05))
            self.k3 = nn.Parameter(torch.tensor(0.01))
            self.cx = nn.Parameter(torch.tensor(0.5))
            self.cy = nn.Parameter(torch.tensor(0.5))
            self.scale = nn.Parameter(torch.tensor(1.0))

        else:
            raise ValueError(f"Unknown complexity level: {complexity}. Choose from: simple, medium, complex, complex_with_hsl, fisheye")
        
        # Initialize for identity mapping (x,y → x,y) or appropriate initialization
        self._initialize_identity()
    
    def _initialize_identity(self):
        """Initialize weights so that the network outputs the same coordinates as input (x,y → x,y)"""
        if self.input_dim == 2 and self.output_dim == 2:
            if self.complexity == 'simple':
                # For simple matrix transformation, initialize to identity
                with torch.no_grad():
                    self.transformation_matrix.data = torch.eye(2)
            
            elif self.complexity == 'medium':
                # Initialize so that the network passes through input coordinates
                with torch.no_grad():
                    # Initialize layer1 with small weights
                    nn.init.normal_(self.layer1.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.layer1.bias)
                    # Initialize layer2 to pass through input (identity-like)
                    self.layer2.weight.data = torch.eye(2) * 0.1
                    nn.init.zeros_(self.layer2.bias)
            
            elif self.complexity == 'complex':
                # Initialize for stable start near identity using the residual path
                with torch.no_grad():
                    nn.init.normal_(self.layer1.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.layer1.bias)
                    nn.init.normal_(self.layer2.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.layer2.bias)
                    # Make the final projection initially zero so output ~= residual(input)
                    nn.init.zeros_(self.layer3.weight)
                    nn.init.zeros_(self.layer3.bias)
                    # Initialize residual if it's a learned projection
                    if hasattr(self, 'residual') and isinstance(self.residual, nn.Linear):
                        if self.residual.in_features == self.residual.out_features:
                            self.residual.weight.data = torch.eye(self.residual.out_features)
                            nn.init.zeros_(self.residual.bias)
                        else:
                            nn.init.normal_(self.residual.weight, mean=0.0, std=0.01)
                            nn.init.zeros_(self.residual.bias)
            elif self.complexity == 'complexer':
                # Initialize for stable start near identity using the residual path
                with torch.no_grad():
                    # MLP initialization (same as complex)
                    nn.init.normal_(self.layer1.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.layer1.bias)
                    nn.init.normal_(self.layer2.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.layer2.bias)
                    nn.init.zeros_(self.layer3.weight)
                    nn.init.zeros_(self.layer3.bias)
                    
                    # CNN initialization (small weights for stability)
                    nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.conv1.bias)
                    nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.conv2.bias)
                    nn.init.zeros_(self.conv3.weight)
                    nn.init.zeros_(self.conv3.bias)
                    
                    # Fusion layer initialization
                    nn.init.normal_(self.fusion.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.fusion.bias)
                    
                    # Initialize residual if it's a learned projection
                    if hasattr(self, 'residual') and isinstance(self.residual, nn.Linear):
                        if self.residual.in_features == self.residual.out_features:
                            self.residual.weight.data = torch.eye(self.residual.out_features)
                            nn.init.zeros_(self.residual.bias)
                        else:
                            nn.init.normal_(self.residual.weight, mean=0.0, std=0.01)
                            nn.init.zeros_(self.residual.bias)
            
            elif self.complexity == 'complex_with_hsl':
                # Initialize for HSL processing
                with torch.no_grad():
                    nn.init.xavier_uniform_(self.coord_layer.weight)
                    nn.init.zeros_(self.coord_layer.bias)
                    nn.init.xavier_uniform_(self.hsl_layer1.weight)
                    nn.init.zeros_(self.hsl_layer1.bias)
                    nn.init.xavier_uniform_(self.combined_layer1.weight)
                    nn.init.zeros_(self.combined_layer1.bias)
                    nn.init.xavier_uniform_(self.combined_layer2.weight)
                    nn.init.zeros_(self.combined_layer2.bias)
                    nn.init.xavier_uniform_(self.output_layer.weight)
                    nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """Apply parametric transformation based on complexity level"""
        batch_size, height, width, channels = x.shape

        # Flatten spatial dimensions for matrix multiplication
        # Use contiguous().view() to preserve gradients
        x_flat = x.contiguous().view(batch_size * height * width, channels)

        if self.complexity == 'simple':
            # Simple matrix transformation
            transformed = torch.matmul(x_flat, self.transformation_matrix.t())

        elif self.complexity == 'medium':
            # Medium: Two-layer with activation
            transformed = self.activation(self.layer1(x_flat))
            transformed = self.layer2(transformed)

        elif self.complexity == 'complex':
            # Complex: Three-layer with residual connection
            transformed = self.activation(self.layer1(x_flat))
            transformed = self.activation(self.layer2(transformed))
            transformed = self.layer3(transformed)

            # Add residual connection
            residual = self.residual(x_flat)
            transformed = transformed + residual

        elif self.complexity == 'complexer':
            # Complexer: MLP branch + CNN branch + fusion
            
            # MLP branch (same as complex)
            mlp_out = self.activation(self.layer1(x_flat))
            mlp_out = self.activation(self.layer2(mlp_out))
            mlp_out = self.layer3(mlp_out)
            
            # CNN branch - process spatial features
            # Reshape to [B, C, H, W] for CNN processing
            x_spatial = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            cnn_out = self.conv_activation(self.conv1(x_spatial))
            cnn_out = self.conv_activation(self.conv2(cnn_out))
            cnn_out = self.conv3(cnn_out)
            
            # Flatten CNN output back to [B*H*W, output_dim]
            cnn_out = cnn_out.permute(0, 2, 3, 1).contiguous().view(batch_size * height * width, self.output_dim)
            
            # Concatenate MLP and CNN outputs
            combined = torch.cat([mlp_out, cnn_out], dim=-1)
            
            # Fuse the outputs
            transformed = self.fusion(combined)
            
            # Add residual connection
            residual = self.residual(x_flat)
            transformed = transformed + residual
            
        elif self.complexity == 'complex_with_hsl':
            # Complex with HSL color space processing
            # Input: [B*H*W, 5] with format [x, y, R, G, B]
            # Output: [B*H*W, 3] with format [R, G, B]
            
            # Reshape for processing
            x_reshaped = x_flat.contiguous().view(batch_size, height, width, channels)
            
            # Split coordinates and RGB
            coords = x_reshaped[..., :2]  # [B, H, W, 2]
            rgb = x_reshaped[..., 2:]     # [B, H, W, 3]
            
            # Convert RGB to HSL
            hsl = rgb_to_hsl(rgb)  # [B, H, W, 3]
            
            # Flatten for network processing
            coords_flat = coords.contiguous().view(batch_size * height * width, 2)
            hsl_flat = hsl.contiguous().view(batch_size * height * width, 3)
            
            # Process coordinates
            coord_features = self.activation(self.coord_layer(coords_flat))
            
            # Process HSL
            hsl_features = self.activation(self.hsl_layer1(hsl_flat))
            
            # Combine features
            combined = torch.cat([coord_features, hsl_features], dim=-1)
            
            # Process combined features
            combined = self.activation(self.combined_layer1(combined))
            combined = self.activation(self.combined_layer2(combined))
            
            # Output HSL adjustments
            hsl_output = self.output_layer(combined)
            
            # Reshape back
            hsl_output = hsl_output.contiguous().view(batch_size, height, width, 3)
            
            # Convert back to RGB
            rgb_output = hsl_to_rgb(hsl_output)
            
            # Flatten for return
            transformed = rgb_output.contiguous().view(batch_size * height * width, self.output_dim)

        elif self.complexity == 'fisheye':
            # Traditional fisheye distortion correction
            # Reshape to coordinate format for fisheye transformation
            coords = x_flat.contiguous().view(batch_size, height, width, channels)
            norm_coords = 2.0 * coords - 1.0
            x_centered = norm_coords[..., 0] - (2.0 * self.cx - 1.0)
            y_centered = norm_coords[..., 1] - (2.0 * self.cy - 1.0)
            r = torch.sqrt(x_centered**2 + y_centered**2)
            r_distorted = r * (1.0 + self.k1 * r + self.k2 * r**2 + self.k3 * r**3)

            x_distorted = torch.where(
                r > 0,
                x_centered * (r_distorted / r) * self.scale,
                x_centered * self.scale
            )
            y_distorted = torch.where(
                r > 0,
                y_centered * (r_distorted / r) * self.scale,
                y_centered * self.scale
            )

            x_out = x_distorted + (2.0 * self.cx - 1.0)
            y_out = y_distorted + (2.0 * self.cy - 1.0)
            x_out = (x_out + 1.0) / 2.0
            y_out = (y_out + 1.0) / 2.0

            transformed_coords = torch.stack([x_out, y_out], dim=-1)
            transformed = transformed_coords.contiguous().view(batch_size * height * width, self.output_dim)

        # Reshape back to spatial dimensions
        transformed = transformed.contiguous().view(batch_size, height, width, self.output_dim)

        return transformed


# =============================================================================
# APPROACH IMPLEMENTATIONS
# =============================================================================

class ApproachA1(BaseApproach):
    """A1: x,y → model → x',y' → bilinear interpolation"""
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None,
                 grid_sample_mode='bilinear'):
        if model_type == 'neural':
            model = NeuralModel(2, 2, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(2, 2, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(2, 2, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA1, self).__init__(model, zero_mask, grid_sample_mode)
    
    def forward(self, x):
        """Apply A1 approach: coordinate transformation only"""
        batch_size, channels, height, width = x.shape
        device = x.device

        # Create coordinate grid
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)

        # Apply model transformation to coordinates
        transformed_coords = self.model(coord_grid)
        # Keep coordinates in [0,1] to preserve grid gradients
        transformed_coords = torch.clamp(transformed_coords, 0.0, 1.0)

        # Convert to grid_sample format [-1, 1]
        sample_coords = 2.0 * transformed_coords - 1.0

        # Apply bilinear sampling to image using configured grid_sample
        transformed_img = grid_sample(
            torch.clamp(x, 0.0, 1.0),
            sample_coords,
            mode=self.grid_sample_mode,
            padding_mode='border',
            align_corners=True
        )

        # Apply zero mask to preserve zero areas
        transformed_img = self.apply_zero_mask(transformed_img)

        return transformed_img


class ApproachA2(BaseApproach):
    """A2: x,y,R,G,B → model → x',y',R,G,B → bilinear first, then delta RGB"""
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None,
                 grid_sample_mode='bilinear'):
        if model_type == 'neural':
            model = NeuralModel(5, 5, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(5, 5, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(5, 5, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA2, self).__init__(model, zero_mask, grid_sample_mode)
    
    def forward(self, x):
        """Apply A2 approach: hybrid transformation (bilinear → delta RGB)"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Create coordinate grid
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Get pixel colors and normalize to [0, 1]
        pixel_colors = x.permute(0, 2, 3, 1)  # [batch, height, width, 3]
        
        # Combine coordinates and colors: [batch, height, width, 5]
        combined_input = torch.cat([coord_grid, pixel_colors], dim=-1)
        
        # Apply model transformation: [batch, height, width, 5] → [batch, height, width, 5]
        transformation_output = self.model(combined_input)

        # Extract coordinate and color transformations
        coord_transform = transformation_output[..., :2]  # [batch, height, width, 2]
        color_transform = transformation_output[..., 2:]  # [batch, height, width, 3]

        # Step 1: Apply bilinear interpolation with coordinate transformation
        coord_transform = torch.clamp(coord_transform, 0.0, 1.0)
        sample_coords = 2.0 * coord_transform - 1.0
        warped_img = grid_sample(
            torch.clamp(x, 0.0, 1.0),
            sample_coords,
            mode=self.grid_sample_mode,
            padding_mode='zeros',
            align_corners=True
        )

        # Step 2: Apply color transformation (delta RGB)
        color_transform = 2.0 * color_transform - 1.0  # Convert from [0,1] to [-1,1]
        final_img = warped_img + color_transform.permute(0, 3, 1, 2)  # Add color deltas

        # Apply zero mask to preserve zero areas
        final_img = self.apply_zero_mask(final_img)

        return final_img


class ApproachA3(BaseApproach):
    """A3: x,y,R,G,B → model → x',y',R,G,B → delta RGB first, then bilinear"""
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None,
                 grid_sample_mode='bilinear'):
        if model_type == 'neural':
            model = NeuralModel(5, 5, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(5, 5, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(5, 5, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA3, self).__init__(model, zero_mask, grid_sample_mode)

    def forward(self, x):
        """Apply A3 approach: hybrid transformation (delta RGB → bilinear)"""
        batch_size, channels, height, width = x.shape
        device = x.device

        # Create coordinate grid
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)

        # Get pixel colors and normalize to [0, 1]
        pixel_colors = x.permute(0, 2, 3, 1)  # [batch, height, width, 3]

        # Combine coordinates and colors: [batch, height, width, 5]
        combined_input = torch.cat([coord_grid, pixel_colors], dim=-1)

        # Apply model transformation: [batch, height, width, 5] → [batch, height, width, 5]
        transformation_output = self.model(combined_input)

        # Extract coordinate and color transformations
        coord_transform = transformation_output[..., :2]  # [batch, height, width, 2]
        color_transform = transformation_output[..., 2:]  # [batch, height, width, 3]

        # Step 1: Apply color transformation (delta RGB) to original image
        color_transform = 2.0 * color_transform - 1.0  # Convert from [0,1] to [-1,1]
        color_modified_img = x + color_transform.permute(0, 3, 1, 2)  # Add color deltas

        # Step 2: Apply bilinear interpolation with coordinate transformation
        coord_transform = torch.clamp(coord_transform, 0.0, 1.0)
        sample_coords = 2.0 * coord_transform - 1.0
        final_img = grid_sample(
            color_modified_img,
            sample_coords,
            mode=self.grid_sample_mode,
            padding_mode='zeros',
            align_corners=True
        )
        
        # Apply zero mask to preserve zero areas
        final_img = self.apply_zero_mask(final_img)

        return final_img

class KANLayer(nn.Module):
    """
    Simple Kolmogorov–Arnold Layer: applies trainable spline-like univariate nonlinearities
    to each input dimension, then mixes them linearly.
    """
    def __init__(self, in_dim, out_dim, num_basis=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis

        # Linear transform of input basis coefficients
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim, num_basis) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Centers and widths of spline bases
        self.centers = nn.Parameter(torch.linspace(0, 1, num_basis).repeat(in_dim, 1))
        self.widths = nn.Parameter(torch.ones(in_dim, num_basis) * 0.25)

    def forward(self, x):
        # x: [B, ..., in_dim]
        # Compute spline basis activations per input dim
        # Gaussian RBF basis functions
        bases = []
        for d in range(self.in_dim):
            xb = x[..., d].unsqueeze(-1)  # [B, ..., 1]
            phi = torch.exp(-((xb - self.centers[d])**2) / (2 * self.widths[d]**2))  # [B, ..., num_basis]
            bases.append(phi)
        bases = torch.stack(bases, dim=-2)  # [B, ..., in_dim, num_basis]

        # Weighted sum over bases → output dims
        out = torch.einsum("...ib,oib->...o", bases, self.weight) + self.bias
        return out

class SimpleKAN(nn.Module):
    def __init__(self, in_dim=5, out_dim=3, hidden_dim=64, num_layers=3, num_basis=4):
        super().__init__()
        print(f"hidden_dim: {hidden_dim}")
        print(f"num_layers: {num_layers}")
        print(f"num_basis: {num_basis}")
        print(f"in_dim: {in_dim}")
        print(f"out_dim: {out_dim}")
        # print("!!!!!!!!!!!!!!!")
        # sys.exit()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i + 1], num_basis))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)  # smooth nonlinearity
        return x


class ApproachA4(BaseApproach):
    """A4: x,y,R,G,B → model → R,G,B → predicts absolute RGB (like A5 net2)
    This variant outputs actual RGB values in [0,1] instead of delta RGB.
    """
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None):
        if model_type == 'neural':
            model = NeuralModel(5, 3, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(5, 3, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(5, 3, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA4, self).__init__(model, zero_mask)

    def forward(self, x):
        """Apply A4 approach: color transformation only (absolute RGB prediction)"""
        batch_size, channels, height, width = x.shape
        device = x.device

        # Create coordinate grid
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)

        # Get pixel colors and normalize to [0, 1]
        pixel_colors = x.permute(0, 2, 3, 1)  # [batch, height, width, 3]

        # Combine coordinates and colors: [batch, height, width, 5]
        combined_input = torch.cat([coord_grid, pixel_colors], dim=-1)

        # Apply model transformation: [batch, height, width, 5] → [batch, height, width, 3]
        predicted_rgb = self.model(combined_input)
        # Constrain to [0,1] like A5 net2
        predicted_rgb = torch.sigmoid(predicted_rgb)
        final_img = predicted_rgb.permute(0, 3, 1, 2)

        # Apply zero mask to preserve zero areas
        final_img = self.apply_zero_mask(final_img)

        return final_img


class ApproachA9(BaseApproach):
    """A9: Multiple pixels (n×n patch) with xyRGB → model → R,G,B → predicts absolute RGB
    Similar to A4 but takes a patch of n×n pixels (where n² = 9, 25, 36, etc.) around each pixel.
    Each pixel in the patch has xyRGB (5 values), so input is n² × 5 values.
    """
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None, patch_size=3):
        """
        Args:
            patch_size: Size of the patch (n). Must be an odd number. patch_size² pixels will be used.
                       Common values: 3 (9 pixels), 5 (25 pixels), 6 (36 pixels)
        """
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
        
        self.patch_size = patch_size
        input_dim = patch_size * patch_size * 5  # n² pixels × 5 values (xyRGB) per pixel
        
        if model_type == 'neural':
            model = NeuralModel(input_dim, 3, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(input_dim, 3, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(input_dim, 3, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA9, self).__init__(model, zero_mask)

    def forward(self, x):
        """Apply A9 approach: patch-based color transformation (absolute RGB prediction)"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Calculate padding needed for patch extraction
        pad_size = self.patch_size // 2
        
        # Pad the image with reflection padding to handle edge cases
        # F.pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        # For 4D tensor [B, C, H, W]: (pad_left, pad_right, pad_top, pad_bottom)
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Create coordinate grid for the original (unpadded) image
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Pad coordinate grid similarly
        # coord_grid is [batch, height, width, 2], need to permute to [batch, 2, height, width] for padding
        coord_grid_padded = F.pad(coord_grid.permute(0, 3, 1, 2), 
                                  (pad_size, pad_size, pad_size, pad_size), 
                                  mode='reflect').permute(0, 2, 3, 1)
        
        # Use unfold to extract patches efficiently (vectorized)
        # x_padded: [batch, channels, height+2*pad, width+2*pad]
        # We'll unfold to get patches: [batch, channels * patch_size * patch_size, height * width]
        x_unfolded = F.unfold(x_padded, kernel_size=self.patch_size, stride=1)
        # x_unfolded: [batch, channels * patch_size * patch_size, height * width]
        
        # Reshape to [batch, height * width, channels, patch_size, patch_size]
        x_unfolded = x_unfolded.view(batch_size, channels, self.patch_size * self.patch_size, height, width)
        x_unfolded = x_unfolded.permute(0, 3, 4, 1, 2)  # [batch, height, width, channels, patch_size²]
        x_unfolded = x_unfolded.contiguous().view(batch_size, height, width, channels, self.patch_size, self.patch_size)
        x_unfolded = x_unfolded.permute(0, 1, 2, 4, 5, 3)  # [batch, height, width, patch_size, patch_size, channels]
        
        # Do the same for coordinates
        coord_unfolded = F.unfold(coord_grid_padded.permute(0, 3, 1, 2), 
                                  kernel_size=self.patch_size, stride=1)
        # coord_unfolded: [batch, 2 * patch_size * patch_size, height * width]
        coord_unfolded = coord_unfolded.view(batch_size, 2, self.patch_size * self.patch_size, height, width)
        coord_unfolded = coord_unfolded.permute(0, 3, 4, 1, 2)  # [batch, height, width, 2, patch_size²]
        coord_unfolded = coord_unfolded.contiguous().view(batch_size, height, width, 2, self.patch_size, self.patch_size)
        coord_unfolded = coord_unfolded.permute(0, 1, 2, 4, 5, 3)  # [batch, height, width, patch_size, patch_size, 2]
        
        # Combine coordinates and colors: [batch, height, width, patch_size, patch_size, 5]
        patch_combined = torch.cat([coord_unfolded, x_unfolded], dim=-1)
        
        # Flatten the patch: [batch, height, width, patch_size * patch_size * 5]
        patch_flat = patch_combined.contiguous().view(batch_size, height, width, -1)
        
        # Apply model transformation: [batch, height, width, patch_size² * 5] → [batch, height, width, 3]
        predicted_rgb = self.model(patch_flat)
        # Constrain to [0,1]
        predicted_rgb = torch.sigmoid(predicted_rgb)
        final_img = predicted_rgb.permute(0, 3, 1, 2)
        
        # Apply zero mask to preserve zero areas
        final_img = self.apply_zero_mask(final_img)
        
        return final_img


class ApproachA10(BaseApproach):
    """A10: x,y,R,G,B → model → delta(R,G,B) → predicts delta RGB (similar to A4 but with delta)
    Similar to A4 but instead of predicting absolute RGB values, predicts delta RGB.
    Delta RGB is in range [-2, 2], then added to input RGB, and final RGB is clamped to [0, 1].
    """
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None):
        if model_type == 'neural':
            model = NeuralModel(5, 3, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(5, 3, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(5, 3, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA10, self).__init__(model, zero_mask)

    def forward(self, x):
        """Apply A10 approach: color transformation with delta RGB prediction"""
        batch_size, channels, height, width = x.shape
        device = x.device

        # Create coordinate grid
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)

        # Get pixel colors (already in [0, 1] range)
        pixel_colors = x.permute(0, 2, 3, 1)  # [batch, height, width, 3]

        # Combine coordinates and colors: [batch, height, width, 5]
        combined_input = torch.cat([coord_grid, pixel_colors], dim=-1)

        # Apply model transformation: [batch, height, width, 5] → [batch, height, width, 3]
        delta_rgb = self.model(combined_input)
        
        # Constrain delta to [0, 1] via sigmoid, then convert to [-2, 2]
        delta_rgb = torch.sigmoid(delta_rgb)  # [0, 1]
        delta_rgb = 4.0 * delta_rgb - 2.0  # Convert from [0, 1] to [-2, 2]
        
        # Add delta RGB to input RGB
        final_rgb = pixel_colors + delta_rgb  # Can be in range [-2, 2]
        
        # Clamp final RGB to [0, 1] to match input range
        final_rgb = torch.clamp(final_rgb, 0.0, 1.0)
        
        # Convert back to [B, C, H, W] format
        final_img = final_rgb.permute(0, 3, 1, 2)

        # Apply zero mask to preserve zero areas
        final_img = self.apply_zero_mask(final_img)

        return final_img


class ApproachA11(BaseApproach):
    """A11: Multiple pixels (n×n patch) with xyRGB → model → delta(R,G,B) → predicts delta RGB
    Similar to A9 but instead of predicting absolute RGB values, predicts delta RGB.
    Delta RGB is in range [-2, 2], then added to input RGB, and final RGB is clamped to [0, 1].
    """
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None, patch_size=3):
        """
        Args:
            patch_size: Size of the patch (n). Must be an odd number. patch_size² pixels will be used.
                       Common values: 3 (9 pixels), 5 (25 pixels), 6 (36 pixels)
        """
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
        
        self.patch_size = patch_size
        input_dim = patch_size * patch_size * 5  # n² pixels × 5 values (xyRGB) per pixel
        
        if model_type == 'neural':
            model = NeuralModel(input_dim, 3, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(input_dim, 3, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(input_dim, 3, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA11, self).__init__(model, zero_mask)

    def forward(self, x):
        """Apply A11 approach: patch-based color transformation with delta RGB prediction"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Calculate padding needed for patch extraction
        pad_size = self.patch_size // 2
        
        # Pad the image with reflection padding to handle edge cases
        # F.pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        # For 4D tensor [B, C, H, W]: (pad_left, pad_right, pad_top, pad_bottom)
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Create coordinate grid for the original (unpadded) image
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Pad coordinate grid similarly
        # coord_grid is [batch, height, width, 2], need to permute to [batch, 2, height, width] for padding
        coord_grid_padded = F.pad(coord_grid.permute(0, 3, 1, 2), 
                                  (pad_size, pad_size, pad_size, pad_size), 
                                  mode='reflect').permute(0, 2, 3, 1)
        
        # Use unfold to extract patches efficiently (vectorized)
        # x_padded: [batch, channels, height+2*pad, width+2*pad]
        # We'll unfold to get patches: [batch, channels * patch_size * patch_size, height * width]
        x_unfolded = F.unfold(x_padded, kernel_size=self.patch_size, stride=1)
        # x_unfolded: [batch, channels * patch_size * patch_size, height * width]
        
        # Reshape to [batch, height * width, channels, patch_size, patch_size]
        x_unfolded = x_unfolded.view(batch_size, channels, self.patch_size * self.patch_size, height, width)
        x_unfolded = x_unfolded.permute(0, 3, 4, 1, 2)  # [batch, height, width, channels, patch_size²]
        x_unfolded = x_unfolded.contiguous().view(batch_size, height, width, channels, self.patch_size, self.patch_size)
        x_unfolded = x_unfolded.permute(0, 1, 2, 4, 5, 3)  # [batch, height, width, patch_size, patch_size, channels]
        
        # Do the same for coordinates
        coord_unfolded = F.unfold(coord_grid_padded.permute(0, 3, 1, 2), 
                                  kernel_size=self.patch_size, stride=1)
        # coord_unfolded: [batch, 2 * patch_size * patch_size, height * width]
        coord_unfolded = coord_unfolded.view(batch_size, 2, self.patch_size * self.patch_size, height, width)
        coord_unfolded = coord_unfolded.permute(0, 3, 4, 1, 2)  # [batch, height, width, 2, patch_size²]
        coord_unfolded = coord_unfolded.contiguous().view(batch_size, height, width, 2, self.patch_size, self.patch_size)
        coord_unfolded = coord_unfolded.permute(0, 1, 2, 4, 5, 3)  # [batch, height, width, patch_size, patch_size, 2]
        
        # Combine coordinates and colors: [batch, height, width, patch_size, patch_size, 5]
        patch_combined = torch.cat([coord_unfolded, x_unfolded], dim=-1)
        
        # Flatten the patch: [batch, height, width, patch_size * patch_size * 5]
        patch_flat = patch_combined.contiguous().view(batch_size, height, width, -1)
        
        # Apply model transformation: [batch, height, width, patch_size² * 5] → [batch, height, width, 3]
        delta_rgb = self.model(patch_flat)
        
        # Constrain delta to [0, 1] via sigmoid, then convert to [-2, 2]
        delta_rgb = torch.sigmoid(delta_rgb)  # [0, 1]
        delta_rgb = 4.0 * delta_rgb - 2.0  # Convert from [0, 1] to [-2, 2]
        
        # Get pixel colors for the center pixel of each patch (original input)
        pixel_colors = x.permute(0, 2, 3, 1)  # [batch, height, width, 3]
        
        # Add delta RGB to input RGB
        final_rgb = pixel_colors + delta_rgb  # Can be in range [-2, 2]
        
        # Clamp final RGB to [0, 1] to match input range
        final_rgb = torch.clamp(final_rgb, 0.0, 1.0)
        
        # Convert back to [B, C, H, W] format
        final_img = final_rgb.permute(0, 3, 1, 2)
        
        # Apply zero mask to preserve zero areas
        final_img = self.apply_zero_mask(final_img)
        
        return final_img


class ApproachA12(BaseApproach):
    """A12: Center pixel (x,y,RGB) + 8 adjacent pixels RGB → model → delta(R,G,B) → predicts delta RGB
    Similar to A11 but uses only center pixel coordinates+RGB and neighbor RGB values.
    Input: 2 (center x,y) + 3 (center RGB) + 24 (8 neighbors × 3 RGB) = 29 values for 3×3 patch
    """
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None, patch_size=3):
        """
        Args:
            patch_size: Size of the patch (n). Must be an odd number (typically 3 for 8 neighbors).
                       For patch_size=3: center (x,y,RGB) + 8 neighbors RGB = 2 + 3 + 24 = 29 input dim
        """
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
        if patch_size < 3:
            raise ValueError(f"patch_size must be at least 3, got {patch_size}")
        
        self.patch_size = patch_size
        pad_size = patch_size // 2
        num_neighbors = patch_size * patch_size - 1  # All pixels except center
        input_dim = 2 + 3 + num_neighbors * 3  # center (x,y,RGB) + neighbors RGB
        
        if model_type == 'neural':
            model = NeuralModel(input_dim, 3, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(input_dim, 3, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(input_dim, 3, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA12, self).__init__(model, zero_mask)

    def forward(self, x):
        """Apply A12 approach: center pixel coordinates + neighbor RGB → delta RGB prediction"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Calculate padding needed for patch extraction
        pad_size = self.patch_size // 2
        
        # Pad the image with reflection padding to handle edge cases
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Create coordinate grid for the original (unpadded) image
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Pad coordinate grid similarly
        coord_grid_padded = F.pad(coord_grid.permute(0, 3, 1, 2), 
                                  (pad_size, pad_size, pad_size, pad_size), 
                                  mode='reflect').permute(0, 2, 3, 1)
        
        # Use unfold to extract patches efficiently
        x_unfolded = F.unfold(x_padded, kernel_size=self.patch_size, stride=1)
        # x_unfolded: [batch, channels * patch_size * patch_size, height * width]
        
        # Reshape to [batch, height, width, patch_size, patch_size, channels]
        x_unfolded = x_unfolded.view(batch_size, channels, self.patch_size * self.patch_size, height, width)
        x_unfolded = x_unfolded.permute(0, 3, 4, 1, 2)  # [batch, height, width, channels, patch_size²]
        x_unfolded = x_unfolded.contiguous().view(batch_size, height, width, channels, self.patch_size, self.patch_size)
        x_unfolded = x_unfolded.permute(0, 1, 2, 4, 5, 3)  # [batch, height, width, patch_size, patch_size, channels]
        
        # Extract coordinates for patches
        coord_unfolded = F.unfold(coord_grid_padded.permute(0, 3, 1, 2), 
                                  kernel_size=self.patch_size, stride=1)
        coord_unfolded = coord_unfolded.view(batch_size, 2, self.patch_size * self.patch_size, height, width)
        coord_unfolded = coord_unfolded.permute(0, 3, 4, 1, 2)  # [batch, height, width, 2, patch_size²]
        coord_unfolded = coord_unfolded.contiguous().view(batch_size, height, width, 2, self.patch_size, self.patch_size)
        coord_unfolded = coord_unfolded.permute(0, 1, 2, 4, 5, 3)  # [batch, height, width, patch_size, patch_size, 2]
        
        # Extract center pixel coordinates (at position [pad_size, pad_size] in the patch)
        center_coords = coord_unfolded[:, :, :, pad_size, pad_size, :]  # [batch, height, width, 2]
        
        # Extract center pixel RGB
        center_rgb = x_unfolded[:, :, :, pad_size, pad_size, :]  # [batch, height, width, 3]
        
        # Extract neighbor RGB values (all pixels except center)
        # Create indices for all positions in the patch
        neighbor_rgb_list = []
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                if i != pad_size or j != pad_size:  # Skip center pixel
                    neighbor_rgb_list.append(x_unfolded[:, :, :, i, j, :])  # [batch, height, width, 3]
        
        # Concatenate all neighbors: [batch, height, width, num_neighbors * 3]
        neighbor_rgb = torch.cat(neighbor_rgb_list, dim=-1)  # [batch, height, width, num_neighbors * 3]
        
        # Combine center coordinates, center RGB, and neighbor RGB: [batch, height, width, 2 + 3 + num_neighbors * 3]
        patch_features = torch.cat([center_coords, center_rgb, neighbor_rgb], dim=-1)
        
        # Apply model transformation: [batch, height, width, input_dim] → [batch, height, width, 3]
        delta_rgb = self.model(patch_features)
        
        # Constrain delta to [0, 1] via sigmoid, then convert to [-2, 2]
        delta_rgb = torch.sigmoid(delta_rgb)  # [0, 1]
        delta_rgb = 4.0 * delta_rgb - 2.0  # Convert from [0, 1] to [-2, 2]
        
        # Get pixel colors for the center pixel of each patch (original input)
        pixel_colors = x.permute(0, 2, 3, 1)  # [batch, height, width, 3]
        
        # Add delta RGB to input RGB
        final_rgb = pixel_colors + delta_rgb  # Can be in range [-2, 2]
        
        # Clamp final RGB to [0, 1] to match input range
        final_rgb = torch.clamp(final_rgb, 0.0, 1.0)
        
        # Convert back to [B, C, H, W] format
        final_img = final_rgb.permute(0, 3, 1, 2)
        
        # Apply zero mask to preserve zero areas
        final_img = self.apply_zero_mask(final_img)
        
        return final_img


class ApproachA13(BaseApproach):
    """A13: Large patch (n×n pixels) with center (x,y,RGB) + neighbors RGB → model → delta(RGB) for m×m center pixels
    Similar to A12 but processes larger patches and outputs multiple pixels at once.
    Takes input_patch_size×input_patch_size pixels (e.g., 5×5=25) and outputs output_patch_size×output_patch_size pixels (e.g., 3×3=9).
    Only needs to be called (H×W)/(output_patch_size²) times instead of H×W times.
    """
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None, 
                 input_patch_size=5, output_patch_size=3):
        """
        Args:
            input_patch_size: Size of input patch (n). Must be odd. input_patch_size² pixels will be used.
                             Default: 5 (25 pixels)
            output_patch_size: Size of output patch (m). Must be odd and <= input_patch_size.
                              Default: 3 (9 pixels)
        """
        if input_patch_size % 2 == 0:
            raise ValueError(f"input_patch_size must be odd, got {input_patch_size}")
        if output_patch_size % 2 == 0:
            raise ValueError(f"output_patch_size must be odd, got {output_patch_size}")
        if output_patch_size > input_patch_size:
            raise ValueError(f"output_patch_size ({output_patch_size}) must be <= input_patch_size ({input_patch_size})")
        if input_patch_size < 3:
            raise ValueError(f"input_patch_size must be at least 3, got {input_patch_size}")
        if output_patch_size < 1:
            raise ValueError(f"output_patch_size must be at least 1, got {output_patch_size}")
        
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        input_pad_size = input_patch_size // 2
        output_pad_size = output_patch_size // 2
        
        # Input: center (x,y,RGB) + neighbors RGB
        num_neighbors = input_patch_size * input_patch_size - 1  # All pixels except center
        input_dim = 2 + 3 + num_neighbors * 3  # center (x,y,RGB) + neighbors RGB
        
        # Output: output_patch_size² pixels × 3 RGB channels
        output_dim = output_patch_size * output_patch_size * 3
        
        if model_type == 'neural':
            model = NeuralModel(input_dim, output_dim, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(input_dim, output_dim, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(input_dim, output_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA13, self).__init__(model, zero_mask)

    def forward(self, x):
        """Apply A13 approach: large patch → delta RGB for multiple center pixels (VECTORIZED)
        
        This vectorized version processes all patches in a SINGLE model forward pass,
        similar to A12. This eliminates the bottleneck of sequential processing.
        
        Key optimization:
        - OLD: 7,396 separate model forward passes (one per patch) - SLOW
        - NEW: 1 model forward pass for all patches - FAST
        
        The patch extraction still uses a loop, but this is much cheaper than
        the model forward passes we eliminated.
        """
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Calculate padding needed for input patch extraction
        input_pad_size = self.input_patch_size // 2
        output_pad_size = self.output_patch_size // 2
        
        # Pad the image with reflection padding to handle edge cases
        x_padded = F.pad(x, (input_pad_size, input_pad_size, input_pad_size, input_pad_size), mode='reflect')
        
        # Create coordinate grid for the original (unpadded) image
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Pad coordinate grid similarly
        coord_grid_padded = F.pad(coord_grid.permute(0, 3, 1, 2), 
                                  (input_pad_size, input_pad_size, input_pad_size, input_pad_size), 
                                  mode='reflect').permute(0, 2, 3, 1)
        
        # Calculate output dimensions (we process with stride = output_patch_size)
        output_height = (height + self.output_patch_size - 1) // self.output_patch_size
        output_width = (width + self.output_patch_size - 1) // self.output_patch_size
        
        # Use unfold to extract all input patches at once (like A12, but with stride=output_patch_size)
        # We need patches centered at: (output_pad_size, output_pad_size), (output_pad_size, output_pad_size + output_patch_size), etc.
        # The top-left of each input patch is: center - input_pad_size
        # So top-left positions are: (output_pad_size - input_pad_size, output_pad_size - input_pad_size), etc.
        # 
        # unfold with stride=output_patch_size extracts patches starting at: 0, output_patch_size, 2*output_patch_size, etc.
        # To get patches starting at (output_pad_size - input_pad_size), we need to add padding
        # 
        # Additional padding needed: (output_pad_size - input_pad_size) on left/top
        # But we already have input_pad_size padding, so we need: output_pad_size - input_pad_size more
        # However, if output_pad_size < input_pad_size, we need to crop instead
        
        # Calculate offset: where should the first patch start?
        # First patch center: (output_pad_size, output_pad_size)
        # First patch top-left: (output_pad_size - input_pad_size, output_pad_size - input_pad_size)
        # In padded coordinates: (output_pad_size - input_pad_size + input_pad_size, ...) = (output_pad_size, output_pad_size)
        
        # Actually, in padded coordinates, the first patch should start at:
        # patch_start = output_pad_size (in original image) = output_pad_size + input_pad_size (in padded image)
        # But unfold with stride=output_patch_size starts at: input_pad_size (in padded image)
        # So we need to shift by: output_pad_size
        
        # Add extra padding to align patch positions with unfold
        # In padded image (with input_pad_size padding), we want patches starting at:
        #   output_pad_size (in original) = output_pad_size + input_pad_size (in padded)
        # unfold with stride=output_patch_size starts at 0 in padded image
        # So we need to add (output_pad_size + input_pad_size) padding on left/top
        extra_pad_h = output_pad_size + input_pad_size
        extra_pad_w = output_pad_size + input_pad_size
        
        # Re-pad with extra padding for alignment
        x_padded_unfold = F.pad(x_padded, (extra_pad_w, 0, extra_pad_h, 0), mode='reflect')
        coord_grid_padded_unfold = F.pad(coord_grid_padded.permute(0, 3, 1, 2), 
                                        (extra_pad_w, 0, extra_pad_h, 0), mode='reflect').permute(0, 2, 3, 1)
        
        # Use unfold to extract all patches at once with stride = output_patch_size
        # This extracts input_patch_size × input_patch_size patches starting at:
        #   Position 0 in the re-padded image = (output_pad_size + input_pad_size) in original padded image
        #   = output_pad_size in original image = center of first output patch - input_pad_size ✓
        x_unfolded = F.unfold(x_padded_unfold, kernel_size=self.input_patch_size, stride=self.output_patch_size)
        # x_unfolded: [batch, channels * input_patch_size * input_patch_size, num_patches_unfold]
        
        # Calculate actual number of patches extracted by unfold
        h_padded_unfold, w_padded_unfold = x_padded_unfold.shape[2], x_padded_unfold.shape[3]
        num_patches_h = (h_padded_unfold - self.input_patch_size) // self.output_patch_size + 1
        num_patches_w = (w_padded_unfold - self.input_patch_size) // self.output_patch_size + 1
        num_patches_unfold = num_patches_h * num_patches_w
        
        # Reshape to [batch, num_patches_unfold, input_patch_size, input_patch_size, channels]
        x_unfolded = x_unfolded.view(batch_size, channels, self.input_patch_size * self.input_patch_size, num_patches_unfold)
        x_unfolded = x_unfolded.permute(0, 3, 1, 2)  # [batch, num_patches_unfold, channels, input_patch_size²]
        x_unfolded = x_unfolded.contiguous().view(batch_size, num_patches_unfold, channels, 
                                                 self.input_patch_size, self.input_patch_size)
        x_unfolded = x_unfolded.permute(0, 1, 3, 4, 2)  # [batch, num_patches_unfold, input_patch_size, input_patch_size, channels]
        
        # Extract coordinates for patches using unfold
        coord_unfolded = F.unfold(coord_grid_padded_unfold.permute(0, 3, 1, 2), 
                                  kernel_size=self.input_patch_size, stride=self.output_patch_size)
        coord_unfolded = coord_unfolded.view(batch_size, 2, self.input_patch_size * self.input_patch_size, num_patches_unfold)
        coord_unfolded = coord_unfolded.permute(0, 3, 1, 2)  # [batch, num_patches_unfold, 2, input_patch_size²]
        coord_unfolded = coord_unfolded.contiguous().view(batch_size, num_patches_unfold, 2, 
                                                          self.input_patch_size, self.input_patch_size)
        coord_unfolded = coord_unfolded.permute(0, 1, 3, 4, 2)  # [batch, num_patches_unfold, input_patch_size, input_patch_size, 2]
        
        # Reshape to [batch, num_patches_h, num_patches_w, input_patch_size, input_patch_size, channels]
        # Use the actual dimensions from unfold
        all_patches = x_unfolded.view(batch_size, num_patches_h, num_patches_w,
                                      self.input_patch_size, self.input_patch_size, channels)
        all_coord_patches = coord_unfolded.view(batch_size, num_patches_h, num_patches_w,
                                                 self.input_patch_size, self.input_patch_size, 2)
        
        # Select only the patches we need (crop to output_height × output_width)
        # unfold might extract more patches than we need due to padding
        if num_patches_h >= output_height and num_patches_w >= output_width:
            all_patches = all_patches[:, :output_height, :output_width, :, :, :]
            all_coord_patches = all_coord_patches[:, :output_height, :output_width, :, :, :]
        else:
            # If we don't have enough patches, pad with zeros (shouldn't happen with proper padding)
            pad_h = max(0, output_height - num_patches_h)
            pad_w = max(0, output_width - num_patches_w)
            if pad_h > 0 or pad_w > 0:
                all_patches = F.pad(all_patches, (0, 0, 0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
                all_coord_patches = F.pad(all_coord_patches, (0, 0, 0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # OPTIMIZED: Vectorize neighbor extraction instead of using loops
        # all_patches: [batch, output_height, output_width, input_patch_size, input_patch_size, channels]
        # all_coord_patches: [batch, output_height, output_width, input_patch_size, input_patch_size, 2]
        
        # Extract center pixel coordinates and RGB
        center_coords = all_coord_patches[:, :, :, input_pad_size, input_pad_size, :]  # [batch, output_height, output_width, 2]
        center_rgb = all_patches[:, :, :, input_pad_size, input_pad_size, :]  # [batch, output_height, output_width, 3]
        
        # Vectorized neighbor extraction: reshape and mask out center
        # Reshape to [batch, output_height, output_width, input_patch_size * input_patch_size, channels]
        patches_flat = all_patches.view(batch_size, output_height, output_width, 
                                       self.input_patch_size * self.input_patch_size, channels)
        
        # Create mask to exclude center pixel (at position input_pad_size * input_patch_size + input_pad_size)
        center_idx = input_pad_size * self.input_patch_size + input_pad_size
        mask = torch.ones(self.input_patch_size * self.input_patch_size, dtype=torch.bool, device=device)
        mask[center_idx] = False
        
        # Extract neighbors using mask: [batch, output_height, output_width, num_neighbors, channels]
        neighbor_rgb_flat = patches_flat[:, :, :, mask, :]  # [batch, output_height, output_width, num_neighbors, channels]
        
        # Flatten neighbors: [batch, output_height, output_width, num_neighbors * channels]
        neighbor_rgb = neighbor_rgb_flat.contiguous().view(batch_size, output_height, output_width, -1)
        
        # Combine features: [batch, output_height, output_width, input_dim]
        patch_features = torch.cat([center_coords, center_rgb, neighbor_rgb], dim=-1)
        
        # Apply model transformation: [batch, output_height, output_width, input_dim] → [batch, output_height, output_width, output_patch_size² * 3]
        delta_rgb_patches = self.model(patch_features)  # [batch, output_height, output_width, output_patch_size² * 3]
        
        # Reshape to [batch, output_height, output_width, output_patch_size, output_patch_size, 3]
        delta_rgb_patches = delta_rgb_patches.view(batch_size, output_height, output_width, 
                                                   self.output_patch_size, self.output_patch_size, 3)
        
        # Constrain delta to [0, 1] via sigmoid, then convert to [-2, 2]
        delta_rgb_patches = torch.sigmoid(delta_rgb_patches)  # [0, 1]
        delta_rgb_patches = 4.0 * delta_rgb_patches - 2.0  # Convert from [0, 1] to [-2, 2]
        
        # OPTIMIZED: Use vectorized output placement with proper broadcasting
        # Initialize full output tensor with same dtype as patches (important for AMP/mixed precision)
        # Very Very important change for amp training issue! 
        # WARNING - AMP training failed, falling back to standard precision: Index put requires the source and destination
        #            dtypes match, got Float for the destination and Half for the source.

        delta_rgb_full = torch.zeros(batch_size, height, width, 3, device=device, dtype=delta_rgb_patches.dtype)
        
        # Reshape patches: [batch, output_height, output_width, output_patch_size, output_patch_size, 3]
        # -> [batch, output_height, output_width, output_patch_size * output_patch_size, 3]
        patches_flat = delta_rgb_patches.view(batch_size, output_height, output_width,
                                             self.output_patch_size * self.output_patch_size, 3)
        
        # Create output indices using proper broadcasting
        # For each patch at (out_h, out_w), we need to place output_patch_size × output_patch_size pixels
        # out_h_all[out_h, out_w, patch_h, patch_w] = out_h_base[out_h] + patch_h_offsets[patch_h]
        # out_w_all[out_h, out_w, patch_h, patch_w] = out_w_base[out_w] + patch_w_offsets[patch_w]
        
        out_h_base = torch.arange(output_height, device=device) * self.output_patch_size  # [output_height]
        out_w_base = torch.arange(output_width, device=device) * self.output_patch_size   # [output_width]
        
        patch_h_offsets = torch.arange(self.output_patch_size, device=device)  # [output_patch_size]
        patch_w_offsets = torch.arange(self.output_patch_size, device=device)  # [output_patch_size]
        
        # Reshape and expand for broadcasting to [output_height, output_width, output_patch_size, output_patch_size]
        # out_h varies along dim 0, out_w along dim 1, patch_h along dim 2, patch_w along dim 3
        out_h_base_4d = out_h_base.view(output_height, 1, 1, 1).expand(-1, output_width, self.output_patch_size, self.output_patch_size)
        patch_h_offsets_4d = patch_h_offsets.view(1, 1, self.output_patch_size, 1).expand(output_height, output_width, -1, self.output_patch_size)
        
        out_w_base_4d = out_w_base.view(1, output_width, 1, 1).expand(output_height, -1, self.output_patch_size, self.output_patch_size)
        patch_w_offsets_4d = patch_w_offsets.view(1, 1, 1, self.output_patch_size).expand(output_height, output_width, self.output_patch_size, -1)
        
        # Combine to get all positions: [output_height, output_width, output_patch_size, output_patch_size]
        out_h_all = out_h_base_4d + patch_h_offsets_4d
        out_w_all = out_w_base_4d + patch_w_offsets_4d
        
        # Flatten: [output_height * output_width * output_patch_size * output_patch_size]
        out_h_flat = out_h_all.flatten()
        out_w_flat = out_w_all.flatten()
        
        # Mask valid positions
        valid_mask = (out_h_flat < height) & (out_w_flat < width)
        
        # Flatten patches: [batch, output_height * output_width * output_patch_size * output_patch_size, 3]
        patches_flat_all = patches_flat.view(batch_size, -1, 3)
        
        # Vectorized placement
        delta_rgb_full[:, out_h_flat[valid_mask], out_w_flat[valid_mask], :] = \
            patches_flat_all[:, valid_mask, :]
        
        # Get pixel colors for all pixels (original input)
        pixel_colors = x.permute(0, 2, 3, 1)  # [batch, height, width, 3]
        
        # Add delta RGB to input RGB
        final_rgb = pixel_colors + delta_rgb_full  # Can be in range [-2, 2]
        
        # Clamp final RGB to [0, 1] to match input range
        final_rgb = torch.clamp(final_rgb, 0.0, 1.0)
        
        # Convert back to [B, C, H, W] format
        final_img = final_rgb.permute(0, 3, 1, 2)
        
        # Apply zero mask to preserve zero areas
        final_img = self.apply_zero_mask(final_img)
        
        return final_img


class ApproachA14(BaseApproach):
    """A14: Patch-based coordinate transformation (similar to A1 but in patches, similar to A13 in structure)
    
    Takes a patch of (x, y) coordinates + RGB values and predicts new (x', y') coordinates for all pixels
    in the patch. Then uses bilinear interpolation to sample RGB from the original image.
    
    Similar to A1: predicts (x', y') and interpolates
    Similar to A13: processes in patches (tiled fashion)
    
    The patch size can vary from 3×3 to the whole image (if patch_size = image_size, it's like A1).
    """
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None,
                 patch_size=3, grid_sample_mode='bilinear'):
        """
        Args:
            patch_size: Size of patch to process at once. Can be any positive integer.
                       Unlike A12/A13, A14 doesn't require odd patch_size because it processes
                       all pixels in the patch equally (no center pixel extraction).
                       Default: 3 (3×3 = 9 pixels per patch)
        """
        if patch_size < 1:
            raise ValueError(f"patch_size must be at least 1, got {patch_size}")
        
        self.patch_size = patch_size
        pad_size = patch_size // 2
        
        # Input: patch of (x, y) coordinates + RGB values
        # For a patch_size × patch_size patch, we have:
        # - patch_size² pixels, each with (x, y, R, G, B) = 5 values
        # Total input: patch_size² × 5
        input_dim = patch_size * patch_size * 5  # (x, y, R, G, B) for each pixel in patch
        
        # Output: (x', y') coordinates for all pixels in the patch
        # Total output: patch_size² × 2
        output_dim = patch_size * patch_size * 2  # (x', y') for each pixel in patch
        
        if model_type == 'neural':
            model = NeuralModel(input_dim, output_dim, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(input_dim, output_dim, model_complexity, hidden_dim=hidden_dim)
        elif model_type == 'kan':
            model = SimpleKAN(input_dim, output_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA14, self).__init__(model, zero_mask, grid_sample_mode)
    
    def forward(self, x):
        """Apply A14 approach: patch-based coordinate transformation with bilinear interpolation
        
        Processes the image in patches:
        1. Extract patches of (x, y) coordinates + RGB values
        2. Predict (x', y') for all pixels in each patch
        3. Use bilinear interpolation to sample RGB from original image
        
        Note: patch_size should be small (3, 5, 7, etc.) for practical use.
        For whole-image processing, use A1 instead.
        """
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Process in patches
        pad_size = self.patch_size // 2
        
        # Pad the image with reflection padding
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Create coordinate grid for the original (unpadded) image
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Pad coordinate grid similarly
        coord_grid_padded = F.pad(coord_grid.permute(0, 3, 1, 2), 
                                  (pad_size, pad_size, pad_size, pad_size), 
                                  mode='reflect').permute(0, 2, 3, 1)
        
        # Calculate output dimensions (process with stride = patch_size for non-overlapping patches)
        output_height = (height + self.patch_size - 1) // self.patch_size
        output_width = (width + self.patch_size - 1) // self.patch_size
        
        # Use unfold to extract all patches at once (similar to A13)
        # Extract patches with stride = patch_size (non-overlapping)
        x_unfolded = F.unfold(x_padded, kernel_size=self.patch_size, stride=self.patch_size)
        # x_unfolded: [batch, channels * patch_size * patch_size, num_patches]
        
        # Calculate number of patches
        h_padded, w_padded = x_padded.shape[2], x_padded.shape[3]
        num_patches_h = (h_padded - self.patch_size) // self.patch_size + 1
        num_patches_w = (w_padded - self.patch_size) // self.patch_size + 1
        num_patches = num_patches_h * num_patches_w
        
        # Reshape to [batch, num_patches, patch_size, patch_size, channels]
        x_unfolded = x_unfolded.view(batch_size, channels, self.patch_size * self.patch_size, num_patches)
        x_unfolded = x_unfolded.permute(0, 3, 1, 2)  # [batch, num_patches, channels, patch_size²]
        x_unfolded = x_unfolded.contiguous().view(batch_size, num_patches, channels, 
                                                   self.patch_size, self.patch_size)
        x_unfolded = x_unfolded.permute(0, 1, 3, 4, 2)  # [batch, num_patches, patch_size, patch_size, channels]
        
        # Extract coordinates for patches using unfold
        coord_unfolded = F.unfold(coord_grid_padded.permute(0, 3, 1, 2), 
                                  kernel_size=self.patch_size, stride=self.patch_size)
        coord_unfolded = coord_unfolded.view(batch_size, 2, self.patch_size * self.patch_size, num_patches)
        coord_unfolded = coord_unfolded.permute(0, 3, 1, 2)  # [batch, num_patches, 2, patch_size²]
        coord_unfolded = coord_unfolded.contiguous().view(batch_size, num_patches, 2, 
                                                          self.patch_size, self.patch_size)
        coord_unfolded = coord_unfolded.permute(0, 1, 3, 4, 2)  # [batch, num_patches, patch_size, patch_size, 2]
        
        # Reshape to [batch, num_patches_h, num_patches_w, patch_size, patch_size, ...]
        x_patches = x_unfolded.view(batch_size, num_patches_h, num_patches_w,
                                    self.patch_size, self.patch_size, channels)
        coord_patches = coord_unfolded.view(batch_size, num_patches_h, num_patches_w,
                                           self.patch_size, self.patch_size, 2)
        
        # Crop to output_height × output_width
        x_patches = x_patches[:, :output_height, :output_width, :, :, :]
        coord_patches = coord_patches[:, :output_height, :output_width, :, :, :]
        
        # Combine coordinates and colors for each patch: [batch, output_height, output_width, patch_size, patch_size, 5]
        patch_combined = torch.cat([coord_patches, x_patches], dim=-1)
        
        # Flatten each patch: [batch, output_height, output_width, patch_size² * 5]
        patch_flat = patch_combined.contiguous().view(batch_size, output_height, output_width, -1)
        
        # Apply model: [batch, output_height, output_width, patch_size² * 5] → [batch, output_height, output_width, patch_size² * 2]
        transformed_coords_patches = self.model(patch_flat)
        
        # Reshape to [batch, output_height, output_width, patch_size, patch_size, 2]
        transformed_coords_patches = transformed_coords_patches.view(batch_size, output_height, output_width,
                                                                     self.patch_size, self.patch_size, 2)
        
        # Clamp coordinates to [0, 1]
        transformed_coords_patches = torch.clamp(transformed_coords_patches, 0.0, 1.0)
        
        # Convert to grid_sample format [-1, 1]
        sample_coords_patches = 2.0 * transformed_coords_patches - 1.0
        
        # Now we need to interpolate RGB for each pixel in each patch
        # We'll use grid_sample, but we need to reshape properly
        # For each patch, we have patch_size² pixels with (x', y') coordinates
        # We need to sample from the original image x
        
        # Initialize full coordinate grid
        sample_coords_full = torch.zeros(batch_size, height, width, 2, device=device, dtype=sample_coords_patches.dtype)
        
        # Place patches in correct positions (similar to A13 output placement)
        out_h_base = torch.arange(output_height, device=device) * self.patch_size
        out_w_base = torch.arange(output_width, device=device) * self.patch_size
        
        patch_h_offsets = torch.arange(self.patch_size, device=device)
        patch_w_offsets = torch.arange(self.patch_size, device=device)
        
        # Create indices for all positions
        out_h_base_4d = out_h_base.view(output_height, 1, 1, 1).expand(-1, output_width, self.patch_size, self.patch_size)
        patch_h_offsets_4d = patch_h_offsets.view(1, 1, self.patch_size, 1).expand(output_height, output_width, -1, self.patch_size)
        out_w_base_4d = out_w_base.view(1, output_width, 1, 1).expand(output_height, -1, self.patch_size, self.patch_size)
        patch_w_offsets_4d = patch_w_offsets.view(1, 1, 1, self.patch_size).expand(output_height, output_width, self.patch_size, -1)
        
        out_h_all = out_h_base_4d + patch_h_offsets_4d
        out_w_all = out_w_base_4d + patch_w_offsets_4d
        
        out_h_flat = out_h_all.flatten()
        out_w_flat = out_w_all.flatten()
        valid_mask = (out_h_flat < height) & (out_w_flat < width)
        
        # Flatten sample_coords_patches: [batch, output_height * output_width * patch_size², 2]
        sample_coords_flat_all = sample_coords_patches.view(batch_size, -1, 2)
        
        # Place coordinates in full grid
        sample_coords_full[:, out_h_flat[valid_mask], out_w_flat[valid_mask], :] = \
            sample_coords_flat_all[:, valid_mask, :]
        
        # Apply bilinear sampling using grid_sample
        transformed_img = grid_sample(
            torch.clamp(x, 0.0, 1.0),
            sample_coords_full,
            mode=self.grid_sample_mode,
            padding_mode='border',
            align_corners=True
        )
        
        transformed_img = self.apply_zero_mask(transformed_img)
        return transformed_img


class ApproachA15(BaseApproach):
    """A15: CoordConv stem + per-pixel MLP head -> delta RGB.

    Replaces A12's hand-crafted 3x3 patch extraction with a learned conv stack.
    Convs share weights across pixels (param-efficient); CoordConv channels (x, y, r)
    give the spatial cue that pure conv equivariance otherwise hides.

    Output is added to the input image as a delta (same residual scheme as A12).
    Effective receptive field per output pixel = 1 + len(conv_channels) * (kernel_size - 1).
    """
    def __init__(self, model_type='neural', model_complexity='complex',
                 hidden_dim=64, zero_mask=None,
                 conv_channels=(16, 16, 24),
                 kernel_size=3,
                 num_mlp_layers=1,
                 use_coord=True,
                 use_radial=True,
                 activation='relu'):
        """
        Args:
            conv_channels: list/tuple of output channels per conv layer, or comma-string.
                           e.g., [16, 16, 24] -> 3 convs (3->16, 16->16, 16->24).
                           Number of conv layers = len(conv_channels).
            kernel_size: odd int >= 1 (default 3).
            hidden_dim: width of each hidden layer in the per-pixel MLP head.
            num_mlp_layers: number of HIDDEN MLP layers in the head (>=1).
                            E.g., 1 means [final_conv_channels -> hidden_dim -> 3].
            use_coord: append normalized (x, y) channels in [-1, 1] to the input (CoordConv).
            use_radial: append normalized radial r = sqrt(x^2 + y^2) channel.
            activation: 'relu' or 'gelu'.
            model_type / model_complexity: ignored (kept for API parity with other approaches).
        """
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(f"kernel_size must be odd and >= 1, got {kernel_size}")

        # Accept list, tuple, or comma-separated string (handy from JSON CLI configs)
        if isinstance(conv_channels, (list, tuple)):
            conv_channels = [int(c) for c in conv_channels]
        elif isinstance(conv_channels, str):
            conv_channels = [int(c.strip()) for c in conv_channels.split(',') if c.strip()]
        else:
            raise ValueError(f"conv_channels must be list/tuple/comma-string, got {type(conv_channels)}")
        if len(conv_channels) < 1:
            raise ValueError("conv_channels must have at least 1 entry")
        if int(num_mlp_layers) < 1:
            raise ValueError(f"num_mlp_layers must be >= 1, got {num_mlp_layers}")

        self.conv_channels = conv_channels
        self.kernel_size = int(kernel_size)
        self.use_coord = bool(use_coord)
        self.use_radial = bool(use_radial)
        self.num_extra_channels = (2 if self.use_coord else 0) + (1 if self.use_radial else 0)

        if activation.lower() == 'relu':
            act = lambda: nn.ReLU(inplace=False)
        elif activation.lower() == 'gelu':
            act = lambda: nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        in_ch = 3 + self.num_extra_channels
        pad = self.kernel_size // 2

        conv_layers = []
        prev_ch = in_ch
        for out_ch in conv_channels:
            conv_layers.append(nn.Conv2d(prev_ch, out_ch, kernel_size=self.kernel_size, padding=pad))
            conv_layers.append(act())
            prev_ch = out_ch
        conv_stem = nn.Sequential(*conv_layers)

        # Per-pixel MLP head implemented as 1x1 convs (no patch extraction needed).
        mlp_layers = []
        prev_ch = conv_channels[-1]
        for _ in range(int(num_mlp_layers)):
            mlp_layers.append(nn.Conv2d(prev_ch, int(hidden_dim), kernel_size=1))
            mlp_layers.append(act())
            prev_ch = int(hidden_dim)
        mlp_layers.append(nn.Conv2d(prev_ch, 3, kernel_size=1))
        mlp_head = nn.Sequential(*mlp_layers)

        model = nn.Sequential(conv_stem, mlp_head)
        super(ApproachA15, self).__init__(model, zero_mask)

        # Lazy cache for the (x, y, r) channel tensor. Keyed on (H, W, device, dtype).
        self._coord_cache_key = None
        self._coord_cache_tensor = None

    def _build_coord_channels(self, batch_size, height, width, device, dtype):
        if self.num_extra_channels == 0:
            return None
        key = (height, width, str(device), str(dtype))
        if self._coord_cache_key != key:
            chans = []
            ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            if self.use_coord:
                chans.append(grid_x)
                chans.append(grid_y)
            if self.use_radial:
                chans.append(torch.sqrt(grid_x * grid_x + grid_y * grid_y))
            coord = torch.stack(chans, dim=0).unsqueeze(0)  # [1, C_extra, H, W]
            self._coord_cache_key = key
            self._coord_cache_tensor = coord
        return self._coord_cache_tensor.expand(batch_size, -1, -1, -1)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        if self.num_extra_channels > 0:
            coord = self._build_coord_channels(batch_size, height, width, x.device, x.dtype)
            x_in = torch.cat([x, coord], dim=1)
        else:
            x_in = x

        delta = self.model(x_in)  # [B, 3, H, W]
        # Same range mapping as A12: sigmoid -> [0, 1] -> [-2, 2]
        delta = torch.sigmoid(delta)
        delta = 4.0 * delta - 2.0

        final = torch.clamp(x + delta, 0.0, 1.0)
        final = self.apply_zero_mask(final)
        return final


# =============================================================================
# TWO-NETWORK APPROACH IMPLEMENTATIONS (Approaches 5, 6, 7)
# =============================================================================

class ApproachA5(BaseApproach):
    """A5: net1(x,y) → x',y' → im'; net2(x',y',RGB) → predicted RGB → im''"""
    def __init__(self, net1_type='neural', net1_complexity='medium', net2_type='neural', net2_complexity='medium', 
                 hidden_dim=256, net1_hidden_dim=None, net2_hidden_dim=None, zero_mask=None, loss_variant='5-1',
                 grid_sample_mode='bilinear', net1_pretrained_path=None, net2_pretrained_path=None, 
                 freeze_net1=False, freeze_net2=False, training_stage='both'):
        # Use separate hidden dimensions if provided, otherwise fall back to shared hidden_dim
        net1_hdim = net1_hidden_dim if net1_hidden_dim is not None else hidden_dim
        net2_hdim = net2_hidden_dim if net2_hidden_dim is not None else hidden_dim
        
        # net1: coordinates only (x,y → x',y')
        if net1_type == 'neural':
            net1 = NeuralModel(2, 2, net1_hdim, net1_complexity)
        elif net1_type == 'parametric':
            net1 = ParametricModel(2, 2, net1_complexity, hidden_dim=net1_hdim)
        else:
            raise ValueError(f"Unsupported net1 type: {net1_type}")
        
        # net2: coordinates + colors (x',y',RGB → predicted RGB)
        if net2_type == 'neural':
            net2 = NeuralModel(5, 3, net2_hdim, net2_complexity)
        elif net2_type == 'parametric':
            net2 = ParametricModel(5, 3, net2_complexity, hidden_dim=net2_hdim)
        else:
            raise ValueError(f"Unsupported net2 type: {net2_type}")
            
        # Use net1 as the main model for BaseApproach, store net2 separately
        super(ApproachA5, self).__init__(net1, zero_mask, grid_sample_mode)
        self.net1 = net1
        self.net2 = net2
        self.loss_variant = loss_variant  # '5-1' or '5-2'
        self.net1_pretrained_path = net1_pretrained_path
        self.net2_pretrained_path = net2_pretrained_path
        self.freeze_net1 = freeze_net1
        self.freeze_net2 = freeze_net2
        self.training_stage = training_stage  # 'net1_only', 'net2_only', 'both'
        
        # Load pretrained net1 if path is provided
        if net1_pretrained_path is not None:
            self.load_pretrained_net1(net1_pretrained_path)
            # Verify that weights were actually loaded
            self.verify_net1_weights()
            # Compare with original pretrained model
            self.compare_with_pretrained_source(net1_pretrained_path)
        
        # Load pretrained net2 if path is provided
        if net2_pretrained_path is not None:
            self.load_pretrained_net2(net2_pretrained_path)
        
        # Set training mode based on stage
        self.set_training_stage(training_stage)
        
        # Apply freezing if requested
        if freeze_net1 or freeze_net2:
            self.freeze_networks(freeze_net1=freeze_net1, freeze_net2=freeze_net2)
    
    def load_pretrained_net1(self, path):
        """Load pretrained net1 weights"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            
            # Get the state dict
            if 'net1_state_dict' in checkpoint:
                state_dict = checkpoint['net1_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Strip "model." prefix if present
            if any(key.startswith('model.') for key in state_dict.keys()):
                state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
                print("Stripped 'model.' prefix from state dict keys")
            
            # For A5, we need to load the full model state dict into net1
            # The checkpoint contains a complete parametric model that should be loaded into net1
            print(f"🔍 Loading pretrained model into net1 for A5 approach")
            print(f"📊 Checkpoint keys: {list(state_dict.keys())}")
            print(f"📊 Net1 keys: {list(self.net1.state_dict().keys())}")
            
            # Store weights BEFORE loading for comparison
            weights_before = {}
            for name, param in self.net1.named_parameters():
                weights_before[name] = param.data.clone()
            
            # Try to load the state dict directly (same approach as corrected_model_test.py)
            try:
                self.net1.load_state_dict(state_dict, strict=False)
                print(f"✅ Successfully loaded pretrained weights into net1")
                
                # Verify that weights actually changed by comparing before/after
                print("🔍 Verifying net1 weights changed after loading:")
                weights_changed = False
                for name, param in self.net1.named_parameters():
                    param_before = weights_before[name]
                    param_after = param.data
                    
                    # Check if weights actually changed
                    if not torch.equal(param_before, param_after):
                        weights_changed = True
                        param_mean = param_after.mean().item()
                        param_std = param_after.std().item()
                        print(f"   ✅ {name}: CHANGED - mean={param_mean:.6f}, std={param_std:.6f}")
                    else:
                        print(f"   ❌ {name}: UNCHANGED - weights not loaded!")
                
                if weights_changed:
                    print("🎉 SUCCESS: Net1 weights were actually loaded and changed!")
                    
                else:
                    print("⚠️  WARNING: Net1 weights did not change - loading may have failed!")
                    
            except Exception as e:
                print(f"❌ Could not load pretrained weights: {e}")
                print("🔄 Using randomly initialized net1")
        else:
            raise FileNotFoundError(f"Pretrained net1 path not found: {path}")
    
    def set_training_stage(self, stage):
        """Set which networks are trainable based on training stage"""
        if stage == 'net1_only':
            # Only train net1, freeze net2
            for param in self.net1.parameters():
                param.requires_grad = True
            for param in self.net2.parameters():
                param.requires_grad = False
        elif stage == 'net2_only':
            # Only train net2, freeze net1
            for param in self.net1.parameters():
                param.requires_grad = False
            for param in self.net2.parameters():
                param.requires_grad = True
        elif stage == 'both':
            # Train both networks
            for param in self.net1.parameters():
                param.requires_grad = True
            for param in self.net2.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown training stage: {stage}")
    
    def verify_net1_weights(self):
        """Verify that net1 is using pretrained weights and not random initialization"""
        print("🔍 Verifying net1 weights...")
        
        # Check if net1 has pretrained weights by examining parameter statistics
        net1_params = list(self.net1.parameters())
        if not net1_params:
            print("❌ No parameters found in net1")
            return False
        
        # Check parameter statistics
        param_stats = []
        for i, param in enumerate(net1_params):
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()
            param_stats.append((mean_val, std_val))
            print(f"  Layer {i}: mean={mean_val:.6f}, std={std_val:.6f}")
        
        # If all parameters are very close to zero or have very small std, they might be uninitialized
        all_near_zero = all(abs(mean) < 1e-6 and std < 1e-6 for mean, std in param_stats)
        if all_near_zero:
            print("⚠️  WARNING: All net1 parameters are near zero - might be uninitialized!")
            return False
        else:
            print("✅ Net1 parameters appear to have non-zero values - likely pretrained weights")
            return True
    
    def compare_with_pretrained_source(self, pretrained_path):
        """Compare loaded net1 weights with the original pretrained model"""
        print("🔍 Comparing with original pretrained model...")
        
        if not os.path.exists(pretrained_path):
            print(f"❌ Pretrained path not found: {pretrained_path}")
            return False
        
        # Load the original checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Get the state dict from checkpoint
        if 'net1_state_dict' in checkpoint:
            original_state_dict = checkpoint['net1_state_dict']
        elif 'model_state_dict' in checkpoint:
            original_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            original_state_dict = checkpoint['state_dict']
        else:
            original_state_dict = checkpoint
        
        # Strip "model." prefix if present
        if any(key.startswith('model.') for key in original_state_dict.keys()):
            original_state_dict = {key.replace('model.', ''): value for key, value in original_state_dict.items()}
        
        # Get current net1 parameters
        current_params = list(self.net1.parameters())
        current_state_dict = self.net1.state_dict()
        
        print(f"📊 Original model has {len(original_state_dict)} parameters")
        print(f"📊 Current net1 has {len(current_state_dict)} parameters")
        
        # Compare matching parameters
        matching_keys = set(current_state_dict.keys()).intersection(set(original_state_dict.keys()))
        print(f"📊 Matching parameters: {len(matching_keys)}")
        
        if matching_keys:
            print("🔍 Comparing parameter statistics:")
            for i, key in enumerate(sorted(matching_keys)):
                original_param = original_state_dict[key]
                current_param = current_state_dict[key]
                
                # Calculate statistics
                orig_mean = original_param.mean().item()
                orig_std = original_param.std().item()
                curr_mean = current_param.mean().item()
                curr_std = current_param.std().item()
                
                # Check if they're identical
                are_identical = torch.equal(original_param, current_param)
                
                print(f"  {key}:")
                print(f"    Original: mean={orig_mean:.6f}, std={orig_std:.6f}")
                print(f"    Current:  mean={curr_mean:.6f}, std={curr_std:.6f}")
                print(f"    Identical: {'✅ YES' if are_identical else '❌ NO'}")
                
                if not are_identical:
                    # Calculate difference
                    diff = torch.abs(original_param - current_param).max().item()
                    print(f"    Max diff: {diff:.8f}")
        else:
            print("❌ No matching parameters found between original and current model")
        
        return len(matching_keys) > 0
    
    def load_pretrained_net2(self, path):
        """Load pretrained net2 weights"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            
            # Get the state dict
            if 'net2_state_dict' in checkpoint:
                state_dict = checkpoint['net2_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Strip "model." prefix if present
            if any(key.startswith('model.') for key in state_dict.keys()):
                state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
                print("Stripped 'model.' prefix from net2 state dict keys")
            
            # Store weights BEFORE loading for comparison
            weights_before = {}
            for name, param in self.net2.named_parameters():
                weights_before[name] = param.data.clone()
            
            # Try to load the state dict directly (same approach as corrected_model_test.py)
            try:
                self.net2.load_state_dict(state_dict, strict=False)
                print(f"✅ Successfully loaded pretrained weights into net2")
                
                # Verify that weights actually changed by comparing before/after
                print("🔍 Verifying net2 weights changed after loading:")
                weights_changed = False
                for name, param in self.net2.named_parameters():
                    param_before = weights_before[name]
                    param_after = param.data
                    
                    # Check if weights actually changed
                    if not torch.equal(param_before, param_after):
                        weights_changed = True
                        param_mean = param_after.mean().item()
                        param_std = param_after.std().item()
                        print(f"   ✅ {name}: CHANGED - mean={param_mean:.6f}, std={param_std:.6f}")
                    else:
                        print(f"   ❌ {name}: UNCHANGED - weights not loaded!")
                
                if weights_changed:
                    print("🎉 SUCCESS: Net2 weights were actually loaded and changed!")
                else:
                    print("⚠️  WARNING: Net2 weights did not change - loading may have failed!")
                    
            except Exception as e:
                print(f"❌ Could not load pretrained weights: {e}")
                print("🔄 Using randomly initialized net2")
        else:
            raise FileNotFoundError(f"Pretrained net2 path not found: {path}")
    
    def freeze_networks(self, freeze_net1=False, freeze_net2=False):
        """Freeze or unfreeze networks based on flags"""
        if freeze_net1:
            for param in self.net1.parameters():
                param.requires_grad = False
            print(f"🔒 Net1 frozen: {sum(p.numel() for p in self.net1.parameters() if not p.requires_grad):,} parameters")
            
        else:
            for param in self.net1.parameters():
                param.requires_grad = True
            print(f"🔓 Net1 trainable: {sum(p.numel() for p in self.net1.parameters() if p.requires_grad):,} parameters")
        
        if freeze_net2:
            for param in self.net2.parameters():
                param.requires_grad = False
            print(f"🔒 Net2 frozen: {sum(p.numel() for p in self.net2.parameters() if not p.requires_grad):,} parameters")
        else:
            for param in self.net2.parameters():
                param.requires_grad = True
            print(f"🔓 Net2 trainable: {sum(p.numel() for p in self.net2.parameters() if p.requires_grad):,} parameters")
    
    
    def save_networks_separately(self, net1_path, net2_path):
        """Save net1 and net2 separately"""
        # Save net1
        torch.save({
            'net1_state_dict': self.net1.state_dict(),
            'net1_type': getattr(self.net1, 'model_type', 'unknown'),
            'net1_complexity': getattr(self.net1, 'complexity', 'unknown')
        }, net1_path)
        
        # Save net2
        torch.save({
            'net2_state_dict': self.net2.state_dict(),
            'net2_type': getattr(self.net2, 'model_type', 'unknown'),
            'net2_complexity': getattr(self.net2, 'complexity', 'unknown')
        }, net2_path)
        
        print(f"Saved net1 to {net1_path}")
        print(f"Saved net2 to {net2_path}")
    
    def forward(self, x):
        """Apply A5 approach: Two-network coordinate + color transformation"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Create coordinate grid and ensure in [0,1]
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Step 1: net1 transforms coordinates (x,y → x',y')
        transformed_coords = self.net1(coord_grid)
        
        # Step 2: Create im' using bilinear interpolation with transformed coordinates
        sample_coords = 2.0 * transformed_coords - 1.0
        im_prime = F.grid_sample(
            x,
            sample_coords,
            mode='bilinear', 
            padding_mode='border',
            align_corners=True,
        )
        
        # Apply zero mask to im_prime
        im_prime = self.apply_zero_mask(im_prime)
        
        # Handle different training stages
        if self.training_stage == 'net1_only':
            # Only return im_prime for net1 training
            return {
                'im_prime': im_prime,
                'transformed_coords': transformed_coords,
                'final': im_prime
            }
        
        # For net2_only and both stages, continue with net2
        # Step 3: Prepare input for net2 (x',y',RGB)
        pixel_colors = im_prime.permute(0, 2, 3, 1)  # [batch, height, width, 3]
        net2_input = torch.cat([transformed_coords, pixel_colors], dim=-1)
        
        # Step 4: net2 predicts actual RGB values (not delta)
        predicted_rgb = self.net2(net2_input)
        
        # Step 5: Apply sigmoid to ensure RGB values are in [0,1] range
        predicted_rgb = torch.sigmoid(predicted_rgb)  # Constrain to [0,1] range
        
        # Step 6: Use predicted RGB directly as im_double_prime
        im_double_prime = predicted_rgb.permute(0, 3, 1, 2)  # Convert to [B, C, H, W]
        
        # Apply zero mask to preserve zero areas
        im_double_prime = self.apply_zero_mask(im_double_prime)
        
        # Return both images for loss computation
        return {
            'im_prime': im_prime,
            'im_double_prime': im_double_prime,
            'transformed_coords': transformed_coords,
            'predicted_rgb': predicted_rgb,  # Changed from delta_rgb to predicted_rgb
            'final': im_double_prime  # Final output for inference
        }


class ApproachA6(BaseApproach):
    """A6: net1(x,y) → x',y' → im'; net2(RGB) → delta RGB → im''"""
    def __init__(self, net1_type='neural', net1_complexity='medium', net2_type='neural', net2_complexity='medium', 
                 hidden_dim=256, net1_hidden_dim=None, net2_hidden_dim=None, zero_mask=None, loss_variant='6-1',
                 grid_sample_mode='bilinear'):
        # Use separate hidden dimensions if provided, otherwise fall back to shared hidden_dim
        net1_hdim = net1_hidden_dim if net1_hidden_dim is not None else hidden_dim
        net2_hdim = net2_hidden_dim if net2_hidden_dim is not None else hidden_dim
        
        # net1: coordinates only (x,y → x',y')
        if net1_type == 'neural':
            net1 = NeuralModel(2, 2, net1_hdim, net1_complexity)
        elif net1_type == 'parametric':
            net1 = ParametricModel(2, 2, net1_complexity, hidden_dim=net1_hdim)
        else:
            raise ValueError(f"Unsupported net1 type: {net1_type}")
        
        # net2: colors only (RGB → delta RGB)
        if net2_type == 'neural':
            net2 = NeuralModel(3, 3, net2_hdim, net2_complexity)
        elif net2_type == 'parametric':
            net2 = ParametricModel(3, 3, net2_complexity, hidden_dim=net2_hdim)
        else:
            raise ValueError(f"Unsupported net2 type: {net2_type}")
            
        # Use net1 as the main model for BaseApproach, store net2 separately
        super(ApproachA6, self).__init__(net1, zero_mask, grid_sample_mode)
        self.net1 = net1
        self.net2 = net2
        self.loss_variant = loss_variant  # '6-1' or '6-2'
    
    def forward(self, x):
        """Apply A6 approach: Two-network coordinate transformation + separate color transformation"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Create coordinate grid
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Step 1: net1 transforms coordinates (x,y → x',y')
        transformed_coords = self.net1(coord_grid)
        
        # Step 2: Create im' using bilinear interpolation with transformed coordinates
        sample_coords = 2.0 * transformed_coords - 1.0
        im_prime = grid_sample(
            torch.clamp(x, 0.0, 1.0),
            sample_coords,
            mode=self.grid_sample_mode, 
            padding_mode='zeros',
            align_corners=True,
        )
        
        # Step 3: Prepare input for net2 (RGB only from im')
        pixel_colors = im_prime.permute(0, 2, 3, 1)  # [batch, height, width, 3]
        
        # Step 4: net2 transforms RGB to get delta RGB
        delta_rgb = self.net2(pixel_colors)
        
        # Step 5: Create im'' by applying delta RGB to im'
        delta_rgb = 2.0 * delta_rgb - 1.0  # Convert from [0,1] to [-1,1]
        im_double_prime = im_prime + delta_rgb.permute(0, 3, 1, 2)
        
        # Apply zero mask to preserve zero areas
        im_prime = self.apply_zero_mask(im_prime)
        im_double_prime = self.apply_zero_mask(im_double_prime)
        
        # Return both images for loss computation
        return {
            'im_prime': im_prime,
            'im_double_prime': im_double_prime,
            'final': im_double_prime  # Final output for inference
        }


class ApproachA7(BaseApproach):
    """A7: net(x,y) → delta RGB → img'"""
    def __init__(self, model_type='neural', model_complexity='medium', hidden_dim=256, zero_mask=None):
        if model_type == 'neural':
            model = NeuralModel(2, 3, hidden_dim, model_complexity)
        elif model_type == 'parametric':
            model = ParametricModel(2, 3, model_complexity, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        super(ApproachA7, self).__init__(model, zero_mask)
    
    def forward(self, x):
        """Apply A7 approach: coordinate-based color transformation only"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Create coordinate grid
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Apply model transformation: coordinates → delta RGB
        delta_rgb = self.model(coord_grid)
        
        # Apply delta RGB to original image
        delta_rgb = 2.0 * delta_rgb - 1.0  # Convert from [0,1] to [-1,1]
        final_img = x + delta_rgb.permute(0, 3, 1, 2)
        
        # Apply zero mask to preserve zero areas
        final_img = self.apply_zero_mask(final_img)
        
        return final_img


class ApproachA8(BaseApproach):
    """A8: net1(x,y,RGB) → delta RGB → im'; net2(x,y) → x',y' → im''"""
    def __init__(self, net1_type='neural', net1_complexity='medium', net2_type='neural', net2_complexity='medium', 
                 hidden_dim=256, net1_hidden_dim=None, net2_hidden_dim=None, zero_mask=None, loss_variant='8-1',
                 grid_sample_mode='bilinear'):
        # Use separate hidden dimensions if provided, otherwise fall back to shared hidden_dim
        net1_hdim = net1_hidden_dim if net1_hidden_dim is not None else hidden_dim
        net2_hdim = net2_hidden_dim if net2_hidden_dim is not None else hidden_dim

        print(f"net1_hdim: {net1_hdim}, net2_hdim: {net2_hdim}")
        print(f"net1_type: {net1_type}, net2_type: {net2_type}")
        print(f"net1_complexity: {net1_complexity}, net2_complexity: {net2_complexity}")
        
        
        # net1: coordinates + colors (x,y,RGB → delta RGB)
        if net1_type == 'neural':
            net1 = NeuralModel(5, 3, net1_hdim, net1_complexity)
        elif net1_type == 'parametric':
            net1 = ParametricModel(5, 3, net1_complexity, hidden_dim=net1_hdim)
        else:
            raise ValueError(f"Unsupported net1 type: {net1_type}")
        
        # net2: coordinates only (x,y → x',y')
        if net2_type == 'neural':
            net2 = NeuralModel(2, 2, net2_hdim, net2_complexity)
        elif net2_type == 'parametric':
            net2 = ParametricModel(2, 2, net2_complexity, hidden_dim=net2_hdim)
        else:
            raise ValueError(f"Unsupported net2 type: {net2_type}")
            
        # Use net1 as the main model for BaseApproach, store net2 separately
        super(ApproachA8, self).__init__(net1, zero_mask, grid_sample_mode)
        self.net1 = net1
        self.net2 = net2
        self.loss_variant = loss_variant  # '8-1' or '8-2'
    
    def forward(self, x):
        """Apply A8 approach: Two-network color + coordinate transformation"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Create coordinate grid
        coord_grid = self.create_coordinate_grid(batch_size, height, width, device)
        
        # Step 1: net1 transforms colors (x,y,RGB → delta RGB)
        # Prepare input for net1: concatenate coordinates and RGB values
        x_flat = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        coord_rgb_input = torch.cat([coord_grid, x_flat], dim=-1)  # [B, H, W, 5]
        
        # Apply net1 to get delta RGB
        delta_rgb = self.net1(coord_rgb_input)
        # Keep deltas in [0,1] before mapping to [-1,1] for stability
        delta_rgb = torch.clamp(delta_rgb, 0.0, 1.0)
        
        # Apply delta RGB to create im' (new image)
        delta_rgb = 2.0 * delta_rgb - 1.0  # Convert from [0,1] to [-1,1]
        im_prime = x + delta_rgb.permute(0, 3, 1, 2)
        
        # Step 2: net2 transforms coordinates (x,y → x',y')
        transformed_coords = self.net2(coord_grid)
        transformed_coords = torch.clamp(transformed_coords, 0.0, 1.0)
        
        # Step 3: Create im'' using bilinear interpolation with transformed coordinates
        # Normalize coordinates to [-1, 1] for grid_sample
        normalized_coords = 2.0 * transformed_coords - 1.0
        normalized_coords = normalized_coords.permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        # Use grid_sample for bilinear interpolation
        im_double_prime = grid_sample(im_prime, normalized_coords.permute(0, 2, 3, 1),
                                       mode=self.grid_sample_mode, padding_mode='border', align_corners=True)
        
        # Apply zero mask to preserve zero areas
        im_prime = self.apply_zero_mask(im_prime)
        im_double_prime = self.apply_zero_mask(im_double_prime)
        
        # Return both images for loss computation
        return {
            'im_prime': im_prime,
            'im_double_prime': im_double_prime,
            'final': im_double_prime  # Final output for inference
        }


# =============================================================================
# FACTORY FUNCTIONS - For Easy Extension
# =============================================================================

# def create_model(model_type='neural', input_dim=2, output_dim=2, model_complexity='medium', **kwargs):
#     """Factory function to create transformation models"""
#     if model_type == 'neural':
#         return NeuralModel(input_dim, output_dim, model_complexity, **kwargs)
#     elif model_type == 'parametric':
#         return ParametricModel(input_dim, output_dim, model_complexity, **kwargs)
#     elif model_type == 'kan':
#         return SimpleKAN(input_dim, output_dim, model_complexity, **kwargs)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")


def create_approach(approach_type='A1', model_type='neural', **kwargs):
    """Factory function to create transformation approaches"""
    # Extract zero_mask and grid_sample config separately as they're passed to BaseApproach constructor
    zero_mask = kwargs.get('zero_mask', None)
    grid_sample_mode = kwargs.get('grid_sample_mode', 'bilinear')

    # Remove these from kwargs to avoid passing them twice
    kwargs_without_base = {k: v for k, v in kwargs.items() 
                          if k not in ['zero_mask', 'grid_sample_mode']}

    if approach_type == 'A1':
        return ApproachA1(model_type, zero_mask=zero_mask, 
                         grid_sample_mode=grid_sample_mode, **kwargs_without_base)
    elif approach_type == 'A2':
        return ApproachA2(model_type, zero_mask=zero_mask, 
                         grid_sample_mode=grid_sample_mode, **kwargs_without_base)
    elif approach_type == 'A3':
        return ApproachA3(model_type, zero_mask=zero_mask, 
                         grid_sample_mode=grid_sample_mode, **kwargs_without_base)
    elif approach_type == 'A4':
        return ApproachA4(model_type, zero_mask=zero_mask, **kwargs_without_base)
    elif approach_type == 'A5':
        # For A5, model_type can be specified for both networks or individually
        net1_type = kwargs_without_base.pop('net1_type', model_type)
        net2_type = kwargs_without_base.pop('net2_type', model_type)
        # Remove model_complexity as A5 uses net1_complexity and net2_complexity
        kwargs_without_base.pop('model_complexity', None)
        return ApproachA5(net1_type=net1_type, net2_type=net2_type, zero_mask=zero_mask, 
                         grid_sample_mode=grid_sample_mode, **kwargs_without_base)
    elif approach_type == 'A6':
        # For A6, model_type can be specified for both networks or individually
        net1_type = kwargs_without_base.pop('net1_type', model_type)
        net2_type = kwargs_without_base.pop('net2_type', model_type)
        # Remove model_complexity as A6 uses net1_complexity and net2_complexity
        kwargs_without_base.pop('model_complexity', None)
        return ApproachA6(net1_type=net1_type, net2_type=net2_type, zero_mask=zero_mask, 
                         grid_sample_mode=grid_sample_mode, **kwargs_without_base)
    elif approach_type == 'A7':
        return ApproachA7(model_type, zero_mask=zero_mask, **kwargs_without_base)
    elif approach_type == 'A8':
        # For A8, model_type can be specified for both networks or individually
        net1_type = kwargs_without_base.pop('net1_type', model_type)
        net2_type = kwargs_without_base.pop('net2_type', model_type)
        # Remove model_complexity as A8 uses net1_complexity and net2_complexity
        kwargs_without_base.pop('model_complexity', None)
        # Ensure net1_complexity and net2_complexity are passed through
        return ApproachA8(net1_type=net1_type, net2_type=net2_type, zero_mask=zero_mask, 
                         grid_sample_mode=grid_sample_mode, **kwargs_without_base)
    elif approach_type == 'A9':
        return ApproachA9(model_type, zero_mask=zero_mask, **kwargs_without_base)
    elif approach_type == 'A10':
        return ApproachA10(model_type, zero_mask=zero_mask, **kwargs_without_base)
    elif approach_type == 'A11':
        return ApproachA11(model_type, zero_mask=zero_mask, **kwargs_without_base)
    elif approach_type == 'A12':
        return ApproachA12(model_type, zero_mask=zero_mask, **kwargs_without_base)
    elif approach_type == 'A13':
        return ApproachA13(model_type, zero_mask=zero_mask, **kwargs_without_base)
    elif approach_type == 'A14':
        return ApproachA14(model_type, zero_mask=zero_mask,
                         grid_sample_mode=grid_sample_mode, **kwargs_without_base)
    elif approach_type == 'A15':
        return ApproachA15(model_type, zero_mask=zero_mask, **kwargs_without_base)
    else:
        raise ValueError(f"Unknown approach type: {approach_type}")


def create_combined_model(model_type='neural', approach_type='A1', **kwargs):
    """Factory function to create combined model + approach"""
    return create_approach(approach_type, model_type, **kwargs)





# 4. UNet++ model for fisheye transformation
class UNetPlusPlusModel(nn.Module):
    def __init__(self, encoder='mobilenet_v2', encoder_weights=None, zero_mask=None, dropout_prob: float = 0.0,
                 decoder_channels=None, encoder_depth=None):
        super(UNetPlusPlusModel, self).__init__()
        
        # Default decoder_channels based on encoder_depth if not provided
        if decoder_channels is None:
            if encoder_depth is not None and encoder_depth == 3:
                decoder_channels = [128, 64, 32]  # Default for depth 3
            elif encoder_depth is not None and encoder_depth == 4:
                decoder_channels = [128, 64, 32, 16]  # Default for depth 4
            else:
                decoder_channels = [256, 128, 64, 32, 16]  # Default for depth 5
        
        # Convert list/tuple to list if needed
        if isinstance(decoder_channels, (list, tuple)):
            decoder_channels = list(decoder_channels)
        elif isinstance(decoder_channels, str):
            # Parse string like "256,128,64,32,16"
            decoder_channels = [int(x.strip()) for x in decoder_channels.split(',')]
        
        # Default encoder_depth is 5 if not provided
        if encoder_depth is None:
            encoder_depth = 5
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=3,
            activation="sigmoid",
            decoder_channels=decoder_channels,
            encoder_depth=encoder_depth
        )
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob and dropout_prob > 0 else None
        # Register zero mask as buffer to preserve zero areas
        self.register_buffer('zero_mask', zero_mask)
    
    def forward(self, x):
        # UNet++ expects input in format [B, C, H, W]
        output = self.model(x)
        if self.dropout is not None and self.training:
            output = self.dropout(output)
        
        # Apply zero mask if available to preserve black borders
        if self.zero_mask is not None:
            batch_size, channels, height, width = output.shape
            # Expand mask to match output dimensions
            mask = self.zero_mask.contiguous().view(1, 1, height, width).expand(batch_size, channels, -1, -1).to(output.device)
            # Zero out areas that should be black
            output = output * (1 - mask)
        
        return output


# DRSFormer wrapper
# class DRSFormerModel(nn.Module):
#     """Wrapper around DRSformer backbone for AR Glasses image-to-image tasks."""
#     def __init__(self, in_channels=3, out_channels=3, zero_mask=None, **kwargs):
#         super(DRSFormerModel, self).__init__()
#         if DRSFormerBackbone is None:
#             raise ImportError("DRSFormer backbone not found. Please add the DRSformer repo to PYTHONPATH or install it.")
#         # DRSformer expects 3-channel input/outputs typically; enforce via conv adapters if needed
#         self.backbone = DRSFormerBackbone(
#                     inp_channels=in_channels, 
#                     out_channels=out_channels
#                     )
#         self.register_buffer('zero_mask', zero_mask)

#     def forward(self, x):
#         output = self.backbone(x)
#         if self.zero_mask is not None:
#             batch_size, channels, height, width = output.shape
#             mask = self.zero_mask.contiguous().view(1, 1, height, width).expand(batch_size, channels, -1, -1).to(output.device)
#             output = output * (1 - mask)
#         return output


# 5. U-Net model for AR Glasses project
class UNet(nn.Module):
    """
    U-Net architecture for image-to-image translation in AR Glasses project.
    Can be used for both NN1 (clean -> distorted) and NN2 (clean -> predistorted) training.
    """
    def __init__(self, in_channels=3, out_channels=3, base_features=64, dropout_prob: float = 0.0):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_layer(in_channels, base_features)
        self.enc2 = self._make_layer(base_features, base_features * 2)
        self.enc3 = self._make_layer(base_features * 2, base_features * 4)
        self.enc4 = self._make_layer(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = self._make_layer(base_features * 8, base_features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, 2)
        self.dec4 = self._make_layer(base_features * 16, base_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, 2)
        self.dec3 = self._make_layer(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, 2)
        self.dec2 = self._make_layer(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, 2)
        self.dec1 = self._make_layer(base_features * 2, base_features)
        
        # Optional dropout before output
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob and dropout_prob > 0 else None
        # Output layer
        self.out_conv = nn.Conv2d(base_features, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        if self.dropout is not None and self.training:
            dec1 = self.dropout(dec1)
        out = self.out_conv(dec1)
        out = self.sigmoid(out)
        
        return out


# 6. ResNet-based Generator for AR Glasses
class ResNetGenerator(nn.Module):
    """
    ResNet-based generator for AR Glasses project.
    Alternative architecture for NN1 and NN2 models.
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6, dropout_prob: float = 0.0):
        super(ResNetGenerator, self).__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        # ResNet blocks
        mult = 2 ** 2
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        # Optional dropout before output
        if dropout_prob and dropout_prob > 0:
            model += [nn.Dropout2d(p=dropout_prob)]

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Sigmoid()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    """ResNet block for the generator"""
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.BatchNorm2d(dim)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)


# 7. Lightweight MobileNet-based model for AR Glasses
class MobileNetGenerator(nn.Module):
    """
    Lightweight MobileNet-based generator for real-time AR applications.
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=32, dropout_prob: float = 0.0):
        super(MobileNetGenerator, self).__init__()
        
        # Initial layer
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 7, padding=3),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        # Encoder with depthwise separable convolutions
        self.enc1 = self._make_mobile_block(ngf, ngf * 2, stride=2)
        self.enc2 = self._make_mobile_block(ngf * 2, ngf * 4, stride=2)
        self.enc3 = self._make_mobile_block(ngf * 4, ngf * 8, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            self._make_mobile_block(ngf * 8, ngf * 8),
            self._make_mobile_block(ngf * 8, ngf * 8),
            self._make_mobile_block(ngf * 8, ngf * 8)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        # Optional dropout before output
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob and dropout_prob > 0 else None
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(ngf, output_nc, 7, padding=3),
            nn.Sigmoid()
        )
    
    def _make_mobile_block(self, in_channels, out_channels, stride=1):
        """Create a depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        if self.dropout is not None and self.training:
            x = self.dropout(x)
        x = self.output(x)
        return x


# =============================================================================
# INVERSE DISTORTION FUNCTIONS
# =============================================================================

def apply_inverse_color_distortion(image_tensor, distortion_type, learned_params):
    """
    Apply inverse color distortion using learned parameters.
    
    Args:
        image_tensor: [C, H, W] tensor
        distortion_type: Type of distortion to invert
        learned_params: Dictionary or tensor of learned parameters (strength/amplitude values)
    
    Returns:
        Corrected image tensor
    """
    # Extract learned strength from params (could be dict or single value)
    if isinstance(learned_params, dict):
        strength = learned_params.get('strength', learned_params.get('amp', 0.0))
    else:
        strength = learned_params
    
    # if distortion_type == 'lsnu':
    #     # Inverse LSNU: image * falloff -> image, so inverse: image / falloff
    #     # Original: falloff = 1.0 - amp * r2, image_out = image * falloff
    #     # Inverse: image_out / falloff = image_out / (1.0 - amp * r2)
    #     h, w = image_tensor.shape[-2:]
    #     y, x = torch.meshgrid(
    #         torch.linspace(-1, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
    #         torch.linspace(-1, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
    #         indexing='ij'
    #     )
    #     r2 = x * x + y * y
    #     amp = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 0.3)
    #     falloff = 1.0 - amp * r2
    #     falloff = torch.clamp(falloff, 0.7, 1.0)
    #     # Inverse: divide by falloff
    #     image_tensor = image_tensor / (falloff.unsqueeze(0) + 1e-8)
    #     image_tensor = torch.clamp(image_tensor, 0, 1)
        
    # elif distortion_type == 'blur':
    #     # Inverse blur: use unsharp masking / deconvolution approximation
    #     # We apply a sharpening kernel based on learned strength
    #     h, w = image_tensor.shape[-2:]
    #     sigma = 0.1 + 2.9 * torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
    #     # Create sharpening kernel (negative of blur, approximated)
    #     ksize = int((sigma * 3.0).item()) * 2 + 1
    #     ksize = max(3, min(ksize, 31))
    #     coords = torch.arange(ksize, device=image_tensor.device, dtype=image_tensor.dtype) - (ksize - 1) / 2
    #     gauss_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    #     gauss_1d = gauss_1d / (gauss_1d.sum() + 1e-8)
    #     # Sharpening: identity - blurred (approximation)
    #     identity_kernel = torch.zeros(ksize, device=image_tensor.device)
    #     identity_kernel[ksize // 2] = 1.0
    #     sharpen_kernel = (1.0 + strength * 2.0) * identity_kernel - strength * gauss_1d
    #     sharpen_kernel = sharpen_kernel / (sharpen_kernel.sum() + 1e-8)
    #     sharpen_x = sharpen_kernel.view(1, 1, 1, ksize)
    #     sharpen_y = sharpen_kernel.view(1, 1, ksize, 1)
    #     x = image_tensor.unsqueeze(0)
    #     C = x.shape[1]
    #     kernel_x = sharpen_x.expand(C, 1, 1, ksize)
    #     kernel_y = sharpen_y.expand(C, 1, ksize, 1)
    #     x = F.conv2d(x, kernel_x, groups=C, padding=(0, ksize//2))
    #     x = F.conv2d(x, kernel_y, groups=C, padding=(ksize//2, 0))
    #     image_tensor = x.squeeze(0)
    #     image_tensor = torch.clamp(image_tensor, 0, 1)
        
    # elif distortion_type == 'spatiotemporal_dithering':
    #     # Inverse dithering: simple denoising approximation
    #     # Apply Gaussian smoothing with learned denoising strength
    #     sigma = torch.clamp(torch.tensor(strength * 5.0, device=image_tensor.device, dtype=image_tensor.dtype), 0.1, 3.0)
    #     ksize = int((sigma * 3.0).item()) * 2 + 1
    #     ksize = max(3, min(ksize, 31))
    #     coords = torch.arange(ksize, device=image_tensor.device, dtype=image_tensor.dtype) - (ksize - 1) / 2
    #     gauss_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    #     gauss_1d = gauss_1d / (gauss_1d.sum() + 1e-8)
    #     gauss_x = gauss_1d.view(1, 1, 1, ksize)
    #     gauss_y = gauss_1d.view(1, 1, ksize, 1)
    #     x = image_tensor.unsqueeze(0)
    #     C = x.shape[1]
    #     kernel_x = gauss_x.expand(C, 1, 1, ksize)
    #     kernel_y = gauss_y.expand(C, 1, ksize, 1)
    #     x = F.conv2d(x, kernel_x, groups=C, padding=(0, ksize//2))
    #     x = F.conv2d(x, kernel_y, groups=C, padding=(ksize//2, 0))
    #     image_tensor = x.squeeze(0)
    #     image_tensor = torch.clamp(image_tensor, 0, 1)
        
    # elif distortion_type == 'elevated_black_level':
    #     # Inverse: original y = (x + p) / (1 + p), so x = y * (1 + p) - p
    #     p = 0.4 * torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 0.5) / 2.0
    #     image_tensor = image_tensor * (1.0 + p) - p
    #     image_tensor = torch.clamp(image_tensor, 0, 1)
        
    # elif distortion_type == 'color_fringes':
    #     # Inverse: reverse the radial scaling
    #     h, w = image_tensor.shape[-2:]
    #     yy, xx = torch.meshgrid(
    #         torch.linspace(-1, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
    #         torch.linspace(-1, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
    #         indexing='ij'
    #     )
    #     r2 = xx * xx + yy * yy
    #     base = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 1.0)
    #     s = 1.0 + 0.005 * base
    #     # Inverse scales (reciprocal)
    #     scales = torch.tensor([1.0/s, 1.0, 1.0/(2.0-s)], device=image_tensor.device, dtype=image_tensor.dtype)
    #     out_channels = []
    #     for c_idx in range(3):
    #         sx = sy = scales[c_idx]
    #         x_d = xx / sx  # Inverse scaling
    #         y_d = yy / sy
    #         grid = torch.stack([x_d, y_d], dim=-1).unsqueeze(0)
    #         ch = image_tensor[c_idx:c_idx+1, :, :].unsqueeze(0)
    #         warped = F.grid_sample(ch, grid, mode='bilinear', padding_mode='border', align_corners=True)
    #         out_channels.append(warped.squeeze(0))
    #     image_tensor = torch.cat(out_channels, dim=0)
    #     image_tensor = torch.clamp(image_tensor, 0, 1)
        
    # elif distortion_type == 'wgnu':
    #     # Inverse WGNU: divide by the gain pattern
    #     h, w = image_tensor.shape[-2:]
    #     yy, xx = torch.meshgrid(
    #         torch.linspace(0, 1, h, device=image_tensor.device, dtype=image_tensor.dtype),
    #         torch.linspace(0, 1, w, device=image_tensor.device, dtype=image_tensor.dtype),
    #         indexing='ij'
    #     )
    #     amp = torch.clamp(torch.tensor(strength, device=image_tensor.device, dtype=image_tensor.dtype), 0.0, 0.5)
    #     freq = 1.0 + 5.0 * amp
    #     phase = torch.zeros(1, device=image_tensor.device, dtype=image_tensor.dtype)  # Assume same phase
    #     stripes = torch.sin(2 * torch.pi * freq * xx + phase)
    #     stripes = stripes / (stripes.abs().amax() + 1e-8)
    #     gain = 1.0 + 0.05 * amp * stripes
    #     image_tensor = image_tensor / (gain.unsqueeze(0) + 1e-8)
    #     image_tensor = torch.clamp(image_tensor, 0, 1)
        
    # elif distortion_type == 'brightness_contrast':
    #     # Inverse: y = (x - 0.5) * c + 0.5, then y = y * b
    #     # Inverse: divide by b, then x = (y - 0.5) / c + 0.5
    #     brightness_factor = 1.0 + strength
    #     contrast_factor = 1.0 + strength
    #     # First undo brightness
    #     image_tensor = image_tensor / (brightness_factor + 1e-8)
    #     # Then undo contrast
    #     image_tensor = (image_tensor - 0.5) / (contrast_factor + 1e-8) + 0.5
    #     image_tensor = torch.clamp(image_tensor, 0, 1)
        
    # elif distortion_type == 'color_balance':
    #     # Inverse: divide by channel factors
    #     balance_factor = 1.0 + strength
    #     out_channels = []
    #     for c in range(image_tensor.shape[0]):
    #         channel_factor = balance_factor * (1.0 + (c - 1) * 0.1)
    #         out_channels.append(image_tensor[c] / (channel_factor + 1e-8))
    #     image_tensor = torch.stack(out_channels, dim=0)
    #     image_tensor = torch.clamp(image_tensor, 0, 1)
        
    return image_tensor


def apply_inverse_geometric_distortion(image_tensor, distortion_type, learned_params):
    """
    Apply inverse geometric distortion using learned parameters.
    
    Args:
        image_tensor: [C, H, W] tensor
        distortion_type: Type of distortion to invert
        learned_params: Dictionary or tensor of learned parameters (strength values)
    
    Returns:
        Corrected image tensor
    """
    # Extract learned strength from params
    if isinstance(learned_params, dict):
        strength = learned_params.get('strength', 0.0)
    else:
        strength = learned_params
    
    if distortion_type == 'fisheye':
        # Inverse fisheye: x_distorted = x / (1 - strength * r2)
        # Inverse: x = x_distorted * (1 - strength * r2) (approximate)
        h, w = image_tensor.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device),
            torch.linspace(-1, 1, w, device=image_tensor.device),
            indexing='ij'
        )
        r2 = x * x + y * y
        # Inverse transformation (approximate)
        inverse_factor = 1.0 - strength * r2
        inverse_factor = torch.clamp(inverse_factor, 0.1, 2.0)
        x_corrected = x * inverse_factor
        y_corrected = y * inverse_factor
        grid = torch.stack([x_corrected, y_corrected], dim=-1).unsqueeze(0)
        image_tensor = F.grid_sample(
            image_tensor.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0)

    elif distortion_type == 'keystone':
        # Inverse keystone: x_distorted = x + keystone_factor * y * x
        # Inverse: x = x_distorted / (1 + keystone_factor * y) (approximate)
        h, w = image_tensor.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image_tensor.device),
            torch.linspace(-1, 1, w, device=image_tensor.device),
            indexing='ij'
        )
        keystone_factor = torch.clamp(torch.tensor(strength, device=image_tensor.device), 0.0, 0.3)
        # Inverse transformation
        denom = 1.0 + keystone_factor * y
        x_corrected = x / (denom + 1e-8)
        y_corrected = y
        grid = torch.stack([x_corrected, y_corrected], dim=-1).unsqueeze(0)
        image_tensor = F.grid_sample(
            image_tensor.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0)
        
    return torch.clamp(image_tensor, 0, 1)


# =============================================================================
# INVERSE DISTORTION MODEL
# =============================================================================

class InverseDistortionModel(nn.Module):
    """
    Model that learns parameters for inverse distortions.
    For each distortion type, learns a single parameter (strength/amplitude)
    that is used to apply the mathematical inverse of the distortion.
    """
    def __init__(self, distortion_type, distortion_category='color', zero_mask=None, initial_strength=0.0):
        """
        Initialize inverse distortion model.
        
        Args:
            distortion_type: Type of distortion ('lsnu', 'blur', 'fisheye', etc.)
            distortion_category: 'color' or 'geo' (geometric)
            zero_mask: Optional zero mask for preserving zero areas
            initial_strength: Initial value for learned strength parameter
        """
        super(InverseDistortionModel, self).__init__()
        self.distortion_type = distortion_type
        self.distortion_category = distortion_category
        
        # Register zero mask as buffer
        self.register_buffer('zero_mask', zero_mask)
        
        # Learn a single parameter (strength/amplitude) for the inverse distortion
        # Initialize to a small value to start near identity
        self.learned_strength = nn.Parameter(torch.tensor(initial_strength, dtype=torch.float32))
        
        # Clip the parameter to reasonable ranges depending on distortion type
        self._init_parameter_bounds()
    
    def _init_parameter_bounds(self):
        """Initialize parameter bounds based on distortion type"""
        if self.distortion_type == 'lsnu':
            self.min_val, self.max_val = 0.0, 0.3
        elif self.distortion_type == 'blur':
            self.min_val, self.max_val = 0.0, 1.0
        elif self.distortion_type == 'spatiotemporal_dithering':
            self.min_val, self.max_val = 0.0, 0.1
        elif self.distortion_type == 'elevated_black_level':
            self.min_val, self.max_val = 0.0, 0.5
        elif self.distortion_type == 'color_fringes':
            self.min_val, self.max_val = 0.0, 1.0
        elif self.distortion_type == 'wgnu':
            self.min_val, self.max_val = 0.0, 0.5
        elif self.distortion_type == 'brightness_contrast':
            self.min_val, self.max_val = 0.0, 1.0
        elif self.distortion_type == 'color_balance':
            self.min_val, self.max_val = 0.0, 1.0
        elif self.distortion_type == 'fisheye':
            self.min_val, self.max_val = 0.0, 1.0
        elif self.distortion_type == 'keystone':
            self.min_val, self.max_val = 0.0, 0.3
        else:
            self.min_val, self.max_val = 0.0, 1.0
    
    def forward(self, x):
        """
        Apply inverse distortion to input image.
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            Corrected image tensor [B, C, H, W]
        """
        # Clamp learned parameter to valid range
        clamped_strength = torch.clamp(self.learned_strength, self.min_val, self.max_val)
        
        # Apply inverse distortion based on category
        if self.distortion_category == 'color':
            # Process each image in batch
            corrected_images = []
            for i in range(x.shape[0]):
                img = x[i]  # [C, H, W]
                corrected = apply_inverse_color_distortion(img, self.distortion_type, clamped_strength)
                corrected_images.append(corrected)
            output = torch.stack(corrected_images, dim=0)
        else:  # geometric
            corrected_images = []
            for i in range(x.shape[0]):
                img = x[i]  # [C, H, W]
                corrected = apply_inverse_geometric_distortion(img, self.distortion_type, clamped_strength)
                corrected_images.append(corrected)
            output = torch.stack(corrected_images, dim=0)
        
        # Apply zero mask if provided
        if self.zero_mask is not None:
            batch_size, channels, height, width = output.shape
            expanded_mask = self.zero_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, -1, -1)
            output = output * (1 - expanded_mask)
        
        return torch.clamp(output, 0, 1)
    
    def get_learned_strength(self):
        """Get the current learned strength parameter value"""
        return torch.clamp(self.learned_strength, self.min_val, self.max_val).item()


# Individual inverse distortion model classes for each distortion type
class InverseLSNUModel(InverseDistortionModel):
    """Inverse LSNU (Light Source Non-Uniformity) distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseLSNUModel, self).__init__('lsnu', 'color', zero_mask, initial_strength)

class InverseBlurModel(InverseDistortionModel):
    """Inverse blur distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseBlurModel, self).__init__('blur', 'color', zero_mask, initial_strength)

class InverseSpatiotemporalDitheringModel(InverseDistortionModel):
    """Inverse spatiotemporal dithering distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseSpatiotemporalDitheringModel, self).__init__('spatiotemporal_dithering', 'color', zero_mask, initial_strength)

class InverseElevatedBlackLevelModel(InverseDistortionModel):
    """Inverse elevated black level distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseElevatedBlackLevelModel, self).__init__('elevated_black_level', 'color', zero_mask, initial_strength)

class InverseColorFringesModel(InverseDistortionModel):
    """Inverse color fringes distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseColorFringesModel, self).__init__('color_fringes', 'color', zero_mask, initial_strength)

class InverseWGNUModel(InverseDistortionModel):
    """Inverse WGNU (Waveguide Non-Uniformity) distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseWGNUModel, self).__init__('wgnu', 'color', zero_mask, initial_strength)

class InverseBrightnessContrastModel(InverseDistortionModel):
    """Inverse brightness/contrast distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseBrightnessContrastModel, self).__init__('brightness_contrast', 'color', zero_mask, initial_strength)

class InverseColorBalanceModel(InverseDistortionModel):
    """Inverse color balance distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseColorBalanceModel, self).__init__('color_balance', 'color', zero_mask, initial_strength)

class InverseFisheyeModel(InverseDistortionModel):
    """Inverse fisheye geometric distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseFisheyeModel, self).__init__('fisheye', 'geo', zero_mask, initial_strength)

class InverseKeystoneModel(InverseDistortionModel):
    """Inverse keystone geometric distortion model"""
    def __init__(self, zero_mask=None, initial_strength=0.0):
        super(InverseKeystoneModel, self).__init__('keystone', 'geo', zero_mask, initial_strength)


# 7.5. CompenNetPlusplus Wrapper
class CompenNetPlusplusWrapper(nn.Module):
    """
    Wrapper for CompenNetPlusplus to make it compatible with single-input interface.
    Uses a complete black surface (all zeros) as the surface image.
    """
    def __init__(self, **kwargs):
        super(CompenNetPlusplusWrapper, self).__init__()
        # Get out_size from kwargs, default to (256, 256) for backward compatibility
        # If image_size is provided, use it for both dimensions
        if 'out_size' in kwargs:
            out_size = kwargs['out_size']
        elif 'image_size' in kwargs:
            out_size = (kwargs['image_size'], kwargs['image_size'])
        else:
            out_size = (256, 256)  # Default
        
        # CompenNetPlusplus now accepts out_size and forwards it to WarpingNet.
        # Default (256, 256) preserves backward-compatible behavior.
        self.model = CompenNetPlusplus(out_size=out_size)
        
        # Store zero_mask if provided (though CompenNetPlusplus doesn't use it directly)
        self.zero_mask = kwargs.get('zero_mask', None)
    
    def forward(self, x):
        """
        Forward pass: uses a complete black surface as the surface image.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Clamp input to [0, 1] range as expected by CompenNetPlusplus
        x = torch.clamp(x, 0.0, 1.0)
        
        # Use complete black surface (all zeros) as surface image
        batch_size, channels, height, width = x.shape
        s = torch.zeros_like(x)  # Complete black surface
        # s = torch.ones_like(x)
        
        # Forward through CompenNetPlusplus
        output = self.model(x, s)
        
        # Apply zero mask if available
        if self.zero_mask is not None:
            output = output * self.zero_mask
        
        return output


# 7.5b. NAFNet Wrapper
class NAFNetWrapper(nn.Module):
    """
    Wrapper around vendored NAFNet (megvii-research, ECCV 2022) so it plugs into
    the same single-input (B,3,H,W) -> (B,3,H,W) interface as our other NN2 models.

    Accepts either preset='small'|'baseline'|'deep32'|'baseline64' (default
    'baseline' = NAFNet-wide-32, ~29 M params), or an explicit (width, enc, mid,
    dec) tuple via kwargs. Clamps input to [0,1] (matches CompenNetPlusplusWrapper);
    output is NOT clamped (the paper doesn't clamp, and forensic audit showed
    a [0,1] output clamp zeros gradient on ~24% of pixels at random init, which
    crippled #154-#158).
    """
    def __init__(self, **kwargs):
        super().__init__()
        preset = kwargs.get('preset', 'baseline')
        if preset is not None:
            self.model = build_nafnet_from_preset(preset, img_channel=3)
            self.preset = preset
        else:
            self.model = NAFNet(
                img_channel=3,
                width=kwargs.get('width', 32),
                middle_blk_num=kwargs.get('middle_blk_num', 12),
                enc_blk_nums=kwargs.get('enc_blk_nums', [2, 2, 4, 8]),
                dec_blk_nums=kwargs.get('dec_blk_nums', [2, 2, 2, 2]),
            )
            self.preset = 'custom'
        self.zero_mask = kwargs.get('zero_mask', None)

    def forward(self, x):
        x = torch.clamp(x, 0.0, 1.0)
        out = self.model(x)
        if self.zero_mask is not None:
            out = out * self.zero_mask
        return out


# 7.6. Sequential Model (Chains multiple models)
class SequentialModel(nn.Module):
    """
    A model that chains multiple models sequentially.
    Input flows through each model in order: input -> model1 -> model2 -> ... -> output
    
    Examples:
        # Example 1: Two models (a1 -> a4)
        model_configs = [
            {'model_type': 'parametric_a1', 'model_complexity': 'complex', 'hidden_dim': 64},
            {'model_type': 'parametric_a4', 'model_complexity': 'complex', 'hidden_dim': 64}
        ]
        model = create_model('sequential', model_configs=model_configs)
        
        # Example 2: Three models with different types
        model_configs = [
            {'model_type': 'parametric_a1', 'model_complexity': 'complex', 'hidden_dim': 64},
            {'model_type': 'parametric_a4', 'model_complexity': 'complex', 'hidden_dim': 64},
            {'model_type': 'unetpp', 'decoder_channels': '64,32,16,8,4'}
        ]
        model = create_model('sequential', model_configs=model_configs)
        
        # Example 3: With patch_size for a11
        model_configs = [
            {'model_type': 'parametric_a1', 'model_complexity': 'complex', 'hidden_dim': 64},
            {'model_type': 'parametric_a11', 'model_complexity': 'complex', 'hidden_dim': 64, 'patch_size': 3}
        ]
        model = create_model('sequential', model_configs=model_configs)
    """
    def __init__(self, model_configs, zero_mask=None, **kwargs):
        """
        Args:
            model_configs: List of dictionaries, each specifying a model configuration.
                          Each dict should contain:
                          - model_type: e.g., 'parametric_a1', 'parametric_a4', 'unetpp', etc.
                          - model_complexity: e.g., 'simple', 'medium', 'complex', 'complexer' (for approach models)
                          - hidden_dim: Hidden dimension (for approach models)
                          - patch_size: Patch size (for a9/a11 models, optional)
                          - decoder_channels: Decoder channels (for unetpp, optional)
                          - Any other model-specific parameters
            zero_mask: Optional zero mask to apply to final output
            **kwargs: Additional arguments (currently unused, for future compatibility)
        """
        super(SequentialModel, self).__init__()
        
        if not isinstance(model_configs, list) or len(model_configs) == 0:
            raise ValueError("model_configs must be a non-empty list of model configurations")
        
        self.model_configs = model_configs
        self.num_models = len(model_configs)
        self.zero_mask = zero_mask
        
        # Create each model in the sequence
        self.models = nn.ModuleList()
        for i, config in enumerate(model_configs):
            if not isinstance(config, dict):
                raise ValueError(f"Model config {i} must be a dictionary, got {type(config)}")
            
            if 'model_type' not in config:
                raise ValueError(f"Model config {i} must have 'model_type' key")
            
            # Extract model_type and create kwargs for this model
            model_type = config['model_type']
            model_kwargs = {k: v for k, v in config.items() if k != 'model_type'}
            
            # Pass zero_mask only to the first model (or handle it at the end)
            # Actually, we'll handle zero_mask only at the final output
            # But if a model needs zero_mask internally, we can pass it
            if i == 0 and zero_mask is not None:
                # Only first model gets zero_mask if needed
                # Most models don't need it, so we'll skip for now
                pass
            
            # Create the model
            try:
                model = create_model(model_type, **model_kwargs)
                self.models.append(model)
            except Exception as e:
                raise RuntimeError(f"Failed to create model {i} (type: {model_type}): {str(e)}")
        
        # Store model info for debugging
        self.model_types = [config['model_type'] for config in model_configs]
    
    def forward(self, x):
        """
        Forward pass through all models sequentially.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Pass input through each model sequentially
        output = x
        for i, model in enumerate(self.models):
            output = model(output)

        # Apply zero mask to final output if available
        if self.zero_mask is not None:
            output = output * self.zero_mask

        return output
    
    def __repr__(self):
        """String representation of the sequential model"""
        model_str = " -> ".join(self.model_types)
        return f"SequentialModel({self.num_models} models: {model_str})"


# 8. Model factory function
def create_model(model_type='unet', **kwargs):
    """
    Factory function to create different model types for AR Glasses project.
    
    Args:
        model_type: Type of model ('unet', 'resnet', 'mobilenet', 'unetpp', 'neural', 'neuralp')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        PyTorch model instance
    """
    if model_type == 'unet':
        return UNet(**kwargs)
    elif model_type == 'resnet':
        return ResNetGenerator(**kwargs)
    elif model_type == 'mobilenet':
        return MobileNetGenerator(**kwargs)
    elif model_type == 'unetpp':
        return UNetPlusPlusModel(**kwargs)
    # elif model_type == 'drsformer':
        # return DRSFormerModel(**kwargs)
    elif model_type == 'neural_a1':
        return create_combined_model('neural', 'A1', **kwargs)
    elif model_type == 'neural_a2':
        return create_combined_model('neural', 'A2', **kwargs)
    elif model_type == 'neural_a3':
        return create_combined_model('neural', 'A3', **kwargs)
    elif model_type == 'neural_a4':
        return create_combined_model('neural', 'A4', **kwargs)
    elif model_type == 'parametric_a1':
        return create_combined_model('parametric', 'A1', **kwargs)
    elif model_type == 'parametric_a2':
        return create_combined_model('parametric', 'A2', **kwargs)
    elif model_type == 'parametric_a3':
        return create_combined_model('parametric', 'A3', **kwargs)
    elif model_type == 'parametric_a4':
        return create_combined_model('parametric', 'A4', **kwargs)
    elif model_type == 'neural_a5':
        return create_combined_model('neural', 'A5', **kwargs)
    elif model_type == 'neural_a6':
        return create_combined_model('neural', 'A6', **kwargs)
    elif model_type == 'neural_a7':
        return create_combined_model('neural', 'A7', **kwargs)
    elif model_type == 'neural_a8':
        return create_combined_model('neural', 'A8', **kwargs)
    elif model_type == 'neural_a9':
        return create_combined_model('neural', 'A9', **kwargs)
    elif model_type == 'neural_a10':
        return create_combined_model('neural', 'A10', **kwargs)
    elif model_type == 'neural_a11':
        return create_combined_model('neural', 'A11', **kwargs)
    elif model_type == 'neural_a12':
        return create_combined_model('neural', 'A12', **kwargs)
    elif model_type == 'neural_a13':
        return create_combined_model('neural', 'A13', **kwargs)
    elif model_type == 'neural_a14':
        return create_combined_model('neural', 'A14', **kwargs)
    elif model_type == 'neural_a15':
        return create_combined_model('neural', 'A15', **kwargs)
    elif model_type == 'parametric_a5':
        return create_combined_model('parametric', 'A5', **kwargs)
    elif model_type == 'parametric_a6':
        return create_combined_model('parametric', 'A6', **kwargs)
    elif model_type == 'parametric_a7':
        return create_combined_model('parametric', 'A7', **kwargs)
    elif model_type == 'parametric_a8':
        return create_combined_model('parametric', 'A8', **kwargs)
    elif model_type == 'parametric_a9':
        return create_combined_model('parametric', 'A9', **kwargs)
    elif model_type == 'parametric_a10':
        return create_combined_model('parametric', 'A10', **kwargs)
    elif model_type == 'parametric_a11':
        return create_combined_model('parametric', 'A11', **kwargs)
    elif model_type == 'parametric_a12':
        return create_combined_model('parametric', 'A12', **kwargs)
    elif model_type == 'parametric_a13':
        return create_combined_model('parametric', 'A13', **kwargs)
    elif model_type == 'parametric_a14':
        return create_combined_model('parametric', 'A14', **kwargs)
    elif model_type == 'parametric_a15':
        return create_combined_model('parametric', 'A15', **kwargs)
    elif model_type == 'kan_a1':
        return create_combined_model('kan', 'A1', **kwargs)
    elif model_type == 'kan_a2':
        return create_combined_model('kan', 'A2', **kwargs)
    elif model_type == 'kan_a3':
        return create_combined_model('kan', 'A3', **kwargs)
    elif model_type == 'kan_a4':
        return create_combined_model('kan', 'A4', **kwargs)
    elif model_type == 'kan_a5':
        return create_combined_model('kan', 'A5', **kwargs)
    elif model_type == 'kan_a6':
        return create_combined_model('kan', 'A6', **kwargs)
    elif model_type == 'kan_a7':
        return create_combined_model('kan', 'A7', **kwargs)
    elif model_type == 'kan_a8':
        return create_combined_model('kan', 'A8', **kwargs)
    elif model_type == 'kan_a9':
        return create_combined_model('kan', 'A9', **kwargs)
    elif model_type == 'kan_a10':
        return create_combined_model('kan', 'A10', **kwargs)
    elif model_type == 'kan_a11':
        return create_combined_model('kan', 'A11', **kwargs)
    elif model_type == 'kan_a12':
        return create_combined_model('kan', 'A12', **kwargs)
    elif model_type == 'kan_a13':
        return create_combined_model('kan', 'A13', **kwargs)
    elif model_type == 'kan_a14':
        return create_combined_model('kan', 'A14', **kwargs)
    # Inverse distortion models (mathematical inverse of distortions)
    elif model_type == 'inverse_lsnu':
        return InverseLSNUModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'inverse_blur':
        return InverseBlurModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'inverse_spatiotemporal_dithering':
        return InverseSpatiotemporalDitheringModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'inverse_elevated_black_level':
        return InverseElevatedBlackLevelModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'inverse_color_fringes':
        return InverseColorFringesModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'inverse_wgnu':
        return InverseWGNUModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'inverse_brightness_contrast':
        return InverseBrightnessContrastModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'inverse_color_balance':
        return InverseColorBalanceModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'inverse_fisheye':
        return InverseFisheyeModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'inverse_keystone':
        return InverseKeystoneModel(zero_mask=kwargs.get('zero_mask', None), initial_strength=kwargs.get('initial_strength', 0.0))
    elif model_type == 'compennetplusplus':
        if not COMPENNETPLUSPLUS_AVAILABLE:
            raise ImportError("CompenNetPlusplus is not available. Make sure compennetplusplus.py and pytorch_tps.py are in the project directory.")
        # Create a wrapper to make CompenNetPlusplus compatible with single-input interface
        return CompenNetPlusplusWrapper(**kwargs)
    elif model_type == 'nafnet':
        if not NAFNET_AVAILABLE:
            raise ImportError("NAFNet is not available. Make sure nafnet_arch.py is in the project directory.")
        return NAFNetWrapper(**kwargs)
    elif model_type == 'sequential':
        # Sequential model that chains multiple models
        if 'model_configs' not in kwargs:
            raise ValueError("Sequential model requires 'model_configs' parameter (list of model configurations)")
        return SequentialModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 9. Model utilities
def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model, model_name="Model"):
    """Print model information including parameter count and architecture"""
    total_params = count_parameters(model)
    print(f"\n{model_name} Information:")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    print(f"Architecture: {model.__class__.__name__}")
    
    # Print model structure
    print(f"\nModel structure:")
    print(model)


def test_gradient_flow():
    """Test that gradients flow correctly through all models, approaches, and complexities"""
    print("🔍 Testing Gradient Flow Through All Models/Approaches/Complexities")
    print("=" * 70)

    # Test configurations
    model_types = ['neural', 'parametric']
    complexities = ['simple', 'medium', 'complex']
    approaches = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']

    # Add fisheye only for parametric models
    parametric_complexities = complexities + ['fisheye']

    # Test parameters
    batch_size = 2
    height, width = 32, 32
    channels = 3
    hidden_dim = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    all_passed = True

    # Test each model type and complexity
    for model_type in model_types:
        print(f"\n📊 Testing {model_type.upper()} Models:")
        print("-" * 40)

        # Get appropriate complexities for this model type
        test_complexities = parametric_complexities if model_type == 'parametric' else complexities

        for complexity in test_complexities:
            print(f"  Testing {model_type} {complexity}...")

            try:
                # Create model
                if model_type == 'neural':
                    model = NeuralModel(2, 2, hidden_dim, complexity)
                else:  # parametric
                    model = ParametricModel(2, 2, complexity)

                model = model.to(device)
                model.train()

                # Create test input
                x = torch.randn(batch_size, height, width, 2, device=device, requires_grad=False)
                x.requires_grad_(True)

                # Forward pass
                output = model(x)

                # Create loss (sum of output to ensure gradients flow)
                loss = output.sum()
                loss.backward()

                # Check gradients
                has_gradients = True
                zero_gradients = 0
                total_params = 0

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        total_params += 1
                        if param.grad is None:
                            has_gradients = False
                            print(f"    ❌ {name}: No gradient")
                        elif param.grad.abs().sum().item() == 0:
                            zero_gradients += 1
                            print(f"    ⚠️  {name}: Zero gradient")

                if has_gradients and zero_gradients == 0:
                    print(f"    ✅ {model_type} {complexity}: All {total_params} parameters have gradients")
                elif has_gradients:
                    print(f"    ⚠️  {model_type} {complexity}: {total_params - zero_gradients}/{total_params} parameters have non-zero gradients")
                    all_passed = False
                else:
                    print(f"    ❌ {model_type} {complexity}: Gradient flow broken")
                    all_passed = False

            except Exception as e:
                print(f"    ❌ {model_type} {complexity}: Error - {str(e)}")
                all_passed = False

    # Test approaches
    print(f"\n🔄 Testing Approaches:")
    print("-" * 40)

    for approach_name in approaches:
        print(f"  Testing Approach {approach_name.upper()}...")

        for model_type in model_types:
            for complexity in (parametric_complexities if model_type == 'parametric' else complexities):
                # Skip combinations that are expected to fail (fisheye with color approaches)
                if (complexity == 'fisheye' and approach_name in ['a2', 'a3', 'a4'] and model_type == 'parametric'):
                    print(f"    ⭕ {approach_name.upper()}-{model_type}-{complexity}: Skipped (fisheye requires 2D I/O)")
                    continue

                try:
                    # Create approach
                    if approach_name == 'a1':
                        approach = ApproachA1(model_type, complexity, hidden_dim)
                    elif approach_name == 'a2':
                        approach = ApproachA2(model_type, complexity, hidden_dim)
                    elif approach_name == 'a3':
                        approach = ApproachA3(model_type, complexity, hidden_dim)
                    elif approach_name == 'a4':
                        approach = ApproachA4(model_type, complexity, hidden_dim)
                    elif approach_name == 'a5':
                        approach = ApproachA5(model_type, complexity, model_type, complexity, hidden_dim)
                    elif approach_name == 'a6':
                        approach = ApproachA6(model_type, complexity, model_type, complexity, hidden_dim)
                    elif approach_name == 'a7':
                        approach = ApproachA7(model_type, complexity, hidden_dim)
                    elif approach_name == 'a8':
                        approach = ApproachA8(model_type, complexity, model_type, complexity, hidden_dim)
                    else:
                        raise ValueError(f"Unknown approach: {approach_name}")

                    approach = approach.to(device)
                    approach.train()

                    # Create test input (RGB image)
                    x = torch.randn(batch_size, channels, height, width, device=device, requires_grad=True)

                    # Forward pass
                    output = approach(x)

                    # Create loss - handle different output formats
                    if approach_name in ['a5', 'a6'] and isinstance(output, dict):
                        # For approaches 5 and 6, sum all outputs
                        loss = output['im_prime'].sum() + output['im_double_prime'].sum()
                    else:
                        loss = output.sum()
                    loss.backward()

                    # Check gradients
                    has_gradients = True
                    zero_gradients = 0
                    total_params = 0

                    for name, param in approach.named_parameters():
                        if param.requires_grad:
                            total_params += 1
                            if param.grad is None:
                                has_gradients = False
                            elif param.grad.abs().sum().item() == 0:
                                zero_gradients += 1

                    if has_gradients and zero_gradients == 0:
                        status = "✅"
                    elif has_gradients:
                        status = "⚠️"
                        all_passed = False
                    else:
                        status = "❌"
                        all_passed = False

                    print(f"    {status} {approach_name.upper()}-{model_type}-{complexity}: {total_params - zero_gradients}/{total_params} parameters")

                except Exception as e:
                    print(f"    ❌ {approach_name.upper()}-{model_type}-{complexity}: Error - {str(e)}")
                    all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED: Gradients flow correctly through all models!")
    else:
        print("⚠️  SOME TESTS FAILED: Check gradient flow issues above")
    print("=" * 70)

    return all_passed


# =============================================================================
# LOSS FUNCTIONS FOR TWO-STAGE TRAINING
# =============================================================================

def geometry_loss(pred, gt, flow=None, alpha=(1.0, 1.0), return_breakdown=False):
    """
    Geometry correction loss for net1.
    
    Combines pixel accuracy and structural similarity.

    Args:
        pred (Tensor): Predicted geometrically corrected image (B, C, H, W)
        gt (Tensor): Ground-truth image (B, C, H, W)
        flow (Tensor, optional): Unused, kept for compatibility
        alpha (tuple): Weights for (L1, SSIM)
        return_breakdown (bool): If True, return breakdown of individual losses
    Returns:
        torch.Tensor or tuple: Combined geometry loss value, or (total, breakdown_dict) if return_breakdown=True
    """
    # 1. L1 reconstruction
    l1 = F.l1_loss(pred, gt)

    # 2. Structural similarity
    ssim_loss = 1 - ssim(pred, gt, data_range=1.0, size_average=True)

    # Combined loss (only L1 and SSIM)
    total = alpha[0]*l1 + alpha[1]*ssim_loss
    
    if return_breakdown:
        breakdown = {
            'l1': l1.item(),
            'ssim': ssim_loss.item(),
            'total': total.item()
        }
        return total, breakdown
    
    return total


def color_loss(pred, gt, beta=(1.0, 1.0, 0.2, 0.5), return_breakdown=False):
    """
    Color correction loss for net2.

    Combines perceptual, Lab-space, histogram, and luminance consistency losses.
    
    Args:
        pred (Tensor): Predicted color-corrected image (B, C, H, W)
        gt (Tensor): Ground-truth target image (B, C, H, W)
        beta (tuple): Weights for (Lab, Perceptual, Histogram, Luminance)
        return_breakdown (bool): If True, return breakdown of individual losses
    Returns:
        torch.Tensor or tuple: Combined color loss value, or (total, breakdown_dict) if return_breakdown=True
    """
    # 1. L1 loss in Lab color space
    lab_pred = rgb_to_lab(pred)
    lab_gt = rgb_to_lab(gt)
    lab_loss = F.l1_loss(lab_pred, lab_gt)

    # 2. Perceptual (LPIPS) loss
    lpips_fn = lpips.LPIPS(net='vgg').to(pred.device)
    perc_loss = lpips_fn(pred, gt).mean()

    # 3. Histogram (color distribution) loss
    def color_hist_loss(img1, img2, bins=64):
        hist_loss = 0
        for c in range(3):
            h1 = torch.histc(img1[:, c], bins=bins, min=0.0, max=1.0)
            h2 = torch.histc(img2[:, c], bins=bins, min=0.0, max=1.0)
            h1, h2 = h1 / h1.sum(), h2 / h2.sum()
            hist_loss += torch.abs(h1 - h2).sum()
        return hist_loss / 3.0
    hist_loss = color_hist_loss(pred, gt)

    # 4. Luminance consistency
    lum_loss = F.l1_loss(lab_pred[:, [0], :, :], lab_gt[:, [0], :, :])

    total = beta[0]*lab_loss + beta[1]*perc_loss + beta[2]*hist_loss + beta[3]*lum_loss
    
    if return_breakdown:
        breakdown = {
            'lab': lab_loss.item(),
            'perceptual': perc_loss.item(),
            'histogram': hist_loss.item(),
            'luminance': lum_loss.item(),
            'total': total.item()
        }
        return total, breakdown
    
    return total


def combined_geometry_color_loss(pred, gt, flow=None, alpha=(1.0, 1.0, 0.5, 0.01), beta=(1.0, 1.0, 0.2, 0.5), gamma=0.5):
    """
    Combined loss function that combines geometry and color losses.
    
    Args:
        pred (Tensor): Predicted image (B, C, H, W)
        gt (Tensor): Ground-truth image (B, C, H, W)
        flow (Tensor, optional): Predicted flow or sampling grid (B, 2, H, W)
        alpha (tuple): Weights for geometry loss components
        beta (tuple): Weights for color loss components
        gamma (float): Weight for combining geometry and color losses
    Returns:
        torch.Tensor: Combined loss value
    """
    geom_loss = geometry_loss(pred, gt, flow, alpha)
    col_loss = color_loss(pred, gt, beta)
    return gamma * geom_loss + (1 - gamma) * col_loss


if __name__ == "__main__":
    test_gradient_flow()