#!/usr/bin/env python3
"""quant_core.py — pure quantization math lifted from Javad's quantization.py.
No framework/dataset/torchmetrics deps. Just numpy + torch."""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple

DEFAULT_ACCUMULATOR_BITS = 64
DEFAULT_USE_SATURATION = True
DEFAULT_ROUNDING_MODE = 'nearest_even'
DEFAULT_REQUANTIZE_OUTPUTS = True
DEFAULT_OUTPUT_BITS = 8

@dataclass
class HardwareConfig:
    accumulator_bits: int = DEFAULT_ACCUMULATOR_BITS
    use_saturation: bool = DEFAULT_USE_SATURATION
    rounding_mode: str = DEFAULT_ROUNDING_MODE
    requantize_outputs: bool = DEFAULT_REQUANTIZE_OUTPUTS
    output_bits: int = DEFAULT_OUTPUT_BITS
    calibration_fraction: float = 0.01
    full_eval_after_calibration: bool = True

DEFAULT_HARDWARE_CONFIG = HardwareConfig()


def mask_to_bits(values: np.ndarray, num_bits: int, is_signed: bool) -> np.ndarray:
    if is_signed:
        mask = (1 << num_bits) - 1
        masked = values & mask
        sign_bit = 1 << (num_bits - 1)
        masked = np.where(masked >= sign_bit, masked - (1 << num_bits), masked)
    else:
        mask = (1 << num_bits) - 1
        masked = values & mask
    return masked.astype(np.int64)


@dataclass
class QuantizedTensor:
    values: np.ndarray
    num_bits: int
    is_signed: bool
    scale: np.ndarray
    zero_point: np.ndarray
    shape: tuple
    per_channel: bool = False
    channel_dim: int = 0

    def __post_init__(self):
        if self.is_signed:
            self.min_val = -(2 ** (self.num_bits - 1))
            self.max_val = 2 ** (self.num_bits - 1) - 1
        else:
            self.min_val = 0
            self.max_val = 2 ** self.num_bits - 1
        self.values = np.clip(self.values, self.min_val, self.max_val).astype(np.int64)
        if self.num_bits < 63:
            self.values = mask_to_bits(self.values, self.num_bits, self.is_signed)
        if not isinstance(self.scale, np.ndarray):
            self.scale = np.array([self.scale], dtype=np.float64)
        elif self.scale.dtype != np.float64:
            self.scale = self.scale.astype(np.float64)
        if not isinstance(self.zero_point, np.ndarray):
            self.zero_point = np.array([self.zero_point], dtype=np.int64)
        elif self.zero_point.dtype != np.int64:
            self.zero_point = self.zero_point.astype(np.int64)


class QuantizationConfig:
    def __init__(self, num_bits=8, symmetric=True, per_channel=False, quantize_bias=False):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.quantize_bias = quantize_bias
        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1


def uniform_quantize(tensor, config, per_channel=False, channel_dim=0):
    # per-tensor path only (per_channel=False in your config)
    tensor_np = tensor.detach().cpu().numpy().astype(np.float64)
    original_shape = tensor.shape
    min_val = np.float64(tensor_np.min())
    max_val = np.float64(tensor_np.max())
    if config.symmetric:
        abs_max = np.maximum(np.abs(min_val), np.abs(max_val))
        qmax_float = np.float64(2 ** (config.num_bits - 1) - 1)
        scale = abs_max / qmax_float if abs_max > 1e-10 else np.float64(1e-10)
        zero_point_val = np.int64(0)
    else:
        qrange = np.float64(config.qmax - config.qmin)
        value_range = max_val - min_val
        scale = value_range / qrange if value_range > 1e-10 else np.float64(1e-10)
        zero_point_val = np.int64(np.round(config.qmin - min_val / scale))
    q_float = tensor_np / scale + np.float64(zero_point_val)
    q_rounded = np.round(q_float)
    q_int = np.clip(q_rounded, config.qmin, config.qmax).astype(np.int64)
    q_int = mask_to_bits(q_int, config.num_bits, config.symmetric)
    qt = QuantizedTensor(
        values=q_int, num_bits=config.num_bits, is_signed=config.symmetric,
        scale=np.array([scale], dtype=np.float64),
        zero_point=np.array([zero_point_val], dtype=np.int64),
        shape=original_shape, per_channel=False, channel_dim=0)
    return qt, torch.tensor(float(scale)), torch.tensor(int(zero_point_val))


def dequantize(qt, device='cpu'):
    zp = np.float64(qt.zero_point[0]); sc = np.float64(qt.scale[0])
    q_float = qt.values.astype(np.float64)
    dq = (q_float - zp) * sc
    return torch.from_numpy(dq.astype(np.float32)).reshape(qt.shape).to(device)


def round_hardware(values, mode='nearest_even'):
    if mode == 'nearest_even':
        return np.rint(values).astype(np.int64)
    elif mode == 'nearest_away':
        return np.round(values).astype(np.int64)
    elif mode == 'truncate':
        return np.trunc(values).astype(np.int64)
    else:
        raise ValueError(f"mode {mode}")


def apply_fixed_accumulator(values, hw_config=None):
    if hw_config is None:
        hw_config = DEFAULT_HARDWARE_CONFIG
    acc_bits = hw_config.accumulator_bits
    acc_min = -(2 ** (acc_bits - 1)); acc_max = 2 ** (acc_bits - 1) - 1
    if hw_config.use_saturation:
        return np.clip(values, acc_min, acc_max).astype(np.int64)
    else:
        return mask_to_bits(values, acc_bits, is_signed=True)


def quantized_matmul_custom(x, w, hw_config=None):
    if hw_config is None:
        hw_config = DEFAULT_HARDWARE_CONFIG
    x_2d = x.values.reshape(-1, x.shape[-1])
    w_2d = w.values.reshape(w.shape[0], -1)
    result_values = np.matmul(x_2d, w_2d.T, dtype=np.int64)
    result_values = apply_fixed_accumulator(result_values, hw_config)
    output_bits = hw_config.accumulator_bits
    output_scale = np.array([np.float64(x.scale[0]) * np.float64(w.scale[0])], dtype=np.float64)
    output_zero_point = np.array([0], dtype=np.int64)
    if len(x.shape) > 2:
        output_shape = x.shape[:-1] + (w.shape[0],)
        result_values = result_values.reshape(output_shape)
    else:
        output_shape = (x.shape[0], w.shape[0])
    return QuantizedTensor(
        values=result_values, num_bits=output_bits, is_signed=True,
        scale=output_scale, zero_point=output_zero_point,
        shape=output_shape, per_channel=False, channel_dim=len(output_shape)-1)


def requantize_direct(qt, target_bits, hw_config=None):
    if hw_config is None:
        hw_config = DEFAULT_HARDWARE_CONFIG
    if qt.num_bits <= target_bits:
        return QuantizedTensor(qt.values.copy(), target_bits, qt.is_signed,
            qt.scale.copy(), qt.zero_point.copy(), qt.shape, qt.per_channel, qt.channel_dim)
    zp = np.float64(qt.zero_point[0]); sc = np.float64(qt.scale[0])
    float_values = (qt.values.astype(np.float64) - zp) * sc
    if qt.is_signed:
        new_qmin = -(2 ** (target_bits - 1)); new_qmax = 2 ** (target_bits - 1) - 1
    else:
        new_qmin = 0; new_qmax = 2 ** target_bits - 1
    abs_max = np.maximum(np.abs(float_values.min()), np.abs(float_values.max()))
    new_scale = abs_max / float(new_qmax) if abs_max > 1e-10 else 1e-10
    q_float = float_values / new_scale
    rounded = round_hardware(q_float, mode=hw_config.rounding_mode)
    clamped = np.clip(rounded, new_qmin, new_qmax).astype(np.int64)
    return QuantizedTensor(clamped, target_bits, qt.is_signed,
        np.array([new_scale], dtype=np.float64), np.array([0], dtype=np.int64),
        qt.shape, False, qt.channel_dim)