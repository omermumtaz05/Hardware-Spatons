#!/usr/bin/env python3
"""ar_distort.py -- AR display distortion model (forward), copied VERBATIM from javad.py
(apply_color_distortion, apply_geometric_distortion, apply_gradient_friendly_distortions).
No dependency on Javad's codebase; needs only torch + kornia (keystone).

AR_86 = the exact projector/glass pipelines stored in #86-NN2_best.pth.
"""
import torch
import torch.nn.functional as F

AR_86 = {
    "projector": {"color_pipeline": [
        {"type": "elevated_black_level", "strength": 0.2},
        {"type": "brightness_both_fixed", "strength": 0.2},
        {"type": "contrast_fixed", "strength": -0.2},
        {"type": "blur", "strength": 0.2},
        {"type": "lsnu", "strength": 0.2}],
        "geometrical_pipeline": [{"type": "keystone", "strength": 0.3}]},
    "glass": {"color_pipeline": [
        {"type": "color_fringes", "strength": 0.4},
        {"type": "vdp_wgnu", "strength": 0.6}],
        "geometrical_pipeline": [
        {"type": "barrel", "strength": 0.3},
        {"type": "fisheye", "strength": 0.3}]},
}

def config_from_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {"projector": ck["projector_config"], "glass": ck["glass_config"]}

# ---------------- verbatim from javad.py ----------------
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


# ---------------- pipeline (same order + clamps as apply_gradient_friendly_distortions) ----------------
def _stage(img, cfg):
    for d in cfg.get("color_pipeline", []):
        img = apply_color_distortion(img, d.get("type", "none"), d.get("strength", 0.3))
    for d in cfg.get("geometrical_pipeline", []):
        img = apply_geometric_distortion(img, d.get("type", "none"), d.get("strength", 0.3))
    return img

def distort(img, ar=AR_86):
    """img: float tensor [3,H,W] in [0,1] -> what the viewer sees through the AR system."""
    img = torch.clamp(img, 0, 1)
    img = _stage(img, ar["projector"])
    img = _stage(img, ar["glass"])
    return torch.clamp(img, 0, 1)
