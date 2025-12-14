"""
Confidence Estimator Module for Virtual Staining

This module provides various methods for estimating confidence in virtual staining predictions.
The core idea is that regions where H&E<->IHC mapping is nearly one-to-one should have
low reconstruction error (high confidence), while regions with one-to-many mapping
(ambiguous, hallucination-prone) should have high error (low confidence).

Methods:
1. Cycle Consistency Error: ||F(G(x)) - x||
2. Multi-Sample Variance: Var across multiple forward passes
3. Worst-Case Error: Max error across multiple samples
4. Ensemble Disagreement: Disagreement between ensemble members
5. SSIM-based Confidence: Structural similarity for reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class ConfidenceEstimator(nn.Module):
    """
    Confidence estimation for virtual staining models.

    Supports multiple confidence estimation strategies:
    - cycle_l1: L1 reconstruction error through cycle
    - cycle_l2: L2 reconstruction error through cycle
    - cycle_ssim: SSIM-based reconstruction quality
    - variance: Prediction variance across multiple samples
    - worst_case: Maximum error across samples
    - combined: Weighted combination of multiple methods
    """

    def __init__(
        self,
        mode: str = 'cycle_l1',
        num_samples: int = 5,
        temperature: float = 5.0,
        bias: float = 2.5,
        use_gpu: bool = True,
        window_size: int = 11
    ):
        """
        Initialize the confidence estimator.

        Args:
            mode: Confidence estimation mode
            num_samples: Number of samples for variance/worst-case estimation
            temperature: Temperature for sigmoid normalization
            bias: Bias for sigmoid normalization
            use_gpu: Whether to use GPU
            window_size: Window size for SSIM computation
        """
        super().__init__()
        self.mode = mode
        self.num_samples = num_samples
        self.temperature = temperature
        self.bias = bias
        self.use_gpu = use_gpu
        self.window_size = window_size

        # Register SSIM window
        self.register_buffer('ssim_window', self._create_ssim_window(window_size, 1))

    def _create_ssim_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def compute_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        size_average: bool = False
    ) -> torch.Tensor:
        """
        Compute SSIM between two images.

        Args:
            img1: First image [B, C, H, W]
            img2: Second image [B, C, H, W]
            window_size: Window size for SSIM
            size_average: Whether to average across spatial dimensions

        Returns:
            SSIM map or scalar
        """
        channel = img1.size(1)
        window = self.ssim_window

        if channel != 1:
            window = self._create_ssim_window(window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=1, keepdim=True)

    def error_to_confidence(self, error: torch.Tensor) -> torch.Tensor:
        """
        Convert reconstruction error to confidence score using sigmoid.

        Args:
            error: Per-pixel error tensor [B, 1, H, W]

        Returns:
            Confidence map in [0, 1] range
        """
        return 1 - torch.sigmoid(error * self.temperature - self.bias)

    def forward(
        self,
        input_img: torch.Tensor,
        recon_img: torch.Tensor,
        method: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute confidence map from input and reconstruction.

        Args:
            input_img: Original input image [B, C, H, W]
            recon_img: Reconstructed image [B, C, H, W]
            method: Override confidence method

        Returns:
            Confidence map [B, 1, H, W]
        """
        mode = method if method else self.mode

        if mode == 'cycle_l1':
            error = torch.abs(recon_img - input_img).mean(dim=1, keepdim=True)
            confidence = self.error_to_confidence(error)

        elif mode == 'cycle_l2':
            error = ((recon_img - input_img) ** 2).mean(dim=1, keepdim=True)
            error = torch.sqrt(error)  # RMSE
            confidence = self.error_to_confidence(error)

        elif mode == 'cycle_ssim':
            ssim_map = self.compute_ssim(input_img, recon_img)
            # SSIM is in [-1, 1], higher is better
            confidence = (ssim_map + 1) / 2

        else:
            # Default to L1
            error = torch.abs(recon_img - input_img).mean(dim=1, keepdim=True)
            confidence = self.error_to_confidence(error)

        return confidence

    def compute_from_samples(
        self,
        input_img: torch.Tensor,
        outputs: torch.Tensor,
        reconstructions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence from multiple samples.

        Args:
            input_img: Original input [B, C, H, W]
            outputs: Multiple generated outputs [N, B, C, H, W]
            reconstructions: Multiple reconstructions [N, B, C, H, W]

        Returns:
            Dictionary with various confidence estimates
        """
        results = {}

        # Mean output
        mean_output = outputs.mean(dim=0)
        results['mean_output'] = mean_output

        # Variance-based confidence
        variance = outputs.var(dim=0).mean(dim=1, keepdim=True)
        results['variance'] = variance
        results['confidence_variance'] = self.error_to_confidence(variance)

        # Mean reconstruction error
        errors = []
        for recon in reconstructions:
            error = torch.abs(recon - input_img).mean(dim=1, keepdim=True)
            errors.append(error)
        errors = torch.stack(errors, dim=0)

        mean_error = errors.mean(dim=0)
        results['mean_error'] = mean_error
        results['confidence_mean'] = self.error_to_confidence(mean_error)

        # Worst-case (max) error
        max_error = errors.max(dim=0)[0]
        results['max_error'] = max_error
        results['confidence_worst_case'] = self.error_to_confidence(max_error)

        # Best-case (min) error
        min_error = errors.min(dim=0)[0]
        results['min_error'] = min_error
        results['confidence_best_case'] = self.error_to_confidence(min_error)

        # Combined confidence (geometric mean)
        conf_var = results['confidence_variance']
        conf_worst = results['confidence_worst_case']
        results['confidence_combined'] = torch.sqrt(conf_var * conf_worst)

        return results


class MultiScaleConfidenceEstimator(ConfidenceEstimator):
    """
    Multi-scale confidence estimation using Gaussian pyramid.
    Provides confidence at multiple resolutions for different-sized structures.
    """

    def __init__(
        self,
        num_scales: int = 4,
        scale_weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize multi-scale estimator.

        Args:
            num_scales: Number of pyramid levels
            scale_weights: Weights for each scale
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        self.num_scales = num_scales
        self.scale_weights = scale_weights or [1.0] * num_scales

    def _downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Downsample image by factor using average pooling."""
        if factor == 1:
            return x
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)

    def forward(
        self,
        input_img: torch.Tensor,
        recon_img: torch.Tensor,
        method: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale confidence.

        Returns:
            Dictionary with confidence at each scale and combined confidence
        """
        results = {}
        confidences = []

        for i in range(self.num_scales):
            factor = 2 ** i
            input_scaled = self._downsample(input_img, factor)
            recon_scaled = self._downsample(recon_img, factor)

            conf = super().forward(input_scaled, recon_scaled, method)
            results[f'confidence_scale_{i}'] = conf
            confidences.append(conf * self.scale_weights[i])

        # Upsample all confidences to original size and combine
        combined = torch.zeros_like(confidences[0])
        total_weight = sum(self.scale_weights)

        for i, conf in enumerate(confidences):
            if i > 0:
                conf = F.interpolate(conf, size=confidences[0].shape[2:],
                                     mode='bilinear', align_corners=False)
            combined += conf

        combined /= total_weight
        results['confidence_combined'] = combined

        return results


class PatchLevelConfidence(nn.Module):
    """
    Compute patch-level confidence scores for region-based abstention.
    Useful for clinical workflows where entire patches may be flagged.
    """

    def __init__(
        self,
        patch_size: int = 64,
        aggregation: str = 'mean',
        threshold: float = 0.5
    ):
        """
        Initialize patch-level confidence.

        Args:
            patch_size: Size of patches for aggregation
            aggregation: How to aggregate pixel confidences ('mean', 'min', 'percentile')
            threshold: Threshold for low-confidence classification
        """
        super().__init__()
        self.patch_size = patch_size
        self.aggregation = aggregation
        self.threshold = threshold

    def forward(
        self,
        pixel_confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute patch-level confidence from pixel-level confidence.

        Args:
            pixel_confidence: Pixel-level confidence [B, 1, H, W]

        Returns:
            patch_confidence: Confidence per patch [B, num_patches_h, num_patches_w]
            low_confidence_mask: Binary mask of low-confidence patches
        """
        B, C, H, W = pixel_confidence.shape
        ps = self.patch_size

        # Reshape into patches
        num_patches_h = H // ps
        num_patches_w = W // ps

        # Crop to fit patches exactly
        confidence_cropped = pixel_confidence[:, :, :num_patches_h * ps, :num_patches_w * ps]

        # Reshape to [B, num_patches_h, num_patches_w, ps, ps]
        confidence_patches = confidence_cropped.view(
            B, 1, num_patches_h, ps, num_patches_w, ps
        ).permute(0, 1, 2, 4, 3, 5).contiguous()

        # Flatten patch pixels
        confidence_patches = confidence_patches.view(B, 1, num_patches_h, num_patches_w, -1)

        # Aggregate
        if self.aggregation == 'mean':
            patch_confidence = confidence_patches.mean(dim=-1).squeeze(1)
        elif self.aggregation == 'min':
            patch_confidence = confidence_patches.min(dim=-1)[0].squeeze(1)
        elif self.aggregation == 'percentile':
            # 10th percentile (conservative)
            k = max(1, int(0.1 * ps * ps))
            patch_confidence = torch.topk(confidence_patches, k, dim=-1, largest=False)[0].mean(dim=-1).squeeze(1)
        else:
            patch_confidence = confidence_patches.mean(dim=-1).squeeze(1)

        # Low confidence mask
        low_confidence_mask = patch_confidence < self.threshold

        return patch_confidence, low_confidence_mask


class ConfidenceCalibrator(nn.Module):
    """
    Calibrate confidence scores using temperature scaling or Platt scaling.
    Improves reliability of confidence estimates.
    """

    def __init__(
        self,
        method: str = 'temperature',
        initial_temp: float = 1.0
    ):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ('temperature', 'platt')
            initial_temp: Initial temperature value
        """
        super().__init__()
        self.method = method

        if method == 'temperature':
            self.temperature = nn.Parameter(torch.tensor(initial_temp))
        elif method == 'platt':
            self.a = nn.Parameter(torch.tensor(1.0))
            self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        """
        Calibrate confidence scores.

        Args:
            confidence: Raw confidence scores [B, 1, H, W]

        Returns:
            Calibrated confidence scores
        """
        if self.method == 'temperature':
            # Temperature scaling on logit space
            logits = torch.log(confidence / (1 - confidence + 1e-8) + 1e-8)
            scaled_logits = logits / self.temperature
            return torch.sigmoid(scaled_logits)

        elif self.method == 'platt':
            # Platt scaling
            logits = torch.log(confidence / (1 - confidence + 1e-8) + 1e-8)
            scaled_logits = self.a * logits + self.b
            return torch.sigmoid(scaled_logits)

        return confidence

    def fit(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_epochs: int = 100,
        lr: float = 0.01
    ):
        """
        Fit calibration parameters on validation data.

        Args:
            predictions: Model confidence predictions
            targets: Binary accuracy labels (1 if correct, 0 if incorrect)
            num_epochs: Number of optimization epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for _ in range(num_epochs):
            optimizer.zero_grad()
            calibrated = self.forward(predictions)
            loss = criterion(calibrated, targets)
            loss.backward()
            optimizer.step()


def compute_abstention_mask(
    confidence_map: torch.Tensor,
    threshold: float = 0.5,
    min_region_size: int = 100,
    erosion_size: int = 3
) -> torch.Tensor:
    """
    Compute abstention mask with morphological post-processing.

    Args:
        confidence_map: Confidence scores [B, 1, H, W]
        threshold: Confidence threshold
        min_region_size: Minimum size for abstention regions
        erosion_size: Erosion kernel size for smoothing

    Returns:
        Binary abstention mask (1 = abstain, 0 = predict)
    """
    # Initial threshold
    abstain_mask = (confidence_map < threshold).float()

    # Morphological operations for smoothing
    if erosion_size > 1:
        kernel = torch.ones(1, 1, erosion_size, erosion_size, device=abstain_mask.device)
        # Erosion followed by dilation (opening)
        eroded = F.conv2d(abstain_mask, kernel, padding=erosion_size // 2)
        eroded = (eroded < erosion_size * erosion_size).float()
        dilated = F.conv2d(1 - eroded, kernel, padding=erosion_size // 2)
        abstain_mask = (dilated < erosion_size * erosion_size).float()

    return abstain_mask


def apply_abstention_visualization(
    image: torch.Tensor,
    confidence_map: torch.Tensor,
    threshold: float = 0.5,
    abstain_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    blend_factor: float = 0.5
) -> torch.Tensor:
    """
    Visualize abstention regions on the generated image.

    Args:
        image: Generated image [B, 3, H, W] in [-1, 1]
        confidence_map: Confidence scores [B, 1, H, W] in [0, 1]
        threshold: Confidence threshold for abstention
        abstain_color: RGB color for abstention overlay (in [0, 1])
        blend_factor: How much to blend abstention color

    Returns:
        Visualization image with abstention regions highlighted
    """
    # Convert image to [0, 1]
    image = (image + 1) / 2

    # Create abstention mask
    abstain_mask = (confidence_map < threshold).float()

    # Create color overlay
    overlay = torch.zeros_like(image)
    overlay[:, 0] = abstain_color[0]
    overlay[:, 1] = abstain_color[1]
    overlay[:, 2] = abstain_color[2]

    # Blend
    result = image * (1 - abstain_mask * blend_factor) + overlay * abstain_mask * blend_factor

    # Convert back to [-1, 1]
    return result * 2 - 1
