"""
Inference Script for Confidence-Aware Virtual Staining

This script performs H&E->IHC virtual staining with confidence estimation.
For each input H&E image, it outputs:
1. Synthesized IHC image
2. Pixel-level confidence map
3. Masked IHC image (optional, with low-confidence regions highlighted)

Usage:
    python inference_confidence.py \
        --dataroot ./datasets/test_images \
        --checkpoints_dir ./checkpoints \
        --name confidence_model \
        --epoch latest \
        --results_dir ./results \
        --confidence_threshold 0.5 \
        --num_latent_samples 5 \
        --save_confidence_overlay
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from models.confidence_estimator import (
    ConfidenceEstimator,
    PatchLevelConfidence,
    apply_abstention_visualization
)
import util.util as util


def save_image_tensor(tensor: torch.Tensor, path: str):
    """Save a tensor as an image file."""
    # Convert from [-1, 1] to [0, 255]
    image = (tensor.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def save_confidence_map(conf_map: torch.Tensor, path: str, colormap: str = 'RdYlGn'):
    """Save confidence map as colored image."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    conf_np = conf_map.squeeze().cpu().numpy()

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(conf_np)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)

    Image.fromarray(colored).save(path)


def compute_error_map(pred: torch.Tensor, target: torch.Tensor, mode: str = 'l1') -> torch.Tensor:
    """
    Compute pixel-wise error between prediction and target.

    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        mode: 'l1', 'l2', or 'ssim'

    Returns:
        Error map [B, 1, H, W] where higher = more error
    """
    if mode == 'l1':
        error = torch.abs(pred - target).mean(dim=1, keepdim=True)
    elif mode == 'l2':
        error = ((pred - target) ** 2).mean(dim=1, keepdim=True)
        error = torch.sqrt(error)
    else:
        error = torch.abs(pred - target).mean(dim=1, keepdim=True)

    return error


def error_to_confidence(error: torch.Tensor, method: str = 'sigmoid',
                        temperature: float = 5.0, bias: float = 0.5) -> torch.Tensor:
    """
    Convert error to confidence score.

    Args:
        error: Error map [B, 1, H, W]
        method: 'sigmoid', 'linear', or 'percentile'
        temperature: Temperature for sigmoid
        bias: Bias for sigmoid

    Returns:
        Confidence map [B, 1, H, W] in [0, 1]
    """
    if method == 'sigmoid':
        # Lower error = higher confidence
        confidence = 1 - torch.sigmoid((error - bias) * temperature)
    elif method == 'linear':
        # Normalize to [0, 1] and invert
        min_err = error.min()
        max_err = error.max()
        if max_err - min_err > 1e-6:
            confidence = 1 - (error - min_err) / (max_err - min_err)
        else:
            confidence = torch.ones_like(error)
    elif method == 'percentile':
        # Use percentile-based normalization
        flat = error.flatten()
        p5 = torch.quantile(flat, 0.05)
        p95 = torch.quantile(flat, 0.95)
        normalized = (error - p5) / (p95 - p5 + 1e-6)
        normalized = torch.clamp(normalized, 0, 1)
        confidence = 1 - normalized
    else:
        confidence = 1 - torch.sigmoid((error - bias) * temperature)

    return confidence


def save_comparison_visualization(
    input_img: torch.Tensor,
    output_img: torch.Tensor,
    gt_img: torch.Tensor,
    cycle_confidence: torch.Tensor,
    gt_error_map: torch.Tensor,
    path: str,
    threshold: float = 0.5
):
    """Save comprehensive comparison visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Convert tensors
    input_np = (input_img.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
    output_np = (output_img.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
    gt_np = (gt_img.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
    cycle_conf_np = cycle_confidence.squeeze().cpu().numpy()
    gt_error_np = gt_error_map.squeeze().cpu().numpy()

    # GT-based confidence (invert error)
    gt_conf_np = 1 - (gt_error_np - gt_error_np.min()) / (gt_error_np.max() - gt_error_np.min() + 1e-6)

    # Row 1: Images
    axes[0, 0].imshow(np.clip(input_np, 0, 1))
    axes[0, 0].set_title('Input H&E', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.clip(output_np, 0, 1))
    axes[0, 1].set_title('Generated IHC', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.clip(gt_np, 0, 1))
    axes[0, 2].set_title('Ground Truth IHC', fontsize=12)
    axes[0, 2].axis('off')

    # Difference image
    diff = np.abs(output_np - gt_np).mean(axis=2)
    im = axes[0, 3].imshow(diff, cmap='hot', vmin=0, vmax=1)
    axes[0, 3].set_title('Absolute Error (Generated vs GT)', fontsize=12)
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Row 2: Confidence maps
    im1 = axes[1, 0].imshow(cycle_conf_np, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Cycle Confidence\n(mean: {cycle_conf_np.mean():.3f})', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(gt_conf_np, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title(f'GT-based Confidence\n(mean: {gt_conf_np.mean():.3f})', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Correlation scatter plot
    cycle_flat = cycle_conf_np.flatten()[::100]  # Subsample for speed
    gt_flat = gt_conf_np.flatten()[::100]
    axes[1, 2].scatter(cycle_flat, gt_flat, alpha=0.3, s=1)
    axes[1, 2].plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
    axes[1, 2].set_xlabel('Cycle Confidence')
    axes[1, 2].set_ylabel('GT-based Confidence')
    correlation = np.corrcoef(cycle_flat, gt_flat)[0, 1]
    axes[1, 2].set_title(f'Correlation: {correlation:.3f}', fontsize=12)
    axes[1, 2].legend()
    axes[1, 2].set_xlim([0, 1])
    axes[1, 2].set_ylim([0, 1])

    # Statistics text
    stats_text = f"""Statistics:

Cycle Confidence:
  Mean: {cycle_conf_np.mean():.4f}
  Std: {cycle_conf_np.std():.4f}

GT Error:
  Mean: {gt_error_np.mean():.4f}
  Std: {gt_error_np.std():.4f}

Correlation: {correlation:.4f}

Coverage (cycle > {threshold}): {(cycle_conf_np >= threshold).mean():.1%}
Coverage (GT-based > {threshold}): {(gt_conf_np >= threshold).mean():.1%}
"""
    axes[1, 3].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 3].transAxes)
    axes[1, 3].set_title('Statistics')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return correlation


def save_composite_visualization(
    input_img: torch.Tensor,
    output_img: torch.Tensor,
    confidence_map: torch.Tensor,
    path: str,
    threshold: float = 0.5
):
    """Save composite visualization with all outputs."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert tensors
    input_np = (input_img.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
    output_np = (output_img.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
    conf_np = confidence_map.squeeze().cpu().numpy()

    # Row 1: Input, Output, Real (if available)
    axes[0, 0].imshow(np.clip(input_np, 0, 1))
    axes[0, 0].set_title('Input H&E', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.clip(output_np, 0, 1))
    axes[0, 1].set_title('Generated IHC', fontsize=12)
    axes[0, 1].axis('off')

    # Masked output
    mask = conf_np >= threshold
    masked_output = output_np.copy()
    # Blend low-confidence regions with red
    masked_output[~mask] = masked_output[~mask] * 0.5 + np.array([1, 0, 0]) * 0.5
    axes[0, 2].imshow(np.clip(masked_output, 0, 1))
    axes[0, 2].set_title(f'Masked (thresh={threshold})', fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: Confidence map, histogram, abstention overlay
    im = axes[1, 0].imshow(conf_np, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_title('Confidence Map', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Confidence histogram
    axes[1, 1].hist(conf_np.flatten(), bins=50, alpha=0.7, color='steelblue')
    axes[1, 1].axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Pixel Count')
    axes[1, 1].set_title('Confidence Distribution')
    axes[1, 1].legend()

    # Coverage statistics
    coverage = (conf_np >= threshold).mean()
    mean_conf = conf_np.mean()
    axes[1, 2].text(0.5, 0.6, f'Coverage: {coverage:.1%}', fontsize=14,
                    ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.4, f'Mean Confidence: {mean_conf:.3f}', fontsize=14,
                    ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Statistics')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


class ConfidenceInference:
    """
    Inference class for confidence-aware virtual staining.
    """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()

        # Create confidence estimator
        self.confidence_estimator = ConfidenceEstimator(
            mode=getattr(opt, 'confidence_mode', 'cycle_l1'),
            num_samples=getattr(opt, 'num_latent_samples', 5)
        ).to(self.device)

        # Patch-level confidence
        self.patch_confidence = PatchLevelConfidence(
            patch_size=getattr(opt, 'confidence_patch_size', 64),
            threshold=opt.confidence_threshold
        )

    def process_single(
        self,
        input_tensor: torch.Tensor,
        direction: str = 'AtoB'
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single image with confidence estimation.

        Args:
            input_tensor: Input image [1, C, H, W] in [-1, 1]
            direction: Translation direction

        Returns:
            Dictionary with output, confidence_map, masked_output, etc.
        """
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            # Use model's confidence-aware generation if available
            if hasattr(self.model, 'generate_with_confidence'):
                results = self.model.generate_with_confidence(
                    input_tensor,
                    direction=direction,
                    apply_mask=True
                )
            else:
                # Fallback: manual confidence computation
                if direction == 'AtoB':
                    netG_forward = self.model.netG_A
                    netG_backward = self.model.netG_B
                else:
                    netG_forward = self.model.netG_B
                    netG_backward = self.model.netG_A

                # Forward pass
                output = netG_forward(input_tensor, layers=[])

                # Backward pass for reconstruction
                recon = netG_backward(output, layers=[])

                # Compute cycle reconstruction error
                cycle_error = torch.abs(recon - input_tensor).mean(dim=1, keepdim=True)

                # Use percentile-based normalization for better confidence calibration
                flat_error = cycle_error.flatten()
                p10 = torch.quantile(flat_error, 0.1)
                p90 = torch.quantile(flat_error, 0.9)
                normalized_error = (cycle_error - p10) / (p90 - p10 + 1e-6)
                normalized_error = torch.clamp(normalized_error, 0, 1)
                confidence_map = 1 - normalized_error

                results = {
                    'output': output,
                    'confidence_map': confidence_map,
                    'reconstruction': recon,
                    'cycle_error': cycle_error
                }

            # Add patch-level confidence
            patch_conf, low_conf_mask = self.patch_confidence(results['confidence_map'])
            results['patch_confidence'] = patch_conf
            results['low_confidence_patches'] = low_conf_mask

            # Create abstention visualization
            results['abstention_viz'] = apply_abstention_visualization(
                results['output'],
                results['confidence_map'],
                threshold=self.opt.confidence_threshold
            )

        return results

    def process_with_multiple_samples(
        self,
        input_tensor: torch.Tensor,
        num_samples: int = 5,
        direction: str = 'AtoB'
    ) -> Dict[str, torch.Tensor]:
        """
        Process with multiple forward passes for robust confidence.

        Args:
            input_tensor: Input image
            num_samples: Number of samples
            direction: Translation direction

        Returns:
            Dictionary with aggregated results
        """
        input_tensor = input_tensor.to(self.device)

        if hasattr(self.model, 'compute_confidence_with_sampling'):
            mean_output, confidence_map, all_outputs = self.model.compute_confidence_with_sampling(
                input_tensor,
                num_samples=num_samples,
                direction=direction
            )
            return {
                'output': mean_output,
                'confidence_map': confidence_map,
                'all_outputs': all_outputs
            }
        else:
            # Fallback to single sample
            return self.process_single(input_tensor, direction)


def run_inference(opt):
    """Main inference function."""
    # Create inference engine
    inferencer = ConfidenceInference(opt)

    # Create dataset
    dataset = create_dataset(opt)

    # Create output directory
    output_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'confidence_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)

    if opt.save_confidence_overlay:
        os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)

    print(f"Processing {len(dataset)} images...")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {opt.confidence_threshold}")
    print(f"Number of latent samples: {opt.num_latent_samples}")

    # Statistics collection
    all_coverages = []
    all_mean_confidences = []
    all_correlations = []
    all_gt_errors = []

    for i, data in enumerate(tqdm(dataset)):
        # Get image paths
        if 'A_paths' in data:
            img_path = data['A_paths'][0]
        else:
            img_path = f"image_{i:05d}"

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Set input
        inferencer.model.set_input(data)
        input_tensor = inferencer.model.real_A

        # Check if ground truth is available (for paired datasets)
        has_gt = hasattr(inferencer.model, 'real_B') and inferencer.model.real_B is not None
        gt_tensor = inferencer.model.real_B if has_gt else None

        # Process with confidence
        if opt.num_latent_samples > 1:
            results = inferencer.process_with_multiple_samples(
                input_tensor,
                num_samples=opt.num_latent_samples
            )
        else:
            results = inferencer.process_single(input_tensor)

        output = results['output']
        confidence_map = results['confidence_map']

        # Compute statistics
        coverage = (confidence_map >= opt.confidence_threshold).float().mean().item()
        mean_confidence = confidence_map.mean().item()
        all_coverages.append(coverage)
        all_mean_confidences.append(mean_confidence)

        # Save outputs
        save_image_tensor(output, os.path.join(output_dir, 'outputs', f'{img_name}.png'))
        save_confidence_map(confidence_map, os.path.join(output_dir, 'confidence_maps', f'{img_name}_conf.png'))

        # If GT available, compute GT-based error and correlation
        if has_gt and gt_tensor is not None:
            gt_error_map = compute_error_map(output, gt_tensor, mode='l1')
            gt_error_mean = gt_error_map.mean().item()
            all_gt_errors.append(gt_error_mean)

            # Save comparison visualization with correlation analysis
            correlation = save_comparison_visualization(
                input_tensor, output, gt_tensor,
                confidence_map, gt_error_map,
                os.path.join(output_dir, 'comparisons', f'{img_name}_comparison.png'),
                threshold=opt.confidence_threshold
            )
            all_correlations.append(correlation)
        else:
            # Save simple composite visualization
            save_composite_visualization(
                input_tensor, output, confidence_map,
                os.path.join(output_dir, 'visualizations', f'{img_name}_viz.png'),
                threshold=opt.confidence_threshold
            )

        # Save overlay
        if opt.save_confidence_overlay:
            overlay = results.get('abstention_viz', output)
            save_image_tensor(overlay, os.path.join(output_dir, 'overlays', f'{img_name}_overlay.png'))

    # Print summary statistics
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(dataset)}")
    print(f"Mean coverage (conf >= {opt.confidence_threshold}): {np.mean(all_coverages):.1%}")
    print(f"Mean confidence: {np.mean(all_mean_confidences):.3f}")

    if all_correlations:
        print(f"\n[GT Analysis]")
        print(f"Mean GT error: {np.mean(all_gt_errors):.4f}")
        print(f"Mean correlation (cycle conf vs GT error): {np.mean(all_correlations):.4f}")
        if np.mean(all_correlations) < 0.3:
            print("\n⚠️  WARNING: Low correlation between cycle confidence and GT error!")
            print("   This suggests cycle-consistency may not be a good proxy for prediction quality.")
            print("   Consider:")
            print("   1. Training longer (more epochs)")
            print("   2. Increasing lambda_cycle weight")
            print("   3. Using a different confidence estimation method")

    print(f"\nResults saved to: {output_dir}")

    # Save summary
    summary = {
        'num_images': len(dataset),
        'confidence_threshold': opt.confidence_threshold,
        'num_latent_samples': opt.num_latent_samples,
        'mean_coverage': float(np.mean(all_coverages)),
        'std_coverage': float(np.std(all_coverages)),
        'mean_confidence': float(np.mean(all_mean_confidences)),
        'std_confidence': float(np.std(all_mean_confidences)),
        'coverages': all_coverages,
        'mean_confidences': all_mean_confidences
    }

    if all_correlations:
        summary['mean_gt_error'] = float(np.mean(all_gt_errors))
        summary['mean_correlation'] = float(np.mean(all_correlations))
        summary['correlations'] = all_correlations
        summary['gt_errors'] = all_gt_errors

    import json
    with open(os.path.join(output_dir, 'inference_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def add_inference_options(parser):
    """Add inference-specific options."""
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Threshold for low-confidence masking')
    parser.add_argument('--num_latent_samples', type=int, default=1,
                        help='Number of latent samples for confidence estimation')
    parser.add_argument('--save_confidence_overlay', action='store_true',
                        help='Save images with confidence overlay')
    parser.add_argument('--confidence_patch_size', type=int, default=64,
                        help='Patch size for patch-level confidence')
    return parser


class ConfidenceTestOptions(TestOptions):
    """Test options with confidence-specific arguments."""

    def initialize(self, parser):
        parser = super().initialize(parser)
        # Add inference-specific options (not already in model)
        # Note: confidence_threshold, num_latent_samples are defined in ConfidenceModel
        parser.add_argument('--save_confidence_overlay', action='store_true',
                            help='Save images with confidence overlay')
        parser.add_argument('--confidence_patch_size', type=int, default=64,
                            help='Patch size for patch-level confidence')
        # Set defaults for confidence model
        parser.set_defaults(model='confidence')
        return parser


if __name__ == '__main__':
    # Parse options with confidence-specific arguments
    opt = ConfidenceTestOptions().parse()

    # Set defaults for inference
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # Run inference
    run_inference(opt)
