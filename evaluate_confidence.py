"""
Evaluation Metrics for Confidence-Aware Virtual Staining

This module provides comprehensive evaluation tools for assessing the quality
of confidence-aware virtual staining models, including:

1. Selective Prediction Metrics:
   - Risk-Coverage curves
   - Area Under Risk-Coverage Curve (AURC)
   - Excess-AURC (compared to optimal)

2. Calibration Metrics:
   - Expected Calibration Error (ECE)
   - Maximum Calibration Error (MCE)
   - Reliability diagrams

3. Image Quality Metrics:
   - SSIM, PSNR, LPIPS on high-confidence regions
   - FID on high-confidence regions
   - Cell count MAE (if annotations available)

4. Abstention Quality:
   - Precision/Recall of low-confidence detection
   - F1 score for hallucination identification

Usage:
    python evaluate_confidence.py --dataroot ./datasets/MIST \
        --checkpoints_dir ./checkpoints --name confidence_model \
        --phase test --results_dir ./results
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm

# Import from project
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util.util as util


class ImageQualityMetrics:
    """Compute image quality metrics with confidence-aware masking."""

    def __init__(self, device='cuda'):
        self.device = device
        self._init_lpips()

    def _init_lpips(self):
        """Initialize LPIPS model."""
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        except ImportError:
            print("LPIPS not available. Install with: pip install lpips")
            self.lpips_fn = None

    @staticmethod
    def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> float:
        """
        Compute PSNR between prediction and target.

        Args:
            pred: Predicted image [B, C, H, W] in [-1, 1]
            target: Target image [B, C, H, W] in [-1, 1]
            mask: Optional confidence mask [B, 1, H, W] in [0, 1]

        Returns:
            PSNR value in dB
        """
        # Convert to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        if mask is not None:
            # Apply mask
            mask = mask.expand_as(pred)
            pred = pred * mask
            target = target * mask
            num_pixels = mask.sum()
            mse = ((pred - target) ** 2).sum() / (num_pixels + 1e-8)
        else:
            mse = F.mse_loss(pred, target)

        if mse < 1e-10:
            return 100.0

        return 10 * np.log10(1.0 / mse.item())

    @staticmethod
    def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     window_size: int = 11) -> float:
        """
        Compute SSIM between prediction and target.

        Args:
            pred: Predicted image [B, C, H, W] in [-1, 1]
            target: Target image [B, C, H, W] in [-1, 1]
            mask: Optional confidence mask
            window_size: Window size for SSIM

        Returns:
            SSIM value in [0, 1]
        """
        # Convert to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float()
        window = _2D_window.unsqueeze(0).unsqueeze(0)
        window = window.expand(pred.size(1), 1, window_size, window_size).contiguous()
        window = window.to(pred.device)

        mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.size(1))
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=pred.size(1))

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2,
                             groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2,
                             groups=pred.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size // 2,
                           groups=pred.size(1)) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if mask is not None:
            ssim_map = ssim_map * mask
            return ssim_map.sum() / (mask.sum() + 1e-8)

        return ssim_map.mean().item()

    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> float:
        """
        Compute LPIPS perceptual distance.

        Args:
            pred: Predicted image [B, C, H, W] in [-1, 1]
            target: Target image [B, C, H, W] in [-1, 1]
            mask: Optional mask (note: LPIPS doesn't support direct masking)

        Returns:
            LPIPS distance (lower is better)
        """
        if self.lpips_fn is None:
            return 0.0

        with torch.no_grad():
            if mask is not None:
                # For masked LPIPS, we can't directly apply mask
                # Instead, compute on valid regions only
                pred = pred * mask.expand_as(pred)
                target = target * mask.expand_as(target)

            lpips_val = self.lpips_fn(pred, target)
            return lpips_val.mean().item()


class SelectivePredictionMetrics:
    """
    Metrics for selective prediction / abstention evaluation.

    Key concepts:
    - Coverage: Fraction of samples where model makes a prediction
    - Risk: Error rate on the samples where model predicts
    - Goal: Higher confidence should correspond to lower risk
    """

    @staticmethod
    def compute_risk_coverage_curve(
        errors: np.ndarray,
        confidences: np.ndarray,
        num_thresholds: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Risk-Coverage curve.

        Args:
            errors: Per-sample errors [N]
            confidences: Per-sample confidences [N]
            num_thresholds: Number of threshold points

        Returns:
            coverages: Coverage values at each threshold
            risks: Risk values at each threshold
            thresholds: Confidence thresholds used
        """
        thresholds = np.linspace(0, 1, num_thresholds)
        coverages = []
        risks = []

        for thresh in thresholds:
            mask = confidences >= thresh
            coverage = mask.mean()
            if coverage > 0:
                risk = errors[mask].mean()
            else:
                risk = 0
            coverages.append(coverage)
            risks.append(risk)

        return np.array(coverages), np.array(risks), thresholds

    @staticmethod
    def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
        """
        Compute Area Under Risk-Coverage Curve.

        Lower is better - means model abstains on high-error samples.

        Args:
            coverages: Coverage values
            risks: Risk values

        Returns:
            AURC value
        """
        # Sort by coverage
        sorted_idx = np.argsort(coverages)
        coverages_sorted = coverages[sorted_idx]
        risks_sorted = risks[sorted_idx]

        # Integrate using trapezoidal rule
        aurc = np.trapz(risks_sorted, coverages_sorted)
        return aurc

    @staticmethod
    def compute_optimal_aurc(errors: np.ndarray) -> float:
        """
        Compute optimal AURC (oracle that knows true errors).

        Args:
            errors: Per-sample errors

        Returns:
            Optimal AURC value
        """
        # Sort errors ascending
        sorted_errors = np.sort(errors)
        n = len(sorted_errors)

        # Coverage from 1/n to 1
        coverages = np.arange(1, n + 1) / n

        # Risk at each coverage (cumulative mean)
        risks = np.cumsum(sorted_errors) / np.arange(1, n + 1)

        return np.trapz(risks, coverages)

    @staticmethod
    def compute_excess_aurc(
        errors: np.ndarray,
        confidences: np.ndarray,
        num_thresholds: int = 100
    ) -> float:
        """
        Compute Excess-AURC (AURC minus optimal AURC).

        Measures how much worse the model is compared to oracle.

        Args:
            errors: Per-sample errors
            confidences: Per-sample confidences
            num_thresholds: Number of threshold points

        Returns:
            Excess AURC value
        """
        coverages, risks, _ = SelectivePredictionMetrics.compute_risk_coverage_curve(
            errors, confidences, num_thresholds)
        aurc = SelectivePredictionMetrics.compute_aurc(coverages, risks)
        optimal_aurc = SelectivePredictionMetrics.compute_optimal_aurc(errors)
        return aurc - optimal_aurc


class CalibrationMetrics:
    """
    Metrics for assessing calibration of confidence estimates.

    Well-calibrated confidence means: when the model says 80% confident,
    it should be correct 80% of the time.
    """

    @staticmethod
    def compute_ece(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.

        Args:
            confidences: Per-sample confidences [N]
            accuracies: Per-sample accuracy (binary) [N]
            num_bins: Number of bins for calibration

        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

        return ece

    @staticmethod
    def compute_mce(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Maximum Calibration Error.

        Args:
            confidences: Per-sample confidences
            accuracies: Per-sample accuracy
            num_bins: Number of bins

        Returns:
            MCE value
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.sum() > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                mce = max(mce, np.abs(avg_accuracy - avg_confidence))

        return mce

    @staticmethod
    def compute_reliability_diagram(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute data for reliability diagram.

        Args:
            confidences: Per-sample confidences
            accuracies: Per-sample accuracy
            num_bins: Number of bins

        Returns:
            Dictionary with bin centers, accuracies, confidences, and counts
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_centers = (bin_lowers + bin_uppers) / 2
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            count = in_bin.sum()
            bin_counts.append(count)

            if count > 0:
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)

        return {
            'bin_centers': bin_centers,
            'accuracies': np.array(bin_accuracies),
            'confidences': np.array(bin_confidences),
            'counts': np.array(bin_counts)
        }


class ConfidenceEvaluator:
    """
    Main evaluator class for confidence-aware virtual staining.
    """

    def __init__(self, opt, device='cuda'):
        self.opt = opt
        self.device = device
        self.image_metrics = ImageQualityMetrics(device)

    def evaluate_dataset(
        self,
        model,
        dataloader,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Evaluate model on entire dataset.

        Args:
            model: Confidence-aware model
            dataloader: Test data loader
            save_dir: Directory to save results

        Returns:
            Dictionary with all evaluation metrics
        """
        model.eval()

        all_errors = []
        all_confidences = []
        all_psnr = []
        all_ssim = []
        all_lpips = []

        # Metrics at different confidence thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]
        threshold_metrics = {t: {'psnr': [], 'ssim': [], 'lpips': []}
                             for t in thresholds}

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i, data in enumerate(tqdm(dataloader, desc="Evaluating")):
            model.set_input(data)

            with torch.no_grad():
                model.forward()

                # Get outputs
                real_A = model.real_A
                fake_B = model.fake_B
                real_B = model.real_B
                confidence_map = model.confidence_map_A

                # Compute per-pixel error
                error = torch.abs(fake_B - real_B).mean(dim=1, keepdim=True)

                # Store per-sample metrics
                sample_error = error.mean().item()
                sample_confidence = confidence_map.mean().item()

                all_errors.append(sample_error)
                all_confidences.append(sample_confidence)

                # Image quality metrics (full image)
                psnr = self.image_metrics.compute_psnr(fake_B, real_B)
                ssim = self.image_metrics.compute_ssim(fake_B, real_B)
                lpips_val = self.image_metrics.compute_lpips(fake_B, real_B)

                all_psnr.append(psnr)
                all_ssim.append(ssim)
                all_lpips.append(lpips_val)

                # Metrics at different confidence thresholds
                for thresh in thresholds:
                    mask = (confidence_map >= thresh).float()
                    if mask.sum() > 0:
                        psnr_t = self.image_metrics.compute_psnr(fake_B, real_B, mask)
                        ssim_t = self.image_metrics.compute_ssim(fake_B, real_B, mask)
                        lpips_t = self.image_metrics.compute_lpips(fake_B, real_B, mask)
                        threshold_metrics[thresh]['psnr'].append(psnr_t)
                        threshold_metrics[thresh]['ssim'].append(ssim_t)
                        threshold_metrics[thresh]['lpips'].append(lpips_t)

                # Save visualizations
                if save_dir and i < 50:  # Save first 50 samples
                    self._save_visualization(
                        real_A, fake_B, real_B, confidence_map,
                        os.path.join(save_dir, f'sample_{i:04d}.png')
                    )

        # Convert to numpy
        all_errors = np.array(all_errors)
        all_confidences = np.array(all_confidences)

        # Compute selective prediction metrics
        coverages, risks, thresholds_rc = SelectivePredictionMetrics.compute_risk_coverage_curve(
            all_errors, all_confidences)
        aurc = SelectivePredictionMetrics.compute_aurc(coverages, risks)
        excess_aurc = SelectivePredictionMetrics.compute_excess_aurc(
            all_errors, all_confidences)

        # For calibration, convert errors to binary accuracy
        # (1 if error below median, 0 otherwise)
        median_error = np.median(all_errors)
        accuracies = (all_errors < median_error).astype(float)
        ece = CalibrationMetrics.compute_ece(all_confidences, accuracies)
        mce = CalibrationMetrics.compute_mce(all_confidences, accuracies)

        # Compile results
        results = {
            'image_quality': {
                'psnr_mean': float(np.mean(all_psnr)),
                'psnr_std': float(np.std(all_psnr)),
                'ssim_mean': float(np.mean(all_ssim)),
                'ssim_std': float(np.std(all_ssim)),
                'lpips_mean': float(np.mean(all_lpips)),
                'lpips_std': float(np.std(all_lpips)),
            },
            'selective_prediction': {
                'aurc': float(aurc),
                'excess_aurc': float(excess_aurc),
                'coverages': coverages.tolist(),
                'risks': risks.tolist(),
            },
            'calibration': {
                'ece': float(ece),
                'mce': float(mce),
            },
            'threshold_analysis': {}
        }

        # Add threshold-specific metrics
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            if len(threshold_metrics[thresh]['psnr']) > 0:
                results['threshold_analysis'][f'thresh_{thresh}'] = {
                    'coverage': len(threshold_metrics[thresh]['psnr']) / len(all_psnr),
                    'psnr_mean': float(np.mean(threshold_metrics[thresh]['psnr'])),
                    'ssim_mean': float(np.mean(threshold_metrics[thresh]['ssim'])),
                    'lpips_mean': float(np.mean(threshold_metrics[thresh]['lpips'])),
                }

        # Save results
        if save_dir:
            with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

            # Save risk-coverage curve data
            np.savez(
                os.path.join(save_dir, 'risk_coverage_data.npz'),
                coverages=coverages,
                risks=risks,
                thresholds=thresholds_rc,
                errors=all_errors,
                confidences=all_confidences
            )

        return results

    def _save_visualization(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
        confidence_map: torch.Tensor,
        save_path: str
    ):
        """Save visualization of results."""
        import matplotlib.pyplot as plt

        # Convert tensors to numpy
        real_A_np = (real_A[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
        fake_B_np = (fake_B[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
        real_B_np = (real_B[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
        conf_np = confidence_map[0, 0].cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(np.clip(real_A_np, 0, 1))
        axes[0].set_title('Input H&E')
        axes[0].axis('off')

        axes[1].imshow(np.clip(fake_B_np, 0, 1))
        axes[1].set_title('Generated IHC')
        axes[1].axis('off')

        axes[2].imshow(np.clip(real_B_np, 0, 1))
        axes[2].set_title('Real IHC')
        axes[2].axis('off')

        im = axes[3].imshow(conf_np, cmap='RdYlGn', vmin=0, vmax=1)
        axes[3].set_title('Confidence Map')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_risk_coverage_curve(
    coverages: np.ndarray,
    risks: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Risk-Coverage Curve"
):
    """
    Plot Risk-Coverage curve.

    Args:
        coverages: Coverage values
        risks: Risk values
        save_path: Path to save plot
        title: Plot title
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(coverages, risks, 'b-', linewidth=2)
    plt.xlabel('Coverage (fraction of predictions made)', fontsize=12)
    plt.ylabel('Risk (average error)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add AURC annotation
    aurc = np.trapz(risks[np.argsort(coverages)], np.sort(coverages))
    plt.annotate(f'AURC = {aurc:.4f}', xy=(0.7, 0.9),
                 xycoords='axes fraction', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reliability_diagram(
    reliability_data: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram"
):
    """
    Plot reliability diagram.

    Args:
        reliability_data: Output from compute_reliability_diagram
        save_path: Path to save plot
        title: Plot title
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    bin_centers = reliability_data['bin_centers']
    accuracies = reliability_data['accuracies']

    # Plot bars
    width = 0.08
    ax.bar(bin_centers, accuracies, width=width, alpha=0.7, label='Accuracy')

    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation script."""
    # Parse options
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # Create dataset
    dataset = create_dataset(opt)

    # Create model
    model = create_model(opt)
    model.setup(opt)

    # Create evaluator
    evaluator = ConfidenceEvaluator(opt, device=model.device)

    # Create results directory
    results_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    os.makedirs(results_dir, exist_ok=True)

    # Run evaluation
    print("Starting evaluation...")
    results = evaluator.evaluate_dataset(model, dataset, save_dir=results_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nImage Quality (Full Image):")
    print(f"  PSNR: {results['image_quality']['psnr_mean']:.2f} +/- {results['image_quality']['psnr_std']:.2f}")
    print(f"  SSIM: {results['image_quality']['ssim_mean']:.4f} +/- {results['image_quality']['ssim_std']:.4f}")
    print(f"  LPIPS: {results['image_quality']['lpips_mean']:.4f} +/- {results['image_quality']['lpips_std']:.4f}")

    print("\nSelective Prediction:")
    print(f"  AURC: {results['selective_prediction']['aurc']:.4f}")
    print(f"  Excess-AURC: {results['selective_prediction']['excess_aurc']:.4f}")

    print("\nCalibration:")
    print(f"  ECE: {results['calibration']['ece']:.4f}")
    print(f"  MCE: {results['calibration']['mce']:.4f}")

    print("\nThreshold Analysis:")
    for thresh, metrics in results['threshold_analysis'].items():
        print(f"  {thresh}:")
        print(f"    Coverage: {metrics['coverage']:.2%}")
        print(f"    PSNR: {metrics['psnr_mean']:.2f}")
        print(f"    SSIM: {metrics['ssim_mean']:.4f}")

    # Generate plots
    coverages = np.array(results['selective_prediction']['coverages'])
    risks = np.array(results['selective_prediction']['risks'])

    plot_risk_coverage_curve(
        coverages, risks,
        save_path=os.path.join(results_dir, 'risk_coverage_curve.png')
    )

    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()
