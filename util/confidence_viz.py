"""
Visualization Utilities for Confidence-Aware Virtual Staining

This module provides comprehensive visualization tools for:
1. Confidence maps with various colormaps
2. Side-by-side comparisons
3. Uncertainty heatmaps
4. Risk-coverage curve plots
5. Reliability diagrams
6. Interactive HTML reports
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import json


def tensor_to_numpy(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    """
    Convert tensor to numpy array for visualization.

    Args:
        tensor: Input tensor [B, C, H, W] or [C, H, W]
        denormalize: If True, convert from [-1, 1] to [0, 1]

    Returns:
        Numpy array [H, W, C] in [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    arr = tensor.detach().cpu().numpy()

    if arr.shape[0] in [1, 3]:
        arr = arr.transpose(1, 2, 0)

    if denormalize:
        arr = (arr + 1) / 2

    return np.clip(arr, 0, 1)


def apply_colormap(
    values: np.ndarray,
    colormap: str = 'RdYlGn',
    vmin: float = 0,
    vmax: float = 1
) -> np.ndarray:
    """
    Apply colormap to grayscale values.

    Args:
        values: Input values [H, W] in [vmin, vmax]
        colormap: Matplotlib colormap name
        vmin: Minimum value
        vmax: Maximum value

    Returns:
        RGB image [H, W, 3] in [0, 1]
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Normalize to [0, 1]
    normalized = (values - vmin) / (vmax - vmin + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(normalized)

    return colored[:, :, :3]


def create_confidence_overlay(
    image: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.5,
    low_conf_color: Tuple[float, float, float] = (1, 0, 0),
    blend_factor: float = 0.4
) -> np.ndarray:
    """
    Create overlay showing low-confidence regions on image.

    Args:
        image: RGB image [H, W, 3] in [0, 1]
        confidence: Confidence map [H, W] in [0, 1]
        threshold: Confidence threshold
        low_conf_color: Color for low-confidence regions (RGB)
        blend_factor: Blend factor for overlay

    Returns:
        Overlaid image [H, W, 3]
    """
    if confidence.ndim == 3:
        confidence = confidence[:, :, 0]

    # Create mask
    low_conf_mask = (confidence < threshold).astype(float)

    # Create color overlay
    overlay = np.zeros_like(image)
    overlay[:, :, 0] = low_conf_color[0]
    overlay[:, :, 1] = low_conf_color[1]
    overlay[:, :, 2] = low_conf_color[2]

    # Expand mask to 3 channels
    mask_3ch = np.stack([low_conf_mask] * 3, axis=-1)

    # Blend
    result = image * (1 - mask_3ch * blend_factor) + overlay * mask_3ch * blend_factor

    return np.clip(result, 0, 1)


def create_comparison_figure(
    images: Dict[str, np.ndarray],
    titles: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create side-by-side comparison figure.

    Args:
        images: Dictionary of images {name: array}
        titles: Dictionary of titles {name: title}
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, img) in enumerate(images.items()):
        row = idx // n_cols
        col = idx % n_cols

        if img.ndim == 2:
            im = axes[row, col].imshow(img, cmap='RdYlGn', vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        else:
            axes[row, col].imshow(img)

        title = titles.get(name, name) if titles else name
        axes[row, col].set_title(title, fontsize=12)
        axes[row, col].axis('off')

    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_risk_coverage_curve(
    coverages: np.ndarray,
    risks: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Risk-Coverage Curve",
    show_aurc: bool = True
) -> None:
    """
    Plot Risk-Coverage curve(s).

    Args:
        coverages: Coverage values [N] or list of [N]
        risks: Risk values [N] or list of [N]
        labels: Labels for multiple curves
        save_path: Path to save plot
        title: Plot title
        show_aurc: Whether to show AURC in legend
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))

    # Handle single curve
    if not isinstance(coverages, list):
        coverages = [coverages]
        risks = [risks]

    colors = plt.cm.tab10(np.linspace(0, 1, len(coverages)))

    for i, (cov, risk) in enumerate(zip(coverages, risks)):
        # Sort by coverage
        sorted_idx = np.argsort(cov)
        cov_sorted = cov[sorted_idx]
        risk_sorted = risk[sorted_idx]

        # Compute AURC
        aurc = np.trapz(risk_sorted, cov_sorted)

        label = labels[i] if labels else f'Model {i+1}'
        if show_aurc:
            label += f' (AURC={aurc:.4f})'

        plt.plot(cov_sorted, risk_sorted, '-', color=colors[i],
                 linewidth=2, label=label)

    plt.xlabel('Coverage (fraction of predictions made)', fontsize=12)
    plt.ylabel('Risk (average error)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram"
) -> Dict:
    """
    Plot reliability diagram for calibration assessment.

    Args:
        confidences: Per-sample confidence scores
        accuracies: Per-sample binary accuracy
        num_bins: Number of confidence bins
        save_path: Path to save plot
        title: Plot title

    Returns:
        Binned statistics
    """
    import matplotlib.pyplot as plt

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

    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)

    # Calculate ECE
    ece = np.sum(np.abs(bin_accuracies - bin_confidences) * bin_counts) / bin_counts.sum()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    width = 0.08
    ax1.bar(bin_centers, bin_accuracies, width=width, alpha=0.7,
            color='steelblue', edgecolor='navy', label='Accuracy')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'{title}\nECE = {ece:.4f}', fontsize=14)
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Count histogram
    ax2.bar(bin_centers, bin_counts, width=width, alpha=0.7,
            color='gray', edgecolor='black')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Sample Distribution', fontsize=14)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {
        'bin_centers': bin_centers,
        'accuracies': bin_accuracies,
        'confidences': bin_confidences,
        'counts': bin_counts,
        'ece': ece
    }


def plot_confidence_histogram(
    confidence_map: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    title: str = "Confidence Distribution"
) -> None:
    """
    Plot histogram of confidence values.

    Args:
        confidence_map: Confidence values [H, W] or flattened
        threshold: Threshold line
        save_path: Path to save plot
        title: Plot title
    """
    import matplotlib.pyplot as plt

    values = confidence_map.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='navy')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold}')

    # Coverage statistics
    coverage = (values >= threshold).mean()
    plt.axvline(values.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean = {values.mean():.3f}')

    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Pixel Count', fontsize=12)
    plt.title(f'{title}\nCoverage at thresh={threshold}: {coverage:.1%}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_html_report(
    results: Dict,
    output_dir: str,
    report_name: str = "confidence_report.html"
) -> None:
    """
    Generate HTML report with all visualizations.

    Args:
        results: Dictionary with evaluation results
        output_dir: Directory containing images
        report_name: Output HTML filename
    """
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Confidence-Aware Virtual Staining Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; color: #4CAF50; font-weight: bold; }
        .metric-label { color: #777; font-size: 0.9em; }
        .image-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }
        .image-container { background: white; padding: 10px; border-radius: 8px; }
        .image-container img { width: 100%; height: auto; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Confidence-Aware Virtual Staining Report</h1>

        <h2>Summary Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{psnr:.2f}</div>
                <div class="metric-label">PSNR (dB)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{ssim:.4f}</div>
                <div class="metric-label">SSIM</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{aurc:.4f}</div>
                <div class="metric-label">AURC</div>
            </div>
        </div>

        <h2>Risk-Coverage Analysis</h2>
        <div class="image-container">
            <img src="risk_coverage_curve.png" alt="Risk-Coverage Curve">
        </div>

        <h2>Threshold Analysis</h2>
        <table>
            <tr>
                <th>Threshold</th>
                <th>Coverage</th>
                <th>PSNR</th>
                <th>SSIM</th>
            </tr>
            {threshold_rows}
        </table>

        <h2>Sample Visualizations</h2>
        <div class="image-grid">
            {sample_images}
        </div>
    </div>
</body>
</html>
"""

    # Format threshold rows
    threshold_rows = ""
    if 'threshold_analysis' in results:
        for thresh, metrics in results['threshold_analysis'].items():
            threshold_rows += f"""
            <tr>
                <td>{thresh}</td>
                <td>{metrics.get('coverage', 0):.1%}</td>
                <td>{metrics.get('psnr_mean', 0):.2f}</td>
                <td>{metrics.get('ssim_mean', 0):.4f}</td>
            </tr>
"""

    # Format sample images
    sample_images = ""
    viz_dir = os.path.join(output_dir, 'visualizations')
    if os.path.exists(viz_dir):
        for img_file in sorted(os.listdir(viz_dir))[:10]:
            sample_images += f"""
            <div class="image-container">
                <img src="visualizations/{img_file}" alt="{img_file}">
            </div>
"""

    # Fill template
    html_content = html_content.format(
        psnr=results.get('image_quality', {}).get('psnr_mean', 0),
        ssim=results.get('image_quality', {}).get('ssim_mean', 0),
        aurc=results.get('selective_prediction', {}).get('aurc', 0),
        threshold_rows=threshold_rows,
        sample_images=sample_images
    )

    # Save HTML
    with open(os.path.join(output_dir, report_name), 'w') as f:
        f.write(html_content)


def save_confidence_visualization_batch(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    confidence_maps: torch.Tensor,
    save_dir: str,
    batch_idx: int = 0,
    threshold: float = 0.5
) -> None:
    """
    Save batch of visualizations.

    Args:
        inputs: Input images [B, C, H, W]
        outputs: Output images [B, C, H, W]
        confidence_maps: Confidence maps [B, 1, H, W]
        save_dir: Output directory
        batch_idx: Batch index for naming
        threshold: Confidence threshold
    """
    os.makedirs(save_dir, exist_ok=True)

    batch_size = inputs.size(0)

    for i in range(batch_size):
        idx = batch_idx * batch_size + i

        # Convert to numpy
        input_np = tensor_to_numpy(inputs[i])
        output_np = tensor_to_numpy(outputs[i])
        conf_np = confidence_maps[i, 0].cpu().numpy()

        # Create visualizations
        images = {
            'input': input_np,
            'output': output_np,
            'confidence': apply_colormap(conf_np),
            'overlay': create_confidence_overlay(output_np, conf_np, threshold)
        }

        titles = {
            'input': 'Input H&E',
            'output': 'Generated IHC',
            'confidence': 'Confidence Map',
            'overlay': f'Overlay (thresh={threshold})'
        }

        create_comparison_figure(
            images, titles,
            save_path=os.path.join(save_dir, f'sample_{idx:05d}.png')
        )
