"""
Quantitative Evaluation: Cycle Confidence vs Stain Predictor Confidence

Evaluates confidence quality using PATCH-LEVEL stain distribution comparison.
No pixel-aligned ground truth required — uses brown ratio statistics from
serial section IHC as reference.

Metrics:
  1. Spearman correlation (confidence vs patch error)
  2. AURC (Area Under Risk-Coverage Curve)
  3. E-AURC (Excess AURC)
  4. Risk at fixed coverage levels
  5. ECE (Expected Calibration Error)

Outputs:
  - Table comparing Cycle vs Stain Predictor (+ MC Dropout baseline)
  - Risk-Coverage curves
  - Scatter plots (confidence vs error)
  - Reliability diagrams

Usage:
    python evaluate_cycle_vs_stain.py \
        --dataroot ../MIST/HER2/TrainValAB \
        --generator_name confidence_her2 \
        --confidence_name unpaired_confidence_her2
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from tqdm import tqdm

from data import create_dataset
from models import create_model
from models.unpaired_confidence import (
    BrownIntensityPredictor,
    ColorDeconvolution,
)


# ============================================================================
# Metric Computation
# ============================================================================

def compute_risk_coverage(errors, confidences, num_thresholds=200):
    """Compute Risk-Coverage curve."""
    thresholds = np.linspace(0, 1, num_thresholds)
    coverages = []
    risks = []

    for t in thresholds:
        mask = confidences >= t
        cov = mask.mean()
        if cov > 0:
            risk = errors[mask].mean()
        else:
            risk = 0.0
        coverages.append(cov)
        risks.append(risk)

    return np.array(coverages), np.array(risks), thresholds


def compute_aurc(coverages, risks):
    """Area Under Risk-Coverage Curve (lower is better)."""
    idx = np.argsort(coverages)
    return np.trapz(risks[idx], coverages[idx])


def compute_optimal_aurc(errors):
    """Oracle AURC: knows true errors, abstains on worst first."""
    sorted_errors = np.sort(errors)
    n = len(sorted_errors)
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(sorted_errors) / np.arange(1, n + 1)
    return np.trapz(risks, coverages)


def compute_ece(confidences, accuracies, num_bins=15):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.mean()
        if prop > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += np.abs(avg_acc - avg_conf) * prop

    return ece


def compute_spearman(confidences, errors):
    """Spearman rank correlation between confidence and negative error."""
    rho, pval = scipy_stats.spearmanr(confidences, -errors)
    return rho, pval


# ============================================================================
# Confidence Methods
# ============================================================================

def compute_cycle_confidence(he_input, generated_ihc, netG_B):
    """Cycle consistency confidence: H&E → IHC → H&E, measure reconstruction error."""
    he_recon = netG_B(generated_ihc, layers=[])
    cycle_error = torch.abs(he_input - he_recon).mean(dim=1, keepdim=True)
    confidence = 1.0 - torch.sigmoid(cycle_error * 5.0 - 2.5)
    return confidence, cycle_error


def compute_stain_confidence(he_input, generated_ihc, stain_predictor, color_deconv):
    """Stain predictor confidence: compare expected vs actual stain intensity."""
    expected_stain, uncertainty = stain_predictor(he_input)
    actual_stain = color_deconv.extract_brown_ratio(generated_ihc)
    deviation = torch.abs(actual_stain - expected_stain)
    normalized_dev = deviation / (uncertainty + 0.1)
    confidence = torch.exp(-normalized_dev)
    return confidence, deviation


def compute_mc_dropout_confidence(he_input, netG_A, n_samples=10):
    """MC Dropout confidence: variance across stochastic forward passes."""
    netG_A.train()  # enable dropout
    outputs = []
    for _ in range(n_samples):
        fake = netG_A(he_input, layers=[])
        outputs.append(fake)
    netG_A.eval()

    outputs = torch.stack(outputs, dim=0)
    variance = outputs.var(dim=0).mean(dim=1, keepdim=True)

    var_flat = variance.view(-1)
    p5 = torch.quantile(var_flat, 0.05)
    p95 = torch.quantile(var_flat, 0.95)
    normalized = (variance - p5) / (p95 - p5 + 1e-8)
    normalized = normalized.clamp(0, 1)

    return 1 - normalized


# ============================================================================
# Patch-Level Error (pseudo ground truth)
# ============================================================================

def compute_patch_stain_error(generated_ihc, real_ihc, color_deconv, patch_size=32):
    """
    Compute patch-level stain distribution error between generated and real IHC.

    Divides image into patches and compares mean brown intensity per patch.
    This does NOT require pixel alignment — only patch-level correspondence.

    Returns:
        error_map: [1, 1, H//ps, W//ps] patch-level error
    """
    gen_brown = color_deconv.extract_brown_ratio(generated_ihc)   # [B,1,H,W]
    real_brown = color_deconv.extract_brown_ratio(real_ihc)       # [B,1,H,W]

    B, C, H, W = gen_brown.shape
    ps = patch_size
    nH, nW = H // ps, W // ps

    # Reshape into patches and compute mean per patch
    gen_patches = gen_brown[:, :, :nH*ps, :nW*ps].reshape(B, C, nH, ps, nW, ps)
    gen_patch_means = gen_patches.mean(dim=(3, 5))  # [B, 1, nH, nW]

    real_patches = real_brown[:, :, :nH*ps, :nW*ps].reshape(B, C, nH, ps, nW, ps)
    real_patch_means = real_patches.mean(dim=(3, 5))  # [B, 1, nH, nW]

    error = torch.abs(gen_patch_means - real_patch_means)
    return error, gen_patch_means, real_patch_means


def downsample_confidence_to_patches(confidence_map, patch_size=32):
    """Average-pool confidence map to match patch grid."""
    return torch.nn.functional.avg_pool2d(confidence_map, patch_size)


# ============================================================================
# Plotting
# ============================================================================

def plot_risk_coverage_comparison(results_dict, save_path):
    """Plot Risk-Coverage curves for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {
        'Cycle Consistency': '#e74c3c',
        'MC Dropout': '#3498db',
        'Stain Predictor (Ours)': '#2ecc71',
    }
    linestyles = {
        'Cycle Consistency': '--',
        'MC Dropout': ':',
        'Stain Predictor (Ours)': '-',
    }

    for name, res in results_dict.items():
        ax.plot(
            res['coverages'], res['risks'],
            label=f"{name} (AURC={res['aurc']:.4f})",
            color=colors.get(name, 'gray'),
            linestyle=linestyles.get(name, '-'),
            linewidth=2,
        )

    # Oracle
    if 'Stain Predictor (Ours)' in results_dict:
        res = results_dict['Stain Predictor (Ours)']
        ax.plot(
            res['oracle_coverages'], res['oracle_risks'],
            label=f"Oracle (AURC={res['optimal_aurc']:.4f})",
            color='black', linestyle='-.', linewidth=1, alpha=0.5,
        )

    ax.set_xlabel('Coverage', fontsize=13)
    ax.set_ylabel('Risk (Patch Stain Error)', fontsize=13)
    ax.set_title('Risk-Coverage Curve: Cycle vs Stain Predictor', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1.02])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_comparison(results_dict, save_path):
    """Scatter plots: confidence vs error for each method."""
    methods = list(results_dict.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = {
        'Cycle Consistency': '#e74c3c',
        'MC Dropout': '#3498db',
        'Stain Predictor (Ours)': '#2ecc71',
    }

    for ax, name in zip(axes, methods):
        res = results_dict[name]
        conf = res['all_confidences']
        err = res['all_errors']

        ax.scatter(conf, err, alpha=0.15, s=8, color=colors.get(name, 'gray'))

        # Trend line
        if len(conf) > 10:
            z = np.polyfit(conf, err, 1)
            p = np.poly1d(z)
            x_line = np.linspace(conf.min(), conf.max(), 100)
            ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.7)

        rho = res['spearman_rho']
        ax.set_title(f'{name}\nSpearman ρ = {rho:.3f}', fontsize=12)
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Patch Stain Error', fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Confidence vs Patch Stain Error',
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reliability_diagrams(results_dict, save_path):
    """Reliability diagrams for calibration analysis."""
    methods = list(results_dict.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = {
        'Cycle Consistency': '#e74c3c',
        'MC Dropout': '#3498db',
        'Stain Predictor (Ours)': '#2ecc71',
    }

    num_bins = 15
    for ax, name in zip(axes, methods):
        res = results_dict[name]
        conf = res['all_confidences']
        err = res['all_errors']

        # Convert error to binary accuracy (below median error = "accurate")
        median_err = np.median(err)
        accuracies = (err <= median_err).astype(float)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_counts = []

        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (conf > lo) & (conf <= hi)
            count = in_bin.sum()
            bin_counts.append(count)
            bin_centers.append((lo + hi) / 2)
            if count > 0:
                bin_accs.append(accuracies[in_bin].mean())
            else:
                bin_accs.append(0)

        bin_centers = np.array(bin_centers)
        bin_accs = np.array(bin_accs)
        bin_counts = np.array(bin_counts)

        # Bar chart
        bar_width = 1.0 / num_bins * 0.8
        ax.bar(
            bin_centers, bin_accs, width=bar_width,
            alpha=0.6, color=colors.get(name, 'gray'),
            edgecolor='black', linewidth=0.5,
        )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect')

        ece = res['ece']
        ax.set_title(f'{name}\nECE = {ece:.4f}', fontsize=12)
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Accuracy (error ≤ median)', fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Reliability Diagrams',
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_risk_at_coverage(results_dict, save_path):
    """Bar chart: risk at fixed coverage levels."""
    coverage_levels = [0.9, 0.7, 0.5, 0.3]
    methods = list(results_dict.keys())

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {
        'Cycle Consistency': '#e74c3c',
        'MC Dropout': '#3498db',
        'Stain Predictor (Ours)': '#2ecc71',
    }

    x = np.arange(len(coverage_levels))
    width = 0.8 / len(methods)

    for i, name in enumerate(methods):
        res = results_dict[name]
        risks_at_cov = []
        for target_cov in coverage_levels:
            covs = res['coverages']
            risks = res['risks']
            # Find closest coverage
            idx = np.argmin(np.abs(covs - target_cov))
            risks_at_cov.append(risks[idx])

        ax.bar(
            x + i * width, risks_at_cov, width,
            label=name, color=colors.get(name, 'gray'),
            edgecolor='black', linewidth=0.5,
        )

    ax.set_xlabel('Target Coverage', fontsize=13)
    ax.set_ylabel('Risk (Patch Stain Error)', fontsize=13)
    ax.set_title('Risk at Fixed Coverage Levels (lower is better)', fontsize=14)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f'{c:.0%}' for c in coverage_levels])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_results_table(results_dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 85)
    print(f"{'Method':<25} {'Spearman ρ↑':>12} {'AURC↓':>10} {'E-AURC↓':>10} {'ECE↓':>10} {'R@50%↓':>10}")
    print("-" * 85)

    for name, res in results_dict.items():
        # Risk at 50% coverage
        covs = res['coverages']
        risks = res['risks']
        idx_50 = np.argmin(np.abs(covs - 0.5))
        r_at_50 = risks[idx_50]

        print(
            f"{name:<25} "
            f"{res['spearman_rho']:>12.4f} "
            f"{res['aurc']:>10.4f} "
            f"{res['e_aurc']:>10.4f} "
            f"{res['ece']:>10.4f} "
            f"{r_at_50:>10.4f}"
        )

    print("=" * 85)
    print("↑ = higher is better, ↓ = lower is better")
    print("R@50% = Risk when keeping top 50% most confident patches\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Quantitative Evaluation: Cycle vs Stain Predictor Confidence'
    )

    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--phase', type=str, default='val')
    parser.add_argument('--generator_name', type=str, required=True)
    parser.add_argument('--generator_epoch', type=str, default='latest')
    parser.add_argument('--confidence_name', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Patch size for patch-level stain error')
    parser.add_argument('--mc_samples', type=int, default=10)
    parser.add_argument('--eval_mc_dropout', action='store_true', default=False,
                        help='Also evaluate MC Dropout (requires dropout layers)')

    args = parser.parse_args()
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x]
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu')

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    class GenOpt:
        pass

    gen_opt = GenOpt()
    gen_opt.dataroot = args.dataroot
    gen_opt.name = args.generator_name
    gen_opt.epoch = args.generator_epoch
    gen_opt.checkpoints_dir = args.checkpoints_dir
    gen_opt.model = 'confidence'
    gen_opt.isTrain = False
    gen_opt.gpu_ids = args.gpu_ids
    gen_opt.input_nc = 3
    gen_opt.output_nc = 3
    gen_opt.ngf = 64
    gen_opt.netG = 'resnet_9blocks'
    gen_opt.normG = 'instance'
    gen_opt.no_dropout = not args.eval_mc_dropout  # need dropout for MC
    gen_opt.init_type = 'normal'
    gen_opt.init_gain = 0.02
    gen_opt.no_antialias = False
    gen_opt.no_antialias_up = False
    gen_opt.weight_norm = 'none'
    gen_opt.load_size = args.load_size
    gen_opt.crop_size = args.crop_size
    gen_opt.preprocess = 'resize_and_crop'
    gen_opt.no_flip = True
    gen_opt.direction = 'AtoB'
    gen_opt.dataset_mode = 'aligned'
    gen_opt.serial_batches = True
    gen_opt.num_threads = 0
    gen_opt.batch_size = 1
    gen_opt.phase = args.phase
    gen_opt.max_dataset_size = float('inf')
    gen_opt.verbose = False
    gen_opt.confidence_mode = 'cycle_l1'
    gen_opt.load_discriminator = False

    model = create_model(gen_opt)
    model.setup(gen_opt)
    model.eval()
    netG_A = model.netG_A
    netG_B = model.netG_B
    print("  Bidirectional generator loaded (G_A + G_B)")

    # Load stain predictor
    confidence_dir = os.path.join(args.checkpoints_dir, args.confidence_name)
    config_path = os.path.join(confidence_dir, 'config.pth')
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location=device)
        predictor_ngf = config.get('predictor_ngf', 64)
    else:
        predictor_ngf = 64

    stain_predictor = BrownIntensityPredictor(input_nc=3, ngf=predictor_ngf).to(device)
    brown_path = os.path.join(confidence_dir, 'latest_brown_predictor.pth')
    if os.path.exists(brown_path):
        checkpoint = torch.load(brown_path, map_location=device)
        stain_predictor.load_state_dict(checkpoint['model_state_dict'])
        stain_predictor.eval()
        print(f"  Stain predictor loaded from {brown_path}")
    else:
        print(f"  WARNING: {brown_path} not found!")

    color_deconv = ColorDeconvolution()

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    dataset = create_dataset(gen_opt)
    print(f"  Dataset: {len(dataset)} images")

    # ------------------------------------------------------------------
    # 3. Collect per-patch confidence and error
    # ------------------------------------------------------------------
    print(f"\nEvaluating (patch_size={args.patch_size})...")

    ps = args.patch_size

    all_data = {
        'Cycle Consistency': {'confidences': [], 'errors': []},
        'Stain Predictor (Ours)': {'confidences': [], 'errors': []},
    }
    if args.eval_mc_dropout:
        all_data['MC Dropout'] = {'confidences': [], 'errors': []}

    for i, data in enumerate(tqdm(dataset, desc="Processing")):
        he_input = data['A'].to(device)
        real_ihc = data['B'].to(device)

        with torch.no_grad():
            # Generate IHC
            generated_ihc = netG_A(he_input, layers=[])

            # Pseudo GT: patch-level stain error
            patch_error, _, _ = compute_patch_stain_error(
                generated_ihc, real_ihc, color_deconv, patch_size=ps
            )
            patch_error_np = patch_error.squeeze().cpu().numpy().flatten()

            # --- Method 1: Cycle Consistency ---
            cycle_conf, _ = compute_cycle_confidence(he_input, generated_ihc, netG_B)
            cycle_conf_patches = downsample_confidence_to_patches(cycle_conf, ps)
            cycle_conf_np = cycle_conf_patches.squeeze().cpu().numpy().flatten()

            # --- Method 2: Stain Predictor ---
            stain_conf, _ = compute_stain_confidence(
                he_input, generated_ihc, stain_predictor, color_deconv
            )
            stain_conf_patches = downsample_confidence_to_patches(stain_conf, ps)
            stain_conf_np = stain_conf_patches.squeeze().cpu().numpy().flatten()

        # Ensure same length
        n = min(len(patch_error_np), len(cycle_conf_np), len(stain_conf_np))
        all_data['Cycle Consistency']['errors'].append(patch_error_np[:n])
        all_data['Cycle Consistency']['confidences'].append(cycle_conf_np[:n])
        all_data['Stain Predictor (Ours)']['errors'].append(patch_error_np[:n])
        all_data['Stain Predictor (Ours)']['confidences'].append(stain_conf_np[:n])

        # --- Method 3: MC Dropout (optional) ---
        if args.eval_mc_dropout:
            with torch.no_grad():
                mc_conf = compute_mc_dropout_confidence(
                    he_input, netG_A, n_samples=args.mc_samples
                )
                mc_conf_patches = downsample_confidence_to_patches(mc_conf, ps)
                mc_conf_np = mc_conf_patches.squeeze().cpu().numpy().flatten()

            all_data['MC Dropout']['errors'].append(patch_error_np[:n])
            all_data['MC Dropout']['confidences'].append(mc_conf_np[:n])

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    print("\nComputing metrics...")

    results_dict = {}

    for name, data_dict in all_data.items():
        errors = np.concatenate(data_dict['errors'])
        confidences = np.concatenate(data_dict['confidences'])

        # Spearman correlation
        rho, pval = compute_spearman(confidences, errors)

        # Risk-Coverage
        coverages, risks, thresholds = compute_risk_coverage(errors, confidences)
        aurc = compute_aurc(coverages, risks)
        opt_aurc = compute_optimal_aurc(errors)
        e_aurc = aurc - opt_aurc

        # Oracle risk-coverage
        sorted_errors = np.sort(errors)
        n_samples = len(sorted_errors)
        oracle_coverages = np.arange(1, n_samples + 1) / n_samples
        oracle_risks = np.cumsum(sorted_errors) / np.arange(1, n_samples + 1)

        # ECE (convert error to binary accuracy using median)
        median_err = np.median(errors)
        accuracies = (errors <= median_err).astype(float)
        ece = compute_ece(confidences, accuracies)

        results_dict[name] = {
            'spearman_rho': rho,
            'spearman_pval': pval,
            'aurc': aurc,
            'optimal_aurc': opt_aurc,
            'e_aurc': e_aurc,
            'ece': ece,
            'coverages': coverages,
            'risks': risks,
            'thresholds': thresholds,
            'oracle_coverages': oracle_coverages,
            'oracle_risks': oracle_risks,
            'all_confidences': confidences,
            'all_errors': errors,
            'n_patches': len(errors),
        }

    # ------------------------------------------------------------------
    # 5. Print table
    # ------------------------------------------------------------------
    print_results_table(results_dict)

    # ------------------------------------------------------------------
    # 6. Save plots
    # ------------------------------------------------------------------
    output_dir = os.path.join(
        args.results_dir,
        f"{args.generator_name}_quantitative_eval",
        f"{args.phase}_{args.generator_epoch}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving plots to {output_dir}...")

    plot_risk_coverage_comparison(
        results_dict,
        os.path.join(output_dir, 'risk_coverage_curve.png'),
    )
    print("  risk_coverage_curve.png")

    plot_scatter_comparison(
        results_dict,
        os.path.join(output_dir, 'scatter_confidence_vs_error.png'),
    )
    print("  scatter_confidence_vs_error.png")

    plot_reliability_diagrams(
        results_dict,
        os.path.join(output_dir, 'reliability_diagrams.png'),
    )
    print("  reliability_diagrams.png")

    plot_risk_at_coverage(
        results_dict,
        os.path.join(output_dir, 'risk_at_coverage.png'),
    )
    print("  risk_at_coverage.png")

    # ------------------------------------------------------------------
    # 7. Save numerical results
    # ------------------------------------------------------------------
    save_results = {}
    for name, res in results_dict.items():
        covs = res['coverages']
        risks_arr = res['risks']

        # Risk at standard coverage levels
        risk_at = {}
        for target in [0.9, 0.7, 0.5, 0.3]:
            idx = np.argmin(np.abs(covs - target))
            risk_at[f'risk@{target:.0%}'] = float(risks_arr[idx])

        save_results[name] = {
            'spearman_rho': float(res['spearman_rho']),
            'spearman_pval': float(res['spearman_pval']),
            'aurc': float(res['aurc']),
            'optimal_aurc': float(res['optimal_aurc']),
            'e_aurc': float(res['e_aurc']),
            'ece': float(res['ece']),
            'n_patches': int(res['n_patches']),
            **risk_at,
        }

    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  results.json")

    # Also save raw data for further analysis
    npz_path = os.path.join(output_dir, 'raw_data.npz')
    npz_data = {}
    for name, res in results_dict.items():
        key = name.replace(' ', '_').replace('(', '').replace(')', '')
        npz_data[f'{key}_confidences'] = res['all_confidences']
        npz_data[f'{key}_errors'] = res['all_errors']
    np.savez(npz_path, **npz_data)
    print(f"  raw_data.npz")

    print(f"\nDone! All results saved to {output_dir}")


if __name__ == '__main__':
    main()
