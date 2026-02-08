"""
Inference Script for Unpaired Confidence Estimation

This script performs inference with confidence estimation that works
WITHOUT pixel-aligned GT. Perfect for serial sections!

Confidence is computed by:
1. Brown Intensity Deviation: Expected vs actual brown staining
2. Hallucination Detector: Real vs generated detection score
3. MC Dropout: Prediction variance

Usage:
    python inference_unpaired_confidence.py \
        --dataroot ./datasets/MIST/HER2/TrainValAB \
        --generator_name confidence_her2 \
        --confidence_name unpaired_confidence_her2 \
        --phase val
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from models.unpaired_confidence import (
    BrownIntensityPredictor,
    HallucinationDetector,
    UnpairedConfidenceEstimator,
    ColorDeconvolution
)


def save_image(tensor, path):
    """Save tensor as image."""
    img = (tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2 * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_heatmap(tensor, path, cmap='RdYlGn', vmin=0, vmax=1):
    """Save tensor as heatmap."""
    import matplotlib.cm as cm

    arr = tensor.squeeze().cpu().detach().numpy()
    cmap_fn = cm.get_cmap(cmap)
    colored = cmap_fn((arr - vmin) / (vmax - vmin + 1e-6))[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    Image.fromarray(colored).save(path)


def create_visualization(
    he_input, generated_ihc, results, save_path, threshold=0.5
):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Helper functions
    def to_img(t):
        return np.clip((t.squeeze().cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)

    def to_map(t):
        return t.squeeze().cpu().detach().numpy()

    # Row 1: Images
    axes[0, 0].imshow(to_img(he_input))
    axes[0, 0].set_title('Input H&E', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(to_img(generated_ihc))
    axes[0, 1].set_title('Generated IHC', fontsize=12)
    axes[0, 1].axis('off')

    # Combined confidence
    combined_conf = to_map(results.get('confidence_combined', results.get('confidence_brown')))
    im = axes[0, 2].imshow(combined_conf, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Combined Confidence\nmean={combined_conf.mean():.3f}', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Masked output
    mask = combined_conf >= threshold
    output_np = to_img(generated_ihc)
    masked = output_np.copy()
    masked[~mask] = masked[~mask] * 0.5 + np.array([1, 0, 0]) * 0.5
    axes[0, 3].imshow(np.clip(masked, 0, 1))
    coverage = mask.mean()
    axes[0, 3].set_title(f'Low-conf highlighted\nCoverage={coverage:.1%}', fontsize=12)
    axes[0, 3].axis('off')

    # Row 2: Individual confidence maps
    if 'expected_brown' in results:
        expected_brown = to_map(results['expected_brown'])
        im = axes[1, 0].imshow(expected_brown, cmap='YlOrBr', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Expected Brown\nmean={expected_brown.mean():.3f}', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    else:
        axes[1, 0].axis('off')

    if 'actual_brown' in results:
        actual_brown = to_map(results['actual_brown'])
        im = axes[1, 1].imshow(actual_brown, cmap='YlOrBr', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Actual Brown (in generated)\nmean={actual_brown.mean():.3f}', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    else:
        axes[1, 1].axis('off')

    if 'brown_deviation' in results:
        deviation = to_map(results['brown_deviation'])
        im = axes[1, 2].imshow(deviation, cmap='hot', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Brown Deviation\nmean={deviation.mean():.3f}', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 2].axis('off')

    if 'confidence_brown' in results:
        brown_conf = to_map(results['confidence_brown'])
        im = axes[1, 3].imshow(brown_conf, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 3].set_title(f'Brown Predictor Confidence\nmean={brown_conf.mean():.3f}', fontsize=12)
        axes[1, 3].axis('off')
        plt.colorbar(im, ax=axes[1, 3], fraction=0.046)
    else:
        axes[1, 3].axis('off')

    # Row 3: Other confidence maps and statistics
    if 'confidence_hallucination' in results:
        halluc_conf = to_map(results['confidence_hallucination'])
        im = axes[2, 0].imshow(halluc_conf, cmap='RdYlGn', vmin=0, vmax=1)
        axes[2, 0].set_title(f'Hallucination Detector\nmean={halluc_conf.mean():.3f}', fontsize=12)
        axes[2, 0].axis('off')
        plt.colorbar(im, ax=axes[2, 0], fraction=0.046)
    else:
        axes[2, 0].axis('off')

    if 'confidence_mc_dropout' in results:
        mc_conf = to_map(results['confidence_mc_dropout'])
        im = axes[2, 1].imshow(mc_conf, cmap='RdYlGn', vmin=0, vmax=1)
        axes[2, 1].set_title(f'MC Dropout Confidence\nmean={mc_conf.mean():.3f}', fontsize=12)
        axes[2, 1].axis('off')
        plt.colorbar(im, ax=axes[2, 1], fraction=0.046)
    else:
        axes[2, 1].axis('off')

    # Histogram
    axes[2, 2].hist(combined_conf.flatten(), bins=50, alpha=0.7, color='steelblue')
    axes[2, 2].axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    axes[2, 2].set_xlabel('Confidence')
    axes[2, 2].set_ylabel('Pixel Count')
    axes[2, 2].set_title('Confidence Distribution')
    axes[2, 2].legend()

    # Statistics text
    stats_text = f"""Statistics:

Combined Confidence:
  Mean: {combined_conf.mean():.4f}
  Std: {combined_conf.std():.4f}
  Min: {combined_conf.min():.4f}
  Max: {combined_conf.max():.4f}

Coverage (conf >= {threshold}): {coverage:.1%}

Key: Red regions indicate
where the model might be
hallucinating (low confidence)
"""
    axes[2, 3].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[2, 3].transAxes)
    axes[2, 3].set_title('Summary')
    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class UnpairedConfidenceInference:
    """Inference with unpaired confidence estimation."""

    def __init__(self, generator_opt, confidence_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load generator
        print("Loading generator...")
        self.generator_model = create_model(generator_opt)
        self.generator_model.setup(generator_opt)
        self.generator_model.eval()

        # Load config
        config_path = os.path.join(confidence_dir, 'config.pth')
        if os.path.exists(config_path):
            config = torch.load(config_path, map_location=self.device)
            predictor_ngf = config.get('predictor_ngf', 64)
            detector_ndf = config.get('detector_ndf', 64)
        else:
            predictor_ngf = 64
            detector_ndf = 64

        # Load brown predictor
        brown_path = os.path.join(confidence_dir, 'latest_brown_predictor.pth')
        self.brown_predictor = None
        if os.path.exists(brown_path):
            print(f"Loading brown predictor from {brown_path}...")
            self.brown_predictor = BrownIntensityPredictor(
                input_nc=3, ngf=predictor_ngf
            ).to(self.device)
            checkpoint = torch.load(brown_path, map_location=self.device)
            self.brown_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.brown_predictor.eval()
            print("Brown predictor loaded.")

        # Load hallucination detector
        detector_path = os.path.join(confidence_dir, 'latest_hallucination_detector.pth')
        self.hallucination_detector = None
        if os.path.exists(detector_path):
            print(f"Loading hallucination detector from {detector_path}...")
            self.hallucination_detector = HallucinationDetector(
                input_nc=3, ndf=detector_ndf
            ).to(self.device)
            checkpoint = torch.load(detector_path, map_location=self.device)
            self.hallucination_detector.load_state_dict(checkpoint['model_state_dict'])
            self.hallucination_detector.eval()
            print("Hallucination detector loaded.")

        # Create combined estimator
        self.estimator = UnpairedConfidenceEstimator(
            brown_predictor=self.brown_predictor,
            hallucination_detector=self.hallucination_detector,
            generator=self.generator_model.netG_A
        )

        self.color_deconv = ColorDeconvolution()

    @torch.no_grad()
    def process(self, he_input, use_mc_dropout=True, mc_samples=10):
        """Process single image with confidence estimation."""
        he_input = he_input.to(self.device)

        # Generate IHC
        generated_ihc = self.generator_model.netG_A(he_input, layers=[])

        # Compute confidence
        results = self.estimator.compute_confidence(
            he_input, generated_ihc,
            use_mc_dropout=use_mc_dropout,
            mc_samples=mc_samples
        )

        return results


def main():
    parser = argparse.ArgumentParser(description='Unpaired Confidence Inference')

    # Data
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--phase', type=str, default='val')

    # Models
    parser.add_argument('--generator_name', type=str, required=True)
    parser.add_argument('--generator_epoch', type=str, default='latest')
    parser.add_argument('--confidence_name', type=str, required=True)

    # Output
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    # Options
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--use_mc_dropout', action='store_true', default=True)
    parser.add_argument('--mc_samples', type=int, default=10)
    parser.add_argument('--gpu_ids', type=str, default='0')

    # Dataset options
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)

    args = parser.parse_args()
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x]

    # Create generator options
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
    gen_opt.no_dropout = True  # Set False to enable MC Dropout variance estimation
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
    gen_opt.nce_layers = '0,4,8,12,16'
    gen_opt.netF = 'mlp_sample'
    gen_opt.netF_nc = 256

    # Create inference engine
    confidence_dir = os.path.join(args.checkpoints_dir, args.confidence_name)
    inferencer = UnpairedConfidenceInference(gen_opt, confidence_dir)

    # Create dataset
    dataset = create_dataset(gen_opt)
    print(f"Dataset size: {len(dataset)}")

    # Create output directory
    output_dir = os.path.join(
        args.results_dir,
        f"{args.generator_name}_unpaired_confidence",
        f"{args.phase}_{args.generator_epoch}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'confidence_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"MC Dropout: {args.use_mc_dropout}, samples: {args.mc_samples}")

    # Process images
    all_coverages = []
    all_confidences = []
    all_deviations = []

    for i, data in enumerate(tqdm(dataset)):
        # Get image name
        if 'A_paths' in data:
            img_name = os.path.splitext(os.path.basename(data['A_paths'][0]))[0]
        else:
            img_name = f"image_{i:05d}"

        he_input = data['A']

        # Process
        results = inferencer.process(
            he_input,
            use_mc_dropout=args.use_mc_dropout,
            mc_samples=args.mc_samples
        )

        generated_ihc = results['generated_ihc']
        confidence = results['confidence_combined']

        # Statistics
        conf_np = confidence.squeeze().cpu().detach().numpy()
        coverage = (conf_np >= args.confidence_threshold).mean()
        mean_conf = conf_np.mean()

        all_coverages.append(coverage)
        all_confidences.append(mean_conf)

        if 'brown_deviation' in results:
            dev = results['brown_deviation'].mean().item()
            all_deviations.append(dev)

        # Save outputs
        save_image(generated_ihc, os.path.join(output_dir, 'outputs', f'{img_name}.png'))
        save_heatmap(confidence, os.path.join(output_dir, 'confidence_maps', f'{img_name}_conf.png'))

        # Create overlay
        output_np = (generated_ihc.squeeze().cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2
        mask = conf_np >= args.confidence_threshold
        overlay = output_np.copy()
        overlay[~mask] = overlay[~mask] * 0.5 + np.array([1, 0, 0]) * 0.5
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(output_dir, 'overlays', f'{img_name}_overlay.png'))

        # Create visualization
        create_visualization(
            he_input, generated_ihc, results,
            os.path.join(output_dir, 'visualizations', f'{img_name}_viz.png'),
            threshold=args.confidence_threshold
        )

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY (Unpaired Confidence)")
    print("=" * 60)
    print(f"Total images: {len(dataset)}")
    print(f"Mean coverage (conf >= {args.confidence_threshold}): {np.mean(all_coverages):.1%}")
    print(f"Mean confidence: {np.mean(all_confidences):.3f}")

    if all_deviations:
        print(f"Mean brown deviation: {np.mean(all_deviations):.4f}")

    print(f"\nResults saved to: {output_dir}")

    # Interpretation
    mean_coverage = np.mean(all_coverages)
    if mean_coverage > 0.9:
        print("\n✓ High coverage - model is generally confident")
    elif mean_coverage > 0.7:
        print("\n📊 Moderate coverage - some uncertain regions detected")
    else:
        print("\n⚠️  Low coverage - many uncertain regions detected")
        print("   Check the visualizations to see where hallucinations might occur")

    # Save summary
    summary = {
        'generator_name': args.generator_name,
        'confidence_name': args.confidence_name,
        'num_images': len(dataset),
        'threshold': args.confidence_threshold,
        'use_mc_dropout': args.use_mc_dropout,
        'mc_samples': args.mc_samples,
        'mean_coverage': float(np.mean(all_coverages)),
        'mean_confidence': float(np.mean(all_confidences)),
        'coverages': [float(c) for c in all_coverages],
        'confidences': [float(c) for c in all_confidences]
    }

    if all_deviations:
        summary['mean_deviation'] = float(np.mean(all_deviations))
        summary['deviations'] = [float(d) for d in all_deviations]

    with open(os.path.join(output_dir, 'inference_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
