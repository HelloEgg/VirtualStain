"""
Inference Script with Learned Error Predictor

This script uses a trained error predictor to estimate confidence.
Unlike cycle consistency or discriminator-based methods, this directly
predicts WHERE errors will occur based on learned patterns.

The error predictor was trained with actual GT error as supervision,
so it learns:
1. Which H&E patterns are ambiguous (could map to brown or blue)
2. Which patterns the generator tends to hallucinate on
3. Which regions are out-of-distribution

Usage:
    python inference_with_error_predictor.py \
        --dataroot ./datasets/MIST/HER2/TrainValAB \
        --generator_name confidence_her2 \
        --predictor_name error_predictor_her2 \
        --results_dir ./results
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import json

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from models.error_predictor import define_error_predictor


def save_image(tensor, path):
    """Save tensor as image."""
    img = (tensor.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_heatmap(tensor, path, cmap='RdYlGn_r'):
    """Save tensor as heatmap."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    arr = tensor.squeeze().cpu().numpy()
    cmap_fn = cm.get_cmap(cmap)
    colored = cmap_fn(arr)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    Image.fromarray(colored).save(path)


def create_comparison_figure(
    input_he, generated_ihc, gt_ihc,
    predicted_error, actual_error,
    save_path
):
    """Create comprehensive comparison figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Helper functions
    def to_img(t):
        return np.clip((t.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)

    def to_map(t):
        return t.squeeze().cpu().numpy()

    # Row 1: Images
    axes[0, 0].imshow(to_img(input_he))
    axes[0, 0].set_title('Input H&E', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(to_img(generated_ihc))
    axes[0, 1].set_title('Generated IHC', fontsize=12)
    axes[0, 1].axis('off')

    if gt_ihc is not None:
        axes[0, 2].imshow(to_img(gt_ihc))
        axes[0, 2].set_title('Ground Truth IHC', fontsize=12)
        axes[0, 2].axis('off')

        # Difference
        diff = np.abs(to_img(generated_ihc) - to_img(gt_ihc)).mean(axis=2)
        im = axes[0, 3].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[0, 3].set_title('Actual Difference', fontsize=12)
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046)
    else:
        axes[0, 2].axis('off')
        axes[0, 3].axis('off')

    # Row 2: Error/Confidence maps
    pred_err = to_map(predicted_error)
    pred_conf = 1 - pred_err  # Convert error to confidence

    im1 = axes[1, 0].imshow(pred_err, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Predicted Error\n(mean={pred_err.mean():.3f})', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(pred_conf, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Predicted Confidence\n(mean={pred_conf.mean():.3f})', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    if actual_error is not None:
        actual_err = to_map(actual_error)

        im3 = axes[1, 2].imshow(actual_err, cmap='hot', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Actual Error (GT)\n(mean={actual_err.mean():.3f})', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

        # Correlation scatter
        pred_flat = pred_err.flatten()[::100]
        actual_flat = actual_err.flatten()[::100]
        axes[1, 3].scatter(actual_flat, pred_flat, alpha=0.3, s=1)
        axes[1, 3].plot([0, 1], [0, 1], 'r--', label='Perfect')
        corr = np.corrcoef(pred_flat, actual_flat)[0, 1]
        axes[1, 3].set_title(f'Correlation: {corr:.3f}', fontsize=12)
        axes[1, 3].set_xlabel('Actual Error')
        axes[1, 3].set_ylabel('Predicted Error')
        axes[1, 3].set_xlim([0, 1])
        axes[1, 3].set_ylim([0, 1])
        axes[1, 3].legend()
    else:
        axes[1, 2].axis('off')
        axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return corr if actual_error is not None else None


def create_overlay_visualization(generated_ihc, predicted_error, threshold=0.5):
    """
    Create visualization with low-confidence regions highlighted in red.

    This shows the pathologist: "These brown regions might be hallucinated"
    """
    # Convert to numpy
    img = (generated_ihc.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
    error = predicted_error.squeeze().cpu().numpy()

    # Create mask for high-error (low-confidence) regions
    high_error_mask = error > threshold

    # Create overlay
    overlay = img.copy()
    # Highlight high-error regions in semi-transparent red
    overlay[high_error_mask] = overlay[high_error_mask] * 0.5 + np.array([1, 0, 0]) * 0.5

    # Convert to uint8
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    return overlay


class ErrorPredictorInference:
    """Inference class using learned error predictor."""

    def __init__(self, generator_opt, predictor_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load generator
        print("Loading generator...")
        self.generator_model = create_model(generator_opt)
        self.generator_model.setup(generator_opt)
        self.generator_model.eval()

        # Load error predictor
        print(f"Loading error predictor from {predictor_path}...")
        checkpoint = torch.load(predictor_path, map_location=self.device)

        predictor_opt = checkpoint.get('opt', None)
        predictor_type = getattr(predictor_opt, 'predictor_type', 'standard') if predictor_opt else 'standard'
        ngf = getattr(predictor_opt, 'predictor_ngf', 64) if predictor_opt else 64

        self.error_predictor = define_error_predictor(
            predictor_type=predictor_type,
            input_nc=3,
            ngf=ngf,
            gpu_ids=generator_opt.gpu_ids
        )
        self.error_predictor.load_state_dict(checkpoint['model_state_dict'])
        self.error_predictor.eval()

        print("Models loaded successfully.")

    def process(self, input_he, gt_ihc=None):
        """
        Process single image and get confidence estimation.

        Args:
            input_he: H&E input tensor [1, 3, H, W]
            gt_ihc: Optional ground truth IHC for evaluation

        Returns:
            Dictionary with generated image, predicted error, confidence, etc.
        """
        input_he = input_he.to(self.device)

        with torch.no_grad():
            # Generate IHC
            generated_ihc = self.generator_model.netG_A(input_he, layers=[])

            # Predict error using learned predictor
            predicted_error = self.error_predictor(input_he, generated_ihc)

            # Convert error to confidence (low error = high confidence)
            predicted_confidence = 1 - predicted_error

        results = {
            'input_he': input_he,
            'generated_ihc': generated_ihc,
            'predicted_error': predicted_error,
            'predicted_confidence': predicted_confidence
        }

        # If GT available, compute actual error for comparison
        if gt_ihc is not None:
            gt_ihc = gt_ihc.to(self.device)
            actual_error = torch.abs(generated_ihc - gt_ihc).mean(dim=1, keepdim=True)

            # Normalize
            p5 = torch.quantile(actual_error, 0.05)
            p95 = torch.quantile(actual_error, 0.95)
            actual_error = ((actual_error - p5) / (p95 - p5 + 1e-6)).clamp(0, 1)

            results['gt_ihc'] = gt_ihc
            results['actual_error'] = actual_error

            # Compute correlation
            pred_flat = predicted_error.flatten().cpu().numpy()
            actual_flat = actual_error.flatten().cpu().numpy()
            correlation = np.corrcoef(pred_flat, actual_flat)[0, 1]
            results['correlation'] = correlation

        return results


def main():
    parser = argparse.ArgumentParser(description='Inference with Learned Error Predictor')

    # Data
    parser.add_argument('--dataroot', type=str, required=True, help='Path to dataset')
    parser.add_argument('--phase', type=str, default='val', help='train, val, test')

    # Models
    parser.add_argument('--generator_name', type=str, required=True,
                        help='Name of trained generator')
    parser.add_argument('--generator_epoch', type=str, default='latest',
                        help='Which epoch of generator to load')
    parser.add_argument('--predictor_name', type=str, required=True,
                        help='Name of trained error predictor')
    parser.add_argument('--predictor_epoch', type=str, default='latest',
                        help='Which epoch of predictor to load')

    # Output
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Checkpoints directory')

    # Options
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Threshold for low-confidence regions')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs')

    # Dataset options
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--preprocess', type=str, default='resize_and_crop')
    parser.add_argument('--no_flip', action='store_true')

    args = parser.parse_args()

    # Parse GPU IDs
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x]

    # Create generator options
    class GeneratorOpt:
        pass

    gen_opt = GeneratorOpt()
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
    gen_opt.no_dropout = True
    gen_opt.init_type = 'normal'
    gen_opt.init_gain = 0.02
    gen_opt.no_antialias = False
    gen_opt.no_antialias_up = False
    gen_opt.load_size = args.load_size
    gen_opt.crop_size = args.crop_size
    gen_opt.preprocess = args.preprocess
    gen_opt.no_flip = True
    gen_opt.direction = 'AtoB'
    gen_opt.dataset_mode = 'aligned'
    gen_opt.serial_batches = True
    gen_opt.num_threads = 0
    gen_opt.batch_size = 1
    gen_opt.phase = args.phase
    gen_opt.max_dataset_size = float('inf')
    gen_opt.verbose = False

    # Confidence model specific
    gen_opt.confidence_mode = 'cycle_l1'
    gen_opt.load_discriminator = False
    gen_opt.nce_layers = '0,4,8,12,16'
    gen_opt.netF = 'mlp_sample'
    gen_opt.netF_nc = 256

    # Load predictor path
    if args.predictor_epoch == 'latest':
        predictor_path = os.path.join(
            args.checkpoints_dir, args.predictor_name, 'latest_error_predictor.pth'
        )
    else:
        predictor_path = os.path.join(
            args.checkpoints_dir, args.predictor_name, f'{args.predictor_epoch}_error_predictor.pth'
        )

    # Create inference engine
    inferencer = ErrorPredictorInference(gen_opt, predictor_path)

    # Create dataset
    dataset = create_dataset(gen_opt)
    print(f"Dataset size: {len(dataset)}")

    # Create output directory
    output_dir = os.path.join(
        args.results_dir,
        f"{args.generator_name}_with_{args.predictor_name}",
        f"{args.phase}_{args.generator_epoch}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'confidence_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")

    # Process images
    all_correlations = []
    all_coverages = []
    all_mean_confidences = []

    for i, data in enumerate(tqdm(dataset)):
        # Get image name
        if 'A_paths' in data:
            img_name = os.path.splitext(os.path.basename(data['A_paths'][0]))[0]
        else:
            img_name = f"image_{i:05d}"

        # Get inputs
        input_he = data['A']
        gt_ihc = data.get('B', None)

        # Process
        results = inferencer.process(input_he, gt_ihc)

        # Save outputs
        save_image(results['generated_ihc'],
                   os.path.join(output_dir, 'outputs', f'{img_name}.png'))

        save_heatmap(results['predicted_confidence'],
                     os.path.join(output_dir, 'confidence_maps', f'{img_name}_conf.png'),
                     cmap='RdYlGn')

        save_heatmap(results['predicted_error'],
                     os.path.join(output_dir, 'confidence_maps', f'{img_name}_error.png'),
                     cmap='hot')

        # Create overlay
        overlay = create_overlay_visualization(
            results['generated_ihc'],
            results['predicted_error'],
            threshold=args.confidence_threshold
        )
        Image.fromarray(overlay).save(
            os.path.join(output_dir, 'overlays', f'{img_name}_overlay.png')
        )

        # Create comparison figure
        actual_error = results.get('actual_error', None)
        corr = create_comparison_figure(
            results['input_he'],
            results['generated_ihc'],
            results.get('gt_ihc', None),
            results['predicted_error'],
            actual_error,
            os.path.join(output_dir, 'comparisons', f'{img_name}_comparison.png')
        )

        # Statistics
        conf = results['predicted_confidence'].squeeze().cpu().numpy()
        coverage = (conf >= args.confidence_threshold).mean()
        mean_conf = conf.mean()

        all_coverages.append(coverage)
        all_mean_confidences.append(mean_conf)

        if corr is not None:
            all_correlations.append(corr)

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY (Learned Error Predictor)")
    print("=" * 60)
    print(f"Total images: {len(dataset)}")
    print(f"Mean coverage (conf >= {args.confidence_threshold}): {np.mean(all_coverages):.1%}")
    print(f"Mean confidence: {np.mean(all_mean_confidences):.3f}")

    if all_correlations:
        mean_corr = np.mean(all_correlations)
        print(f"\n[Correlation with GT Error]")
        print(f"Mean correlation: {mean_corr:.4f}")

        if mean_corr > 0.5:
            print("\n✓ Good correlation! The error predictor successfully learned")
            print("  to predict where the generator will make errors.")
        elif mean_corr > 0.3:
            print("\n📊 Moderate correlation. The predictor captures some error patterns.")
        else:
            print("\n⚠️  Low correlation. Consider:")
            print("   1. Training the error predictor longer")
            print("   2. Using a more expressive predictor architecture (--predictor_type dual)")
            print("   3. Checking if the generator's errors are predictable")

    # Save summary
    summary = {
        'generator_name': args.generator_name,
        'predictor_name': args.predictor_name,
        'num_images': len(dataset),
        'confidence_threshold': args.confidence_threshold,
        'mean_coverage': float(np.mean(all_coverages)),
        'mean_confidence': float(np.mean(all_mean_confidences)),
        'coverages': [float(c) for c in all_coverages],
        'mean_confidences': [float(c) for c in all_mean_confidences]
    }

    if all_correlations:
        summary['mean_correlation'] = float(np.mean(all_correlations))
        summary['correlations'] = [float(c) for c in all_correlations]

    with open(os.path.join(output_dir, 'inference_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
