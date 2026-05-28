"""
Visualization: Cycle Consistency vs Stain Predictor Confidence

This script generates the key figure for the paper, demonstrating that:
- Cycle consistency cannot detect "plausible but wrong" generations (one-to-many blind spot)
- Stain Intensity Predictor can detect these cases by comparing expected vs actual stain

Layout (1 row, 7 columns):
  [H&E Input] [Generated IHC] [H&E Recon] [Cycle Error] [Expected Stain] [Stain Deviation] [Reference IHC]

Usage:
    python visualize_cycle_vs_stain.py \
        --dataroot ../MIST/HER2/TrainValAB \
        --generator_name confidence_her2 \
        --confidence_name unpaired_confidence_her2 \
        --num_images 20
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm

from data import create_dataset
from models import create_model
from models.unpaired_confidence import (
    BrownIntensityPredictor,
    ColorDeconvolution,
)


def tensor_to_image(tensor):
    """Convert [-1, 1] tensor to [0, 1] numpy image (H, W, 3)."""
    return np.clip(
        (tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2,
        0, 1,
    )


def tensor_to_map(tensor):
    """Convert single-channel tensor to (H, W) numpy array."""
    return tensor.squeeze().cpu().detach().numpy()


def compute_cycle_error(he_orig, he_recon):
    """Pixel-wise L1 cycle reconstruction error, averaged over channels."""
    error = torch.abs(he_orig - he_recon).mean(dim=1, keepdim=True)  # [B,1,H,W]
    return error


def cycle_error_to_confidence(error):
    """Convert cycle error to confidence via sigmoid."""
    # Normalize: low error → high confidence
    confidence = 1.0 - torch.sigmoid(error * 5.0 - 2.5)
    return confidence


def create_comparison_figure(
    he_input,
    generated_ihc,
    he_recon,
    cycle_error_map,
    cycle_conf_map,
    expected_stain,
    stain_deviation,
    stain_conf_map,
    reference_ihc,
    save_path,
    img_name="",
):
    """Create the main paper figure comparing cycle vs stain predictor."""

    fig = plt.figure(figsize=(28, 8))
    gs = gridspec.GridSpec(2, 7, height_ratios=[3, 1], hspace=0.3, wspace=0.15)

    # --- Top row: main images ---

    # 0. Input H&E
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tensor_to_image(he_input))
    ax.set_title("Input H&E", fontsize=11, fontweight="bold")
    ax.axis("off")

    # 0-1. Generated IHC
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(tensor_to_image(generated_ihc))
    ax.set_title("Generated IHC", fontsize=11, fontweight="bold")
    ax.axis("off")

    # 1a. H&E Reconstructed (cycle)
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(tensor_to_image(he_recon))
    ax.set_title("H&E Recon (cycle)", fontsize=11, fontweight="bold")
    ax.axis("off")

    # 1b. Cycle Error Map
    ax = fig.add_subplot(gs[0, 3])
    ce = tensor_to_map(cycle_error_map)
    im = ax.imshow(ce, cmap="hot", vmin=0, vmax=ce.max() + 1e-6)
    mean_ce = ce.mean()
    ax.set_title(f"Cycle Error\nmean={mean_ce:.4f}", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2a. Expected Stain (from H&E)
    ax = fig.add_subplot(gs[0, 4])
    es = tensor_to_map(expected_stain)
    im = ax.imshow(es, cmap="YlOrBr", vmin=0, vmax=max(es.max(), 0.2))
    ax.set_title(
        f"Expected Stain\n(from H&E, mean={es.mean():.3f})",
        fontsize=11,
        fontweight="bold",
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2b. Stain Deviation
    ax = fig.add_subplot(gs[0, 5])
    sd = tensor_to_map(stain_deviation)
    im = ax.imshow(sd, cmap="hot", vmin=0, vmax=max(sd.max(), 0.2))
    ax.set_title(
        f"Stain Deviation\nmean={sd.mean():.4f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3. Reference IHC (real)
    ax = fig.add_subplot(gs[0, 6])
    ax.imshow(tensor_to_image(reference_ihc))
    ax.set_title("Reference IHC\n(real, serial section)", fontsize=11, fontweight="bold")
    ax.axis("off")

    # --- Bottom row: confidence comparison ---

    # Cycle confidence
    ax = fig.add_subplot(gs[1, 2:4])
    cc = tensor_to_map(cycle_conf_map)
    im = ax.imshow(cc, cmap="RdYlGn", vmin=0, vmax=1)
    coverage_cycle = (cc >= 0.5).mean() * 100
    ax.set_title(
        f"Cycle Confidence  (mean={cc.mean():.3f}, coverage@0.5={coverage_cycle:.1f}%)",
        fontsize=10,
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Stain confidence
    ax = fig.add_subplot(gs[1, 4:6])
    sc = tensor_to_map(stain_conf_map)
    im = ax.imshow(sc, cmap="RdYlGn", vmin=0, vmax=1)
    coverage_stain = (sc >= 0.5).mean() * 100
    ax.set_title(
        f"Stain Confidence  (mean={sc.mean():.3f}, coverage@0.5={coverage_stain:.1f}%)",
        fontsize=10,
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotation: key message
    ax = fig.add_subplot(gs[1, 0:2])
    ax.axis("off")
    msg = (
        "Cycle confidence can miss\n"
        '"plausible but wrong" regions\n'
        "(one-to-many blind spot).\n\n"
        "Stain predictor catches them\n"
        "by comparing expected vs\n"
        "actual chromogen distribution."
    )
    ax.text(
        0.5, 0.5, msg, transform=ax.transAxes,
        fontsize=10, va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    # Summary stats
    ax = fig.add_subplot(gs[1, 6])
    ax.axis("off")
    summary = (
        f"Cycle conf: {cc.mean():.3f}\n"
        f"Stain conf: {sc.mean():.3f}\n"
        f"Cycle err:  {mean_ce:.4f}\n"
        f"Stain dev:  {sd.mean():.4f}\n"
        f"\nCycle cov:  {coverage_cycle:.1f}%\n"
        f"Stain cov:  {coverage_stain:.1f}%"
    )
    ax.text(
        0.5, 0.5, summary, transform=ax.transAxes,
        fontsize=9, va="center", ha="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
    )

    fig.suptitle(
        f"Cycle Consistency vs Stain Predictor Confidence — {img_name}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Cycle vs Stain Predictor Confidence"
    )

    # Data
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--phase", type=str, default="val")

    # Models
    parser.add_argument("--generator_name", type=str, required=True,
                        help="Name of the bidirectional generator checkpoint (has G_A and G_B)")
    parser.add_argument("--generator_epoch", type=str, default="latest")
    parser.add_argument("--confidence_name", type=str, required=True,
                        help="Name of the unpaired confidence checkpoint (has brown predictor)")

    # Output
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")

    # Options
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)

    args = parser.parse_args()
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x]
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_ids else "cpu")

    # ------------------------------------------------------------------
    # 1. Load bidirectional generator (G_A: H&E→IHC, G_B: IHC→H&E)
    # ------------------------------------------------------------------
    print("Loading bidirectional generator...")

    class GenOpt:
        pass

    gen_opt = GenOpt()
    gen_opt.dataroot = args.dataroot
    gen_opt.name = args.generator_name
    gen_opt.epoch = args.generator_epoch
    gen_opt.checkpoints_dir = args.checkpoints_dir
    gen_opt.model = "confidence"
    gen_opt.isTrain = False
    gen_opt.gpu_ids = args.gpu_ids
    gen_opt.input_nc = 3
    gen_opt.output_nc = 3
    gen_opt.ngf = 64
    gen_opt.netG = "resnet_9blocks"
    gen_opt.normG = "instance"
    gen_opt.no_dropout = True
    gen_opt.init_type = "normal"
    gen_opt.init_gain = 0.02
    gen_opt.no_antialias = False
    gen_opt.no_antialias_up = False
    gen_opt.weight_norm = "none"
    gen_opt.load_size = args.load_size
    gen_opt.crop_size = args.crop_size
    gen_opt.preprocess = "resize_and_crop"
    gen_opt.no_flip = True
    gen_opt.direction = "AtoB"
    gen_opt.dataset_mode = "aligned"
    gen_opt.serial_batches = True
    gen_opt.num_threads = 0
    gen_opt.batch_size = 1
    gen_opt.phase = args.phase
    gen_opt.max_dataset_size = float("inf")
    gen_opt.verbose = False
    gen_opt.confidence_mode = "cycle_l1"
    gen_opt.load_discriminator = False

    model = create_model(gen_opt)
    model.setup(gen_opt)
    model.eval()

    netG_A = model.netG_A  # H&E → IHC
    netG_B = model.netG_B  # IHC → H&E

    # ------------------------------------------------------------------
    # 2. Load Stain Intensity Predictor
    # ------------------------------------------------------------------
    print("Loading stain intensity predictor...")

    confidence_dir = os.path.join(args.checkpoints_dir, args.confidence_name)
    config_path = os.path.join(confidence_dir, "config.pth")
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location=device)
        predictor_ngf = config.get("predictor_ngf", 64)
    else:
        predictor_ngf = 64

    stain_predictor = BrownIntensityPredictor(input_nc=3, ngf=predictor_ngf).to(device)
    brown_path = os.path.join(confidence_dir, "latest_brown_predictor.pth")
    if os.path.exists(brown_path):
        checkpoint = torch.load(brown_path, map_location=device)
        stain_predictor.load_state_dict(checkpoint["model_state_dict"])
        stain_predictor.eval()
        print(f"  Loaded from {brown_path}")
    else:
        print(f"  WARNING: {brown_path} not found. Stain predictor will be random.")

    color_deconv = ColorDeconvolution()

    # ------------------------------------------------------------------
    # 3. Load dataset (aligned: has both H&E and reference IHC)
    # ------------------------------------------------------------------
    dataset = create_dataset(gen_opt)
    print(f"Dataset size: {len(dataset)}")

    # ------------------------------------------------------------------
    # 4. Output directory
    # ------------------------------------------------------------------
    output_dir = os.path.join(
        args.results_dir,
        f"{args.generator_name}_cycle_vs_stain",
        f"{args.phase}_{args.generator_epoch}",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # 5. Process images
    # ------------------------------------------------------------------
    num_images = min(args.num_images, len(dataset))

    stats_cycle = []
    stats_stain = []

    for i, data in enumerate(tqdm(dataset, total=num_images, desc="Generating")):
        if i >= num_images:
            break

        # Parse image name
        if "A_paths" in data:
            img_name = os.path.splitext(os.path.basename(data["A_paths"][0]))[0]
        else:
            img_name = f"image_{i:05d}"

        he_input = data["A"].to(device)          # H&E input
        reference_ihc = data["B"].to(device)      # Real IHC (serial section, not pixel-aligned)

        with torch.no_grad():
            # --- Forward: H&E → IHC ---
            generated_ihc = netG_A(he_input, layers=[])

            # --- Cycle: IHC → H&E ---
            he_recon = netG_B(generated_ihc, layers=[])

            # --- Cycle error & confidence ---
            cycle_error_map = compute_cycle_error(he_input, he_recon)
            cycle_conf_map = cycle_error_to_confidence(cycle_error_map)

            # --- Stain predictor ---
            expected_stain, uncertainty = stain_predictor(he_input)
            actual_stain = color_deconv.extract_brown_ratio(generated_ihc)
            stain_deviation = torch.abs(actual_stain - expected_stain)
            normalized_dev = stain_deviation / (uncertainty + 0.1)
            stain_conf_map = torch.exp(-normalized_dev)

        # Collect stats
        stats_cycle.append(tensor_to_map(cycle_conf_map).mean())
        stats_stain.append(tensor_to_map(stain_conf_map).mean())

        # Save figure
        save_path = os.path.join(output_dir, f"{img_name}.png")
        create_comparison_figure(
            he_input=he_input,
            generated_ihc=generated_ihc,
            he_recon=he_recon,
            cycle_error_map=cycle_error_map,
            cycle_conf_map=cycle_conf_map,
            expected_stain=expected_stain,
            stain_deviation=stain_deviation,
            stain_conf_map=stain_conf_map,
            reference_ihc=reference_ihc,
            save_path=save_path,
            img_name=img_name,
        )

    # ------------------------------------------------------------------
    # 6. Summary statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Summary over {num_images} images:")
    print(f"  Cycle confidence:  mean={np.mean(stats_cycle):.4f}, std={np.std(stats_cycle):.4f}")
    print(f"  Stain confidence:  mean={np.mean(stats_stain):.4f}, std={np.std(stats_stain):.4f}")
    print(f"{'='*50}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
