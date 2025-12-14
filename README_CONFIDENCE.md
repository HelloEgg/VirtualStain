# Confidence-Aware Virtual Staining Framework

This extension implements a **confidence-aware H&E→IHC virtual staining framework** that addresses the critical issue of hallucination in virtual staining. The core principle is: **"If we don't know, we say we don't know."**

## Overview

Virtual staining from H&E to IHC is inherently a one-to-many mapping problem. Some regions have deterministic H&E→IHC relationships (high confidence), while others have ambiguous mappings where multiple IHC patterns could correspond to the same H&E input (low confidence, hallucination-prone).

Our framework:
1. Trains bidirectional translation models (H&E↔IHC)
2. Uses cycle consistency error to estimate prediction confidence
3. Provides pixel-level confidence maps with synthesized IHC images
4. Enables selective prediction: abstain on low-confidence regions

## Key Components

### 1. Bidirectional Model (`models/confidence_model.py`)

```python
# Train bidirectional generators
netG_A: H&E → IHC (forward)
netG_B: IHC → H&E (backward)

# Confidence estimation via cycle consistency
IHC_hat = G_A(H&E)
H&E_rec = G_B(IHC_hat)
error = ||H&E_rec - H&E||  # Low error → High confidence
```

### 2. Confidence Estimator (`models/confidence_estimator.py`)

Multiple confidence estimation strategies:
- **cycle_l1**: L1 reconstruction error
- **cycle_l2**: L2 (RMSE) reconstruction error
- **cycle_ssim**: SSIM-based reconstruction quality
- **variance**: Prediction variance across multiple samples
- **worst_case**: Maximum error across samples

### 3. Evaluation Metrics (`evaluate_confidence.py`)

- **Risk-Coverage Curves**: Plot error vs. coverage at different thresholds
- **AURC (Area Under Risk-Coverage)**: Lower is better
- **ECE/MCE**: Expected/Maximum Calibration Error
- **Selective Prediction**: Quality metrics on high-confidence regions only

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_REPO/AdaptiveSupervisedPatchNCE.git
cd AdaptiveSupervisedPatchNCE

# Create conda environment
conda env create -f environment.yml
conda activate asp

# Install additional dependencies for confidence estimation
pip install lpips matplotlib seaborn
```

## Usage

### Training

```bash
# Train the confidence-aware bidirectional model
python train_confidence.py \
    --dataroot ./datasets/MIST/HER2/TrainValAB \
    --name confidence_her2 \
    --model confidence \
    --lambda_cycle 10.0 \
    --lambda_cycle_B 10.0 \
    --confidence_mode worst_case \
    --num_latent_samples 5 \
    --n_epochs 30 \
    --n_epochs_decay 10
```

**Key Training Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--lambda_cycle` | 10.0 | Weight for forward cycle loss (H&E→IHC→H&E) |
| `--lambda_cycle_B` | 10.0 | Weight for backward cycle loss (IHC→H&E→IHC) |
| `--confidence_mode` | cycle_l1 | Confidence estimation method |
| `--num_latent_samples` | 5 | Samples for variance/worst-case estimation |
| `--confidence_threshold` | 0.5 | Threshold for low-confidence masking |

### Inference with Confidence Maps

```bash
python inference_confidence.py \
    --dataroot ./datasets/test_images \
    --name confidence_her2 \
    --epoch latest \
    --confidence_threshold 0.5 \
    --num_latent_samples 5 \
    --save_confidence_overlay
```

**Outputs:**
- `outputs/`: Generated IHC images
- `confidence_maps/`: Pixel-level confidence maps
- `visualizations/`: Composite visualizations
- `overlays/`: Images with low-confidence regions highlighted

### Evaluation

```bash
python evaluate_confidence.py \
    --dataroot ./datasets/MIST/HER2/TrainValAB \
    --name confidence_her2 \
    --phase test \
    --epoch latest
```

**Metrics Reported:**
- PSNR, SSIM, LPIPS (full image and high-confidence regions)
- AURC, Excess-AURC (selective prediction quality)
- ECE, MCE (confidence calibration)
- Coverage at different thresholds

## Framework Architecture

```
                    ┌─────────────┐
                    │   H&E Input │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   G_A       │ (H&E → IHC)
                    │ (Forward)   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ IHC_1   │  │ IHC_2   │  │ IHC_N   │  (Multiple samples)
        └────┬────┘  └────┬────┘  └────┬────┘
              │            │            │
              ▼            ▼            ▼
        ┌─────────────────────────────────┐
        │           G_B (Backward)        │ (IHC → H&E)
        └──────────────┬──────────────────┘
                       │
              ┌────────┼────────┐
              │        │        │
              ▼        ▼        ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ H&E_1'  │  │ H&E_2'  │  │ H&E_N'  │  (Reconstructions)
        └────┬────┘  └────┬────┘  └────┬────┘
              │            │            │
              ▼            ▼            ▼
        ┌─────────────────────────────────┐
        │    Reconstruction Errors        │
        │    e_i = ||H&E_i' - H&E||       │
        └──────────────┬──────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Confidence Map │
              │ (worst-case or │
              │  variance)     │
              └────────────────┘
```

## Confidence Interpretation

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| > 0.8 | High confidence: H&E↔IHC mapping is nearly 1:1 | Trust prediction |
| 0.5 - 0.8 | Medium confidence: Some ambiguity exists | Use with caution |
| < 0.5 | Low confidence: High hallucination risk | Abstain / Flag for review |

## Clinical Integration

The framework outputs enable clinical decision support:

1. **Triage**: Flag low-confidence regions for additional IHC testing
2. **Quality Control**: Identify problematic tissue regions automatically
3. **Interpretability**: Provide transparent uncertainty information

### Example Workflow

```python
from models import create_model
from models.confidence_estimator import PatchLevelConfidence

# Load model
model = create_model(opt)
model.setup(opt)

# Process image
result = model.generate_with_confidence(input_image, apply_mask=True)

# Get outputs
synthesized_ihc = result['output']
confidence_map = result['confidence_map']
mask = result['mask']  # Binary mask of high-confidence regions

# Patch-level analysis for clinical workflow
patch_conf = PatchLevelConfidence(patch_size=64, threshold=0.5)
patch_confidence, low_conf_patches = patch_conf(confidence_map)

# Flag low-confidence patches for pathologist review
for patch_idx in low_conf_patches.nonzero():
    flag_for_review(patch_idx)
```

## Experimental Results (Expected)

When properly trained, you should observe:

1. **Risk-Coverage Trade-off**: Higher confidence threshold → Lower risk but lower coverage
2. **Improved Metrics on High-Confidence Regions**: SSIM/PSNR increase when evaluating only high-confidence areas
3. **Calibrated Confidence**: ECE < 0.1 after calibration
4. **Meaningful Abstention**: Low-confidence regions correlate with high error regions

## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{confidence_virtual_staining,
  title={Confidence-Aware Virtual Staining: Knowing What We Don't Know},
  author={Your Name},
  booktitle={MICCAI},
  year={2025}
}
```

## Acknowledgments

This work builds upon:
- [Contrastive Learning for Unpaired Image-to-Image Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation)
- [Adaptive Supervised PatchNCE](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
