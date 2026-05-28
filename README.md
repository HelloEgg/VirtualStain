# Confidence-Aware Virtual Staining

Code for the VirtualStain JBHI manuscript.

This repository contains the remaining manuscript-related code for H&E-to-IHC virtual staining with confidence estimation. The original AdaptiveSupervisedPatchNCE / PatchNCE / CUT training code has been removed.

## What Is Included

- Bidirectional H&E <-> IHC virtual staining
- Cycle-consistency confidence maps
- Stain-predictor and hallucination-detector confidence for weakly aligned or serial-section data
- Learned error-predictor confidence
- Risk-coverage, calibration, and confidence visualization scripts

## Environment

```bash
conda env create -f environment.yml
conda activate virtual_stain
```

If your environment is missing optional visualization packages:

```bash
pip install matplotlib seaborn tqdm dominate
```

## Dataset Layout

Set `--dataroot` to a folder with domain A as H&E and domain B as IHC.

For paired or approximately aligned data, use:

```text
datasets/YourDataset/
  trainA/   # H&E training patches
  trainB/   # IHC training patches
  valA/     # H&E validation patches
  valB/     # IHC validation patches
  testA/    # H&E test patches
  testB/    # IHC test patches, if available
```

For `--dataset_mode aligned`, each image in `trainA` must have the same relative filename in `trainB`. For example:

```text
trainA/case001_patch0001.png
trainB/case001_patch0001.png
```

The same rule applies to `valA`/`valB` and `testA`/`testB`. If `testA` is absent but `valA` exists, the loader falls back to `valA`/`valB` for test phase.

For serial-section or unpaired confidence experiments, the confidence-module training scripts can use:

```bash
--dataset_mode unaligned
```

In that mode, `trainA` contains H&E patches and `trainB` contains IHC patches from the same dataset distribution, but filenames do not need to match.

## Step 1: Train The Virtual Staining Generator

This trains the bidirectional H&E <-> IHC model and saves checkpoints under `checkpoints/confidence_her2`.

```bash
python train_confidence.py \
  --dataroot ./datasets/MIST/HER2/TrainValAB \
  --name confidence_her2 \
  --model confidence \
  --dataset_mode aligned \
  --direction AtoB \
  --netG resnet_6blocks \
  --netD n_layers \
  --load_size 1024 \
  --crop_size 512 \
  --preprocess crop \
  --lambda_cycle 10.0 \
  --lambda_cycle_B 10.0 \
  --lambda_gp 10.0 \
  --confidence_mode cycle_l1
```

For CPU-only smoke tests, add:

```bash
--gpu_ids -1 --load_size 256 --crop_size 256 --n_epochs 1 --n_epochs_decay 0
```

## Step 2: Run Basic Confidence Inference

This generates synthesized IHC images, confidence maps, overlays, and a summary under `results/`.

```bash
python inference_confidence.py \
  --dataroot ./datasets/MIST/HER2/TrainValAB \
  --name confidence_her2 \
  --model confidence \
  --phase test \
  --epoch latest \
  --confidence_mode cycle_l1 \
  --confidence_threshold 0.5 \
  --save_confidence_overlay
```

Useful confidence modes:

```text
cycle_l1
cycle_l2
variance
worst_case
mc_dropout
discriminator
ensemble
```

## Step 3: Train Optional Confidence Modules

Use this when H&E and IHC are weakly aligned or serial sections. It trains a brown-intensity predictor and hallucination detector using a frozen generator.

```bash
python train_unpaired_confidence.py \
  --dataroot ./datasets/MIST/HER2/TrainValAB \
  --generator_name confidence_her2 \
  --name unpaired_confidence_her2 \
  --dataset_mode unaligned \
  --n_epochs 50
```

Then run inference with those confidence modules:

```bash
python inference_unpaired_confidence.py \
  --dataroot ./datasets/MIST/HER2/TrainValAB \
  --generator_name confidence_her2 \
  --confidence_name unpaired_confidence_her2 \
  --phase test \
  --confidence_threshold 0.5
```

## Step 4: Train Optional Error Predictor

Use this when paired or approximately aligned ground truth IHC is available during training. The predictor learns where the generator tends to make errors.

```bash
python train_error_predictor.py \
  --dataroot ./datasets/MIST/HER2/TrainValAB \
  --generator_name confidence_her2 \
  --name error_predictor_her2 \
  --dataset_mode aligned \
  --n_epochs 50
```

Then run inference with learned error confidence:

```bash
python inference_with_error_predictor.py \
  --dataroot ./datasets/MIST/HER2/TrainValAB \
  --generator_name confidence_her2 \
  --predictor_name error_predictor_her2 \
  --phase test \
  --confidence_threshold 0.5
```

## Evaluation And Figures

Evaluate confidence quality and selective prediction metrics:

```bash
python evaluate_confidence.py \
  --dataroot ./datasets/MIST/HER2/TrainValAB \
  --name confidence_her2 \
  --model confidence \
  --phase test \
  --epoch latest
```

Compare cycle confidence against stain-predictor confidence:

```bash
python evaluate_cycle_vs_stain.py \
  --dataroot ./datasets/MIST/HER2/TrainValAB \
  --generator_name confidence_her2 \
  --confidence_name unpaired_confidence_her2 \
  --phase test
```

Generate manuscript-style visualizations:

```bash
python visualize_cycle_vs_stain.py \
  --dataroot ./datasets/MIST/HER2/TrainValAB \
  --generator_name confidence_her2 \
  --confidence_name unpaired_confidence_her2 \
  --phase test
```

## Outputs

Training checkpoints:

```text
checkpoints/<experiment_name>/
```

Inference outputs:

```text
results/<experiment_name>/
  outputs/
  confidence_maps/
  visualizations/
  overlays/
```

## Quick Sanity Check

Before launching a long training run:

```bash
python train_confidence.py --help
python inference_confidence.py --help
python evaluate_confidence.py --help
```

If those commands work and your dataset folders match the layout above, the repository is ready for experiments.
