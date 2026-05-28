# VirtualStain

Code for H&E <-> IHC virtual staining from the VirtualStain JBHI manuscript.

This repository keeps only the unpaired image-to-image generation pipeline described in the manuscript method.

## Method Components

- Cycle-consistent H&E <-> IHC generators with residual blocks, self-attention, and decoder skip connections
- Multi-scale spectral-normalized discriminators for both H&E and IHC domains
- HE_MIL_PPIE: multiple-instance H&E pathology information extractor using 9 random 96 x 96 crops
- Weakly supervised IHC classifier trained from DAB-intensity labels
- Pathology consistency loss between HE_MIL_PPIE predictions and generated-IHC classifier predictions
- Confusion discriminator that compares one generated IHC image against real IHC candidates
- Nuclei/topology preservation loss using stain-specific nuclei maps

## Environment

```powershell
conda env create -f environment.yml
conda activate virtual_stain
```

If optional visualization packages are missing:

```powershell
pip install dominate visdom tqdm pillow
```

## Dataset Layout

Use domain A for H&E and domain B for IHC. Training is unpaired, so filenames do not need to match.

```text
datasets/VirtualStain/
  trainA/   H&E training patches
  trainB/   IHC training patches
  valA/     H&E validation patches
  valB/     IHC validation patches
  testA/    H&E test patches
  testB/    IHC test patches, optional for single-domain inference
```

For inference with only H&E images, you can point `--dataroot` directly at a folder of H&E images and use `--dataset_mode single`.

## Train

```powershell
python .\train_virtual_stain.py `
  --dataroot .\datasets\VirtualStain `
  --name virtual_stain_he_ihc `
  --model virtual_stain `
  --dataset_mode unaligned `
  --load_size 286 `
  --crop_size 256 `
  --batch_size 1 `
  --display_id -1
```

Main manuscript loss weights are exposed as options:

```text
--lambda_GAN 1
--lambda_cycle 10
--lambda_identity 5
--lambda_patho 1
--lambda_topo 1
--lambda_confusion 1
```

Checkpoints are saved under:

```text
checkpoints/<experiment_name>/
```

## Inference

For a dataset with `testA/`:

```powershell
python .\inference_virtual_stain.py `
  --dataroot .\datasets\VirtualStain `
  --name virtual_stain_he_ihc `
  --model virtual_stain `
  --dataset_mode single `
  --phase test `
  --epoch latest `
  --num_test 100 `
  --eval
```

For a folder that directly contains H&E images:

```powershell
python .\inference_virtual_stain.py `
  --dataroot .\datasets\VirtualStain\testA `
  --name virtual_stain_he_ihc `
  --model virtual_stain `
  --dataset_mode single `
  --epoch latest `
  --num_test 100 `
  --eval
```

Outputs are saved under:

```text
results/<experiment_name>/<phase>_<epoch>/
  fake_IHC/
  reconstructed_HE/
```

## Quick Checks

```powershell
python .\train_virtual_stain.py --help
python .\inference_virtual_stain.py --help
```
