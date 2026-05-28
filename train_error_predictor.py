"""
Training Script for Learned Error Predictor

This script trains an error predictor network that learns to predict
WHERE the generator will make errors, using actual GT error as supervision.

The key insight:
- During training, we have GT → we can compute actual errors
- ErrorPredictor learns: given (H&E, generated_IHC) → predict error map
- At test time: ErrorPredictor predicts error without needing GT

Usage:
    # First, train your main generator (G: H&E → IHC)
    python train_confidence.py --dataroot ./datasets/... --name confidence_model

    # Then, train the error predictor using the frozen generator
    python train_error_predictor.py \
        --dataroot ./datasets/MIST/HER2/TrainValAB \
        --generator_name confidence_her2 \
        --generator_epoch latest \
        --name error_predictor_her2 \
        --n_epochs 50

The trained error predictor can then be used for confidence estimation.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

from options.base_options import BaseOptions
from data import create_dataset
from models import create_model
from models.error_predictor import (
    ErrorPredictorNetwork,
    ErrorPredictorLight,
    DualInputErrorPredictor,
    define_error_predictor
)
import util.util as util


class ErrorPredictorOptions(BaseOptions):
    """Options for error predictor training."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # Generator checkpoint (frozen, used to generate fake IHC)
        parser.add_argument('--generator_name', type=str, required=True,
                            help='Name of the trained generator checkpoint')
        parser.add_argument('--generator_epoch', type=str, default='latest',
                            help='Which epoch of the generator to load')

        # Error predictor architecture
        parser.add_argument('--predictor_type', type=str, default='standard',
                            choices=['standard', 'light', 'dual'],
                            help='Type of error predictor network')
        parser.add_argument('--predictor_ngf', type=int, default=64,
                            help='Number of filters in error predictor')

        # Training
        parser.add_argument('--n_epochs', type=int, default=50,
                            help='Number of epochs to train')
        parser.add_argument('--lr', type=float, default=0.0002,
                            help='Learning rate')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='Adam beta1')
        parser.add_argument('--beta2', type=float, default=0.999,
                            help='Adam beta2')

        # Loss weights
        parser.add_argument('--lambda_l1', type=float, default=1.0,
                            help='Weight for L1 loss')
        parser.add_argument('--lambda_l2', type=float, default=0.0,
                            help='Weight for L2 loss')
        parser.add_argument('--lambda_ssim', type=float, default=0.0,
                            help='Weight for SSIM loss')
        parser.add_argument('--lambda_focal', type=float, default=1.0,
                            help='Weight for focal loss (emphasize high-error regions)')

        # Logging
        parser.add_argument('--print_freq', type=int, default=100,
                            help='Frequency of printing training progress')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='Frequency of saving checkpoints')
        parser.add_argument('--display_freq', type=int, default=400,
                            help='Frequency of displaying images')

        # Model defaults
        parser.set_defaults(
            model='confidence',
            dataset_mode='aligned',
            batch_size=4,
            load_size=256,
            crop_size=256,
            preprocess='resize_and_crop',
            no_flip=False
        )

        return parser

    def parse(self):
        opt = super().parse()
        opt.isTrain = True
        return opt


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss - emphasizes high-error regions.

    Standard MSE treats all pixels equally, but we want the predictor
    to focus more on regions with high error (the hallucinated brown regions).
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted error [B, 1, H, W]
            target: Actual error [B, 1, H, W]

        Returns:
            Weighted loss focusing on high-error regions
        """
        # Basic MSE per pixel
        mse = (pred - target) ** 2

        # Weight by target error (high error = high weight)
        # This makes the model focus more on predicting high-error regions correctly
        weight = (target ** self.gamma) + 0.1  # +0.1 to avoid zero weight

        # Normalize weights
        weight = weight / weight.mean()

        return (mse * weight).mean()


class ErrorPredictorTrainer:
    """Trainer class for the error predictor."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the frozen generator
        self._load_generator()

        # Create error predictor
        self.error_predictor = define_error_predictor(
            predictor_type=opt.predictor_type,
            input_nc=opt.input_nc,
            ngf=opt.predictor_ngf,
            gpu_ids=opt.gpu_ids
        )

        # Losses
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 = nn.MSELoss()
        self.criterion_focal = FocalMSELoss(gamma=2.0)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.error_predictor.parameters(),
            lr=opt.lr,
            betas=(opt.beta1, opt.beta2)
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

        # Checkpoint directory
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    def _load_generator(self):
        """Load the pre-trained frozen generator."""
        print(f"Loading generator from {self.opt.generator_name}...")

        # Create a temporary opt for loading the generator
        gen_opt = type(self.opt)()
        gen_opt.__dict__.update(self.opt.__dict__)
        gen_opt.name = self.opt.generator_name
        gen_opt.epoch = self.opt.generator_epoch
        gen_opt.isTrain = False
        gen_opt.model = 'confidence'

        # Create and setup the generator model
        self.generator_model = create_model(gen_opt)
        self.generator_model.setup(gen_opt)
        self.generator_model.eval()

        # Freeze generator
        for param in self.generator_model.netG_A.parameters():
            param.requires_grad = False

        print("Generator loaded and frozen.")

    def compute_actual_error(self, fake_ihc, real_ihc):
        """
        Compute the actual pixel-wise error between generated and real IHC.

        Args:
            fake_ihc: Generated IHC [B, 3, H, W]
            real_ihc: Real IHC [B, 3, H, W]

        Returns:
            error_map: Per-pixel error [B, 1, H, W] normalized to [0, 1]
        """
        # L1 error across channels
        error = torch.abs(fake_ihc - real_ihc).mean(dim=1, keepdim=True)

        # Normalize to [0, 1] range using percentile
        B = error.shape[0]
        error_normalized = torch.zeros_like(error)

        for b in range(B):
            e = error[b]
            p5 = torch.quantile(e, 0.05)
            p95 = torch.quantile(e, 0.95)
            e_norm = (e - p5) / (p95 - p5 + 1e-6)
            error_normalized[b] = e_norm.clamp(0, 1)

        return error_normalized

    def train_step(self, data):
        """Single training step."""
        # Get inputs
        real_he = data['A'].to(self.device)  # H&E input
        real_ihc = data['B'].to(self.device)  # Real IHC (GT)

        # Generate fake IHC using frozen generator
        with torch.no_grad():
            fake_ihc = self.generator_model.netG_A(real_he, layers=[])

        # Compute actual error (this is our supervision signal)
        actual_error = self.compute_actual_error(fake_ihc, real_ihc)

        # Predict error using error predictor
        predicted_error = self.error_predictor(real_he, fake_ihc)

        # Compute losses
        loss_dict = {}

        if self.opt.lambda_l1 > 0:
            loss_l1 = self.criterion_l1(predicted_error, actual_error) * self.opt.lambda_l1
            loss_dict['L1'] = loss_l1
        else:
            loss_l1 = 0

        if self.opt.lambda_l2 > 0:
            loss_l2 = self.criterion_l2(predicted_error, actual_error) * self.opt.lambda_l2
            loss_dict['L2'] = loss_l2
        else:
            loss_l2 = 0

        if self.opt.lambda_focal > 0:
            loss_focal = self.criterion_focal(predicted_error, actual_error) * self.opt.lambda_focal
            loss_dict['Focal'] = loss_focal
        else:
            loss_focal = 0

        total_loss = loss_l1 + loss_l2 + loss_focal
        loss_dict['Total'] = total_loss

        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return loss_dict, {
            'real_he': real_he,
            'real_ihc': real_ihc,
            'fake_ihc': fake_ihc,
            'actual_error': actual_error,
            'predicted_error': predicted_error
        }

    def validate(self, val_loader):
        """Validation loop."""
        self.error_predictor.eval()

        total_l1 = 0
        total_correlation = 0
        n_batches = 0

        with torch.no_grad():
            for data in val_loader:
                real_he = data['A'].to(self.device)
                real_ihc = data['B'].to(self.device)

                # Generate fake IHC
                fake_ihc = self.generator_model.netG_A(real_he, layers=[])

                # Compute actual and predicted error
                actual_error = self.compute_actual_error(fake_ihc, real_ihc)
                predicted_error = self.error_predictor(real_he, fake_ihc)

                # L1 error
                total_l1 += torch.abs(predicted_error - actual_error).mean().item()

                # Correlation
                pred_flat = predicted_error.flatten().cpu().numpy()
                actual_flat = actual_error.flatten().cpu().numpy()
                corr = np.corrcoef(pred_flat, actual_flat)[0, 1]
                if not np.isnan(corr):
                    total_correlation += corr

                n_batches += 1

        self.error_predictor.train()

        return {
            'val_l1': total_l1 / n_batches,
            'val_correlation': total_correlation / n_batches
        }

    def save(self, epoch):
        """Save checkpoint."""
        save_path = os.path.join(self.save_dir, f'{epoch}_error_predictor.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.error_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'opt': self.opt
        }, save_path)
        print(f"Saved checkpoint: {save_path}")

        # Also save as latest
        latest_path = os.path.join(self.save_dir, 'latest_error_predictor.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.error_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'opt': self.opt
        }, latest_path)

    def load(self, epoch='latest'):
        """Load checkpoint."""
        if epoch == 'latest':
            load_path = os.path.join(self.save_dir, 'latest_error_predictor.pth')
        else:
            load_path = os.path.join(self.save_dir, f'{epoch}_error_predictor.pth')

        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.error_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from {load_path}")
            return checkpoint['epoch']
        return 0


def save_visualization(visuals, save_dir, epoch, batch_idx):
    """Save visualization images."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert tensors to numpy
    def to_np(t):
        return (t[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2

    def to_np_gray(t):
        return t[0, 0].cpu().numpy()

    # Row 1: Images
    axes[0, 0].imshow(np.clip(to_np(visuals['real_he']), 0, 1))
    axes[0, 0].set_title('Input H&E')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.clip(to_np(visuals['fake_ihc']), 0, 1))
    axes[0, 1].set_title('Generated IHC')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.clip(to_np(visuals['real_ihc']), 0, 1))
    axes[0, 2].set_title('Real IHC (GT)')
    axes[0, 2].axis('off')

    # Row 2: Error maps
    actual_err = to_np_gray(visuals['actual_error'])
    pred_err = to_np_gray(visuals['predicted_error'])

    im1 = axes[1, 0].imshow(actual_err, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Actual Error (mean={actual_err.mean():.3f})')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(pred_err, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Predicted Error (mean={pred_err.mean():.3f})')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    # Scatter plot
    actual_flat = actual_err.flatten()[::100]
    pred_flat = pred_err.flatten()[::100]
    axes[1, 2].scatter(actual_flat, pred_flat, alpha=0.3, s=1)
    axes[1, 2].plot([0, 1], [0, 1], 'r--')
    corr = np.corrcoef(actual_flat, pred_flat)[0, 1]
    axes[1, 2].set_title(f'Correlation: {corr:.3f}')
    axes[1, 2].set_xlabel('Actual Error')
    axes[1, 2].set_ylabel('Predicted Error')
    axes[1, 2].set_xlim([0, 1])
    axes[1, 2].set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'epoch{epoch}_batch{batch_idx}.png'), dpi=100)
    plt.close()


def train(opt):
    """Main training function."""
    # Create dataset
    dataset = create_dataset(opt)
    print(f"Dataset size: {len(dataset)}")

    # Create trainer
    trainer = ErrorPredictorTrainer(opt)

    # Training loop
    total_iters = 0
    vis_dir = os.path.join(opt.checkpoints_dir, opt.name, 'visualizations')

    print("\n" + "=" * 60)
    print("Starting Error Predictor Training")
    print("=" * 60)
    print(f"Predictor type: {opt.predictor_type}")
    print(f"Learning rate: {opt.lr}")
    print(f"Epochs: {opt.n_epochs}")
    print("=" * 60 + "\n")

    for epoch in range(1, opt.n_epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        n_batches = 0

        for i, data in enumerate(tqdm(dataset, desc=f"Epoch {epoch}")):
            total_iters += 1

            # Training step
            losses, visuals = trainer.train_step(data)
            epoch_loss += losses['Total'].item()
            n_batches += 1

            # Print progress
            if total_iters % opt.print_freq == 0:
                loss_str = ', '.join([f'{k}: {v.item():.4f}' for k, v in losses.items()])
                print(f"[Epoch {epoch}, Iter {total_iters}] {loss_str}")

            # Save visualization
            if total_iters % opt.display_freq == 0:
                save_visualization(visuals, vis_dir, epoch, i)

        # Epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / n_batches
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s, Avg Loss: {avg_loss:.4f}")

        # Update scheduler
        trainer.scheduler.step()

        # Save checkpoint
        if epoch % opt.save_epoch_freq == 0:
            trainer.save(epoch)

    # Save final model
    trainer.save('latest')
    print("\nTraining completed!")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
Your error predictor is trained and saved to:
  {os.path.join(opt.checkpoints_dir, opt.name)}

To use it for inference with learned confidence:
  python inference_with_error_predictor.py \\
      --dataroot ./datasets/MIST/HER2/TrainValAB \\
      --generator_name {opt.generator_name} \\
      --predictor_name {opt.name}
""")


if __name__ == '__main__':
    opt = ErrorPredictorOptions().parse()
    train(opt)
