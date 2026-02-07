"""
Training Script for Unpaired Confidence Estimation

This script trains confidence estimation models that work WITHOUT pixel-aligned GT.
Perfect for histopathology where H&E and IHC come from serial sections.

Two models are trained:
1. Brown Intensity Predictor: Learns which H&E patterns produce brown staining
2. Hallucination Detector: Learns to distinguish real IHC from generated IHC

Usage:
    python train_unpaired_confidence.py \
        --dataroot ./datasets/MIST/HER2/TrainValAB \
        --generator_name confidence_her2 \
        --name unpaired_confidence_her2 \
        --n_epochs 50

Then use for inference:
    python inference_unpaired_confidence.py \
        --dataroot ./datasets/MIST/HER2/TrainValAB \
        --generator_name confidence_her2 \
        --confidence_name unpaired_confidence_her2
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from options.base_options import BaseOptions
from data import create_dataset
from models import create_model
from models.unpaired_confidence import (
    BrownIntensityPredictor,
    HallucinationDetector,
    ColorDeconvolution,
    train_brown_predictor,
    train_hallucination_detector
)


class UnpairedConfidenceOptions(BaseOptions):
    """Options for unpaired confidence training."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # Generator checkpoint
        parser.add_argument('--generator_name', type=str, required=True,
                            help='Name of the trained generator checkpoint')
        parser.add_argument('--generator_epoch', type=str, default='latest',
                            help='Which epoch of the generator to load')

        # Model architecture
        parser.add_argument('--predictor_ngf', type=int, default=64,
                            help='Number of filters in brown predictor')
        parser.add_argument('--detector_ndf', type=int, default=64,
                            help='Number of filters in hallucination detector')

        # Training
        parser.add_argument('--n_epochs', type=int, default=50,
                            help='Number of epochs to train each model')
        parser.add_argument('--lr', type=float, default=0.0002,
                            help='Learning rate')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='Adam beta1')
        parser.add_argument('--beta2', type=float, default=0.999,
                            help='Adam beta2')

        # What to train
        parser.add_argument('--train_brown_predictor', action='store_true', default=True,
                            help='Train the brown intensity predictor')
        parser.add_argument('--train_hallucination_detector', action='store_true', default=True,
                            help='Train the hallucination detector')

        # Logging
        parser.add_argument('--print_freq', type=int, default=100,
                            help='Frequency of printing progress')
        parser.add_argument('--save_freq', type=int, default=10,
                            help='Frequency of saving checkpoints (epochs)')

        parser.set_defaults(
            model='confidence',
            dataset_mode='aligned',
            batch_size=8,
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


class UnpairedConfidenceTrainer:
    """Trainer for unpaired confidence models."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Load frozen generator
        self._load_generator()

        # Create models
        self.brown_predictor = BrownIntensityPredictor(
            input_nc=opt.input_nc,
            ngf=opt.predictor_ngf
        ).to(self.device)

        self.hallucination_detector = HallucinationDetector(
            input_nc=opt.output_nc,
            ndf=opt.detector_ndf
        ).to(self.device)

        # Color deconvolution
        self.color_deconv = ColorDeconvolution()

    def _load_generator(self):
        """Load the pre-trained frozen generator."""
        print(f"Loading generator from {self.opt.generator_name}...")

        gen_opt = type(self.opt)()
        gen_opt.__dict__.update(self.opt.__dict__)
        gen_opt.name = self.opt.generator_name
        gen_opt.epoch = self.opt.generator_epoch
        gen_opt.isTrain = False
        gen_opt.model = 'confidence'

        self.generator_model = create_model(gen_opt)
        self.generator_model.setup(gen_opt)
        self.generator_model.eval()

        # Freeze generator
        for param in self.generator_model.netG_A.parameters():
            param.requires_grad = False

        print("Generator loaded and frozen.")

    def train_brown_predictor(self, dataloader):
        """Train the brown intensity predictor."""
        print("\n" + "=" * 60)
        print("Training Brown Intensity Predictor")
        print("=" * 60)
        print("This model learns: H&E pattern → expected brown intensity")
        print("No pixel alignment needed - uses patch-level statistics!")
        print("=" * 60 + "\n")

        optimizer = torch.optim.Adam(
            self.brown_predictor.parameters(),
            lr=self.opt.lr,
            betas=(self.opt.beta1, self.opt.beta2)
        )

        self.brown_predictor.train()

        for epoch in range(1, self.opt.n_epochs + 1):
            epoch_loss = 0
            epoch_deviation = 0
            n_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                he = batch['A'].to(self.device)
                ihc = batch['B'].to(self.device)

                # Get actual brown intensity from IHC
                # Note: This is NOT pixel-aligned, but captures similar distribution
                with torch.no_grad():
                    actual_brown = self.color_deconv.extract_brown_ratio(ihc)

                # Predict brown intensity from H&E
                predicted_brown, uncertainty = self.brown_predictor(he)

                # Gaussian NLL loss - naturally handles uncertainty
                diff = predicted_brown - actual_brown
                nll_loss = 0.5 * (diff ** 2 / (uncertainty + 0.01) + torch.log(uncertainty + 0.01))
                loss = nll_loss.mean()

                # Also add a consistency loss within patches
                # Similar H&E regions should predict similar brown
                smoothness_loss = self._patch_smoothness_loss(predicted_brown) * 0.1

                total_loss = loss + smoothness_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_deviation += torch.abs(diff).mean().item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_deviation = epoch_deviation / n_batches
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Avg Deviation={avg_deviation:.4f}")

            if epoch % self.opt.save_freq == 0:
                self.save_brown_predictor(epoch)

        self.save_brown_predictor('latest')
        print("Brown predictor training complete!")

    def _patch_smoothness_loss(self, pred):
        """Encourage spatial smoothness in predictions."""
        dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        return dx.mean() + dy.mean()

    def train_hallucination_detector(self, dataloader):
        """Train the hallucination detector."""
        print("\n" + "=" * 60)
        print("Training Hallucination Detector")
        print("=" * 60)
        print("This model learns: Real IHC vs Generated IHC")
        print("Captures patterns the generator tends to hallucinate!")
        print("=" * 60 + "\n")

        optimizer = torch.optim.Adam(
            self.hallucination_detector.parameters(),
            lr=self.opt.lr,
            betas=(self.opt.beta1, self.opt.beta2)
        )
        criterion = nn.BCELoss()

        self.hallucination_detector.train()
        self.generator_model.netG_A.eval()

        for epoch in range(1, self.opt.n_epochs + 1):
            epoch_loss = 0
            epoch_acc_real = 0
            epoch_acc_fake = 0
            n_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                he = batch['A'].to(self.device)
                real_ihc = batch['B'].to(self.device)

                # Generate fake IHC
                with torch.no_grad():
                    fake_ihc = self.generator_model.netG_A(he, layers=[])

                # Predict on real
                pred_real = self.hallucination_detector(real_ihc)
                loss_real = criterion(pred_real, torch.ones_like(pred_real))

                # Predict on fake
                pred_fake = self.hallucination_detector(fake_ihc)
                loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

                loss = (loss_real + loss_fake) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute accuracy
                acc_real = (pred_real > 0.5).float().mean().item()
                acc_fake = (pred_fake < 0.5).float().mean().item()

                epoch_loss += loss.item()
                epoch_acc_real += acc_real
                epoch_acc_fake += acc_fake
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_acc_real = epoch_acc_real / n_batches
            avg_acc_fake = epoch_acc_fake / n_batches
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
                  f"Acc(real)={avg_acc_real:.1%}, Acc(fake)={avg_acc_fake:.1%}")

            if epoch % self.opt.save_freq == 0:
                self.save_hallucination_detector(epoch)

        self.save_hallucination_detector('latest')
        print("Hallucination detector training complete!")

    def save_brown_predictor(self, epoch):
        """Save brown predictor checkpoint."""
        path = os.path.join(self.save_dir, f'{epoch}_brown_predictor.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.brown_predictor.state_dict(),
        }, path)
        print(f"Saved brown predictor: {path}")

    def save_hallucination_detector(self, epoch):
        """Save hallucination detector checkpoint."""
        path = os.path.join(self.save_dir, f'{epoch}_hallucination_detector.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.hallucination_detector.state_dict(),
        }, path)
        print(f"Saved hallucination detector: {path}")

    def save_all(self, epoch='latest'):
        """Save all models."""
        self.save_brown_predictor(epoch)
        self.save_hallucination_detector(epoch)

        # Save combined config
        config_path = os.path.join(self.save_dir, 'config.pth')
        torch.save({
            'generator_name': self.opt.generator_name,
            'predictor_ngf': self.opt.predictor_ngf,
            'detector_ndf': self.opt.detector_ndf,
        }, config_path)


def main():
    opt = UnpairedConfidenceOptions().parse()

    # Create dataset
    dataset = create_dataset(opt)
    print(f"Dataset size: {len(dataset)}")

    # Create trainer
    trainer = UnpairedConfidenceTrainer(opt)

    # Train models
    if opt.train_brown_predictor:
        trainer.train_brown_predictor(dataset)

    if opt.train_hallucination_detector:
        trainer.train_hallucination_detector(dataset)

    # Save final models
    trainer.save_all('latest')

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"""
Models saved to: {os.path.join(opt.checkpoints_dir, opt.name)}

To use for inference:
    python inference_unpaired_confidence.py \\
        --dataroot ./datasets/MIST/HER2/TrainValAB \\
        --generator_name {opt.generator_name} \\
        --confidence_name {opt.name} \\
        --phase val
""")


if __name__ == '__main__':
    main()
