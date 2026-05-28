"""
Training Script for Confidence-Aware Virtual Staining Model

This script trains a bidirectional H&E<->IHC translation model with
cycle consistency-based confidence estimation.

Key Features:
1. Bidirectional generators (H&E->IHC and IHC->H&E)
2. Cycle consistency loss for confidence estimation
3. Multi-scale Gaussian pyramid reconstruction loss

Usage:
    python train_confidence.py \
        --dataroot ./datasets/MIST/HER2/TrainValAB \
        --name confidence_her2 \
        --model confidence \
        --lambda_cycle 10.0 \
        --lambda_cycle_B 10.0 \
        --confidence_mode worst_case \
        --num_latent_samples 5

For full options, see: python train_confidence.py --help
"""

import time
import os
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model


def train():
    """Main training function."""
    # Parse options
    opt = TrainOptions().parse()

    # Force model to confidence if not specified
    if opt.model != 'confidence':
        print(f"Note: Changing model from '{opt.model}' to 'confidence'")
        opt.model = 'confidence'

    # Create dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'The number of training images = {dataset_size}')

    # Create model
    model = create_model(opt)
    model.setup(opt)

    # Create visualizer
    from util.visualizer import Visualizer
    visualizer = Visualizer(opt)

    # Total iterations counter
    total_iters = 0

    # Data-dependent initialization
    print("Performing data-dependent initialization...")
    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.parallelize()  # Wrap networks in DataParallel for multi-GPU
            print("Data-dependent initialization complete.")
            break

    # Training loop
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        # Set epoch for dataset (if needed for augmentation scheduling)
        if hasattr(dataset.dataset, 'set_epoch'):
            dataset.dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            # Add epoch info to data
            data['current_epoch'] = epoch
            data['current_iter'] = total_iters

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # Forward pass and optimization
            model.set_input(data)
            model.optimize_parameters()

            # Display images on visdom
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result)

            # Print losses
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                # Also print confidence statistics if available
                if hasattr(model, 'confidence_map_A') and model.confidence_map_A is not None:
                    conf_mean_A = model.confidence_map_A.mean().item()
                    conf_mean_B = model.confidence_map_B.mean().item() if model.confidence_map_B is not None else 0
                    print(f'  [Confidence] A: {conf_mean_A:.4f}, B: {conf_mean_B:.4f}')

                if opt.display_id is not None and opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # Save latest model
            if total_iters % opt.save_latest_freq == 0:
                print(f'Saving the latest model (epoch {epoch}, total_iters {total_iters})')
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # End of epoch
        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t '
              f'Time Taken: {time.time() - epoch_start_time:.0f} sec')

        # Save model at epoch frequency
        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}')
            model.save_networks('latest')
            model.save_networks(epoch)

        # Update learning rate
        model.update_learning_rate()

    print("Training complete!")


if __name__ == '__main__':
    train()
