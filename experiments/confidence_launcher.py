"""
Launcher for Confidence-Aware Bidirectional Virtual Staining Model

This launcher configures training for the confidence-aware H&E<->IHC
translation model with cycle consistency-based confidence estimation.

Usage:
    python -m experiments.confidence_launcher launch
    python -m experiments.confidence_launcher test
"""

from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            Options(
                # Data configuration
                dataroot="datasets/MIST/HER2/TrainValAB",
                name="confidence_he_ihc",
                checkpoints_dir='checkpoints',

                # Model configuration
                model='confidence',  # Use our confidence-aware model

                # Training schedule
                n_epochs=30,  # Initial learning rate epochs
                n_epochs_decay=10,  # Linear decay epochs

                # Network architecture
                netD='n_layers',
                ndf=32,
                netG='resnet_6blocks',
                n_layers_D=5,
                normG='instance',
                normD='instance',
                weight_norm='spectral',
                n_downsampling=2,

                # GAN loss weight
                lambda_GAN=1.0,

                # Cycle consistency loss weights
                lambda_cycle=10.0,  # Forward cycle: H&E -> IHC -> H&E
                lambda_cycle_B=10.0,  # Backward cycle: IHC -> H&E -> IHC

                # Gaussian Pyramid loss
                lambda_gp=10.0,
                gp_weights='[0.015625,0.03125,0.0625,0.125,0.25,1.0]',

                # Confidence estimation settings
                confidence_mode='cycle_l1',  # Options: cycle_l1, cycle_l2, cycle_ssim, variance, worst_case
                num_latent_samples=5,  # Number of samples for variance estimation
                confidence_threshold=0.5,  # Threshold for low-confidence masking
                use_dropout_inference=False,  # Use dropout for uncertainty

                # Data loading
                dataset_mode='aligned',
                direction='AtoB',
                num_threads=15,
                batch_size=1,
                load_size=1024,
                crop_size=512,
                preprocess='crop',

                # Visualization and saving
                display_winsize=512,
                update_html_freq=100,
                save_epoch_freq=5,
            ),
        ]

    def commands(self):
        return ["python train_confidence.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        """Generate test commands for evaluation with confidence maps."""
        opts = self.common_options()
        phase = 'test'

        for opt in opts:
            opt.set(
                crop_size=1024,
                num_test=1000,
                phase=phase,
            )
            opt.remove(
                'n_epochs', 'n_epochs_decay',
                'update_html_freq', 'save_epoch_freq',
                'continue_train', 'epoch_count'
            )

        return ["python inference_confidence.py " + str(opt) for opt in opts]

    def eval_commands(self):
        """Generate evaluation commands for confidence metrics."""
        opts = self.common_options()
        phase = 'test'

        for opt in opts:
            opt.set(
                crop_size=1024,
                num_test=1000,
                phase=phase,
            )
            opt.remove(
                'n_epochs', 'n_epochs_decay',
                'update_html_freq', 'save_epoch_freq',
                'continue_train', 'epoch_count'
            )

        return ["python evaluate_confidence.py " + str(opt) for opt in opts]


class ConfidenceAblationLauncher(TmuxLauncher):
    """
    Launcher for ablation studies on confidence estimation methods.
    """

    def common_options(self):
        base_opts = {
            'dataroot': "datasets/MIST/HER2/TrainValAB",
            'checkpoints_dir': 'checkpoints',
            'model': 'confidence',
            'n_epochs': 30,
            'n_epochs_decay': 10,
            'netD': 'n_layers',
            'ndf': 32,
            'netG': 'resnet_6blocks',
            'n_layers_D': 5,
            'normG': 'instance',
            'normD': 'instance',
            'weight_norm': 'spectral',
            'lambda_GAN': 1.0,
            'lambda_cycle': 10.0,
            'lambda_cycle_B': 10.0,
            'lambda_gp': 10.0,
            'gp_weights': '[0.015625,0.03125,0.0625,0.125,0.25,1.0]',
            'dataset_mode': 'aligned',
            'direction': 'AtoB',
            'num_threads': 15,
            'batch_size': 1,
            'load_size': 1024,
            'crop_size': 512,
            'preprocess': 'crop',
            'display_winsize': 512,
            'update_html_freq': 100,
            'save_epoch_freq': 5,
        }

        # Ablation configurations
        ablations = [
            # Confidence mode ablations
            {'name': 'conf_cycle_l1', 'confidence_mode': 'cycle_l1', 'num_latent_samples': 1},
            {'name': 'conf_cycle_l2', 'confidence_mode': 'cycle_l2', 'num_latent_samples': 1},
            {'name': 'conf_variance_5', 'confidence_mode': 'variance', 'num_latent_samples': 5},
            {'name': 'conf_variance_10', 'confidence_mode': 'variance', 'num_latent_samples': 10},
            {'name': 'conf_worst_case_5', 'confidence_mode': 'worst_case', 'num_latent_samples': 5},

            # Cycle weight ablations
            {'name': 'conf_cycle_5', 'confidence_mode': 'cycle_l1', 'lambda_cycle': 5.0, 'lambda_cycle_B': 5.0},
            {'name': 'conf_cycle_20', 'confidence_mode': 'cycle_l1', 'lambda_cycle': 20.0, 'lambda_cycle_B': 20.0},
        ]

        options_list = []
        for ablation in ablations:
            opts = {**base_opts}
            opts.update(ablation)
            options_list.append(Options(**opts))

        return options_list

    def commands(self):
        return ["python train_confidence.py " + str(opt) for opt in self.common_options()]


class MISTDatasetLauncher(TmuxLauncher):
    """
    Launcher for training on different MIST dataset markers.
    Supports: HER2, Ki67, ER, PR, etc.
    """

    def common_options(self):
        base_opts = {
            'checkpoints_dir': 'checkpoints',
            'model': 'confidence',
            'n_epochs': 30,
            'n_epochs_decay': 10,
            'netD': 'n_layers',
            'ndf': 32,
            'netG': 'resnet_6blocks',
            'n_layers_D': 5,
            'normG': 'instance',
            'normD': 'instance',
            'weight_norm': 'spectral',
            'lambda_GAN': 1.0,
            'lambda_cycle': 10.0,
            'lambda_cycle_B': 10.0,
            'lambda_gp': 10.0,
            'gp_weights': '[0.015625,0.03125,0.0625,0.125,0.25,1.0]',
            'confidence_mode': 'worst_case',
            'num_latent_samples': 5,
            'dataset_mode': 'aligned',
            'direction': 'AtoB',
            'num_threads': 15,
            'batch_size': 1,
            'load_size': 1024,
            'crop_size': 512,
            'preprocess': 'crop',
            'display_winsize': 512,
            'update_html_freq': 100,
            'save_epoch_freq': 5,
        }

        markers = ['HER2', 'Ki67', 'ER', 'PR']
        options_list = []

        for marker in markers:
            opts = {**base_opts}
            opts['dataroot'] = f"datasets/MIST/{marker}/TrainValAB"
            opts['name'] = f"confidence_{marker.lower()}"
            options_list.append(Options(**opts))

        return options_list

    def commands(self):
        return ["python train_confidence.py " + str(opt) for opt in self.common_options()]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m experiments.confidence_launcher [launch|test|eval]")
        sys.exit(1)

    cmd = sys.argv[1]
    launcher = Launcher()

    if cmd == "launch":
        for cmd_str in launcher.commands():
            print(cmd_str)
    elif cmd == "test":
        for cmd_str in launcher.test_commands():
            print(cmd_str)
    elif cmd == "eval":
        for cmd_str in launcher.eval_commands():
            print(cmd_str)
