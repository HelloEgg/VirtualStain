"""
Confidence-Aware Bidirectional Virtual Staining Model

This model implements the H&E <-> IHC bidirectional translation framework
with cycle-consistency based confidence estimation for hallucination prevention.

Key Components:
1. Forward Generator G: H&E -> IHC
2. Backward Generator F: IHC -> H&E
3. Confidence Estimator: Uses cycle reconstruction error for confidence

Reference: "Confidence-Aware Virtual Staining: Knowing What We Don't Know"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from .asp_loss import AdaptiveSupervisedPatchNCELoss
from .gauss_pyramid import Gauss_Pyramid_Conv
import util.util as util


class ConfidenceModel(BaseModel):
    """
    Confidence-Aware Bidirectional Virtual Staining Model

    This model trains two generators:
    - netG_A: H&E -> IHC (forward translation)
    - netG_B: IHC -> H&E (backward translation)

    Confidence is estimated via:
    1. Cycle consistency error: ||F(G(x)) - x||
    2. Multi-latent sampling variance (if using stochastic generators)
    3. Worst-case reconstruction error across multiple samples
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add model-specific options"""
        # Inherit from CPT options
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'])
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07)
        parser.add_argument('--num_patches', type=int, default=256)
        parser.add_argument('--flip_equivariance', type=util.str2bool, nargs='?', const=True, default=False)
        parser.set_defaults(pool_size=0, dataset_mode='aligned')  # Use aligned for paired H&E-IHC

        # Gaussian Pyramid and ASP options
        parser.add_argument('--lambda_gp', type=float, default=1.0)
        parser.add_argument('--gp_weights', type=str, default='uniform')
        parser.add_argument('--lambda_asp', type=float, default=0.0)
        parser.add_argument('--asp_loss_mode', type=str, default='none')
        parser.add_argument('--n_downsampling', type=int, default=2)

        # Confidence-specific options
        parser.add_argument('--lambda_cycle', type=float, default=10.0,
                            help='weight for cycle consistency loss')
        parser.add_argument('--lambda_cycle_B', type=float, default=10.0,
                            help='weight for backward cycle consistency loss')
        parser.add_argument('--num_latent_samples', type=int, default=5,
                            help='number of latent samples for confidence estimation')
        parser.add_argument('--confidence_mode', type=str, default='cycle_l1',
                            choices=['cycle_l1', 'cycle_l2', 'cycle_ssim', 'variance', 'worst_case'],
                            help='method for confidence estimation')
        parser.add_argument('--confidence_threshold', type=float, default=0.5,
                            help='threshold for low-confidence masking')
        parser.add_argument('--use_dropout_inference', type=util.str2bool, default=False,
                            help='use dropout at inference for uncertainty estimation')
        parser.add_argument('--dropout_rate', type=float, default=0.5,
                            help='dropout rate for uncertainty estimation')

        opt, _ = parser.parse_known_args()

        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(nce_idt=False, lambda_NCE=10.0, flip_equivariance=False,
                                n_epochs=20, n_epochs_decay=10)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # Loss names for logging
        self.loss_names = ['G_A', 'G_B', 'D_A', 'D_B', 'cycle_A', 'cycle_B', 'NCE_A', 'NCE_B']

        # Visual names
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B',
                             'confidence_map_A', 'confidence_map_B']

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'F_A', 'F_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # Define networks
        # G_A: H&E -> IHC (forward)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                         opt.normG, not opt.no_dropout, opt.init_type,
                                         opt.init_gain, opt.no_antialias, opt.no_antialias_up,
                                         self.gpu_ids, opt)
        # G_B: IHC -> H&E (backward)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG,
                                         opt.normG, not opt.no_dropout, opt.init_type,
                                         opt.init_gain, opt.no_antialias, opt.no_antialias_up,
                                         self.gpu_ids, opt)

        # Feature networks for NCE loss
        self.netF_A = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
                                         opt.init_type, opt.init_gain, opt.no_antialias,
                                         self.gpu_ids, opt)
        self.netF_B = networks.define_F(opt.output_nc, opt.netF, opt.normG, not opt.no_dropout,
                                         opt.init_type, opt.init_gain, opt.no_antialias,
                                         self.gpu_ids, opt)

        if self.isTrain:
            # Discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                            opt.normD, opt.init_type, opt.init_gain,
                                            opt.no_antialias, self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                            opt.normD, opt.init_type, opt.init_gain,
                                            opt.no_antialias, self.gpu_ids, opt)

            # Loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(opt).to(self.device)
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()

            # Optimizers
            self.optimizer_G = torch.optim.Adam(
                list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(
                list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Gaussian Pyramid for multi-scale loss
            if self.opt.lambda_gp > 0:
                self.P = Gauss_Pyramid_Conv(num_high=5)
                self.criterionGP = nn.L1Loss()
                if self.opt.gp_weights == 'uniform':
                    self.gp_weights = [1.0] * 6
                else:
                    self.gp_weights = eval(self.opt.gp_weights)
                self.loss_names += ['GP_A', 'GP_B']

            # ASP loss
            if self.opt.lambda_asp > 0:
                self.criterionASP = AdaptiveSupervisedPatchNCELoss(self.opt).to(self.device)
                self.loss_names += ['ASP_A', 'ASP_B']

        # Initialize confidence maps
        self.confidence_map_A = None
        self.confidence_map_B = None

    def data_dependent_initialize(self, data):
        """Initialize feature networks based on data shape"""
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()

        if self.opt.isTrain:
            self.compute_D_loss().backward()
            self.compute_G_loss().backward()

            # Initialize F network optimizers
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(
                    list(self.netF_A.parameters()) + list(self.netF_B.parameters()),
                    lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def set_input(self, input):
        """Unpack input data"""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']

    def forward(self):
        """Forward pass for both directions"""
        # Forward: H&E -> IHC
        self.fake_B = self.netG_A(self.real_A, layers=[])
        # Backward: Reconstruct H&E
        self.rec_A = self.netG_B(self.fake_B, layers=[])

        # Backward: IHC -> H&E
        self.fake_A = self.netG_B(self.real_B, layers=[])
        # Forward: Reconstruct IHC
        self.rec_B = self.netG_A(self.fake_A, layers=[])

        # Compute confidence maps
        self._compute_confidence_maps()

    def _compute_confidence_maps(self):
        """Compute pixel-level confidence maps based on cycle reconstruction error"""
        with torch.no_grad():
            if self.opt.confidence_mode == 'cycle_l1':
                # L1 reconstruction error
                error_A = torch.abs(self.rec_A - self.real_A)
                error_B = torch.abs(self.rec_B - self.real_B)
            elif self.opt.confidence_mode == 'cycle_l2':
                # L2 reconstruction error
                error_A = (self.rec_A - self.real_A) ** 2
                error_B = (self.rec_B - self.real_B) ** 2
            else:
                error_A = torch.abs(self.rec_A - self.real_A)
                error_B = torch.abs(self.rec_B - self.real_B)

            # Average across channels and normalize to [0, 1]
            error_A = error_A.mean(dim=1, keepdim=True)
            error_B = error_B.mean(dim=1, keepdim=True)

            # Convert error to confidence (lower error = higher confidence)
            # Normalize using sigmoid for smooth transition
            self.confidence_map_A = 1 - torch.sigmoid(error_A * 5 - 2.5)
            self.confidence_map_B = 1 - torch.sigmoid(error_B * 5 - 2.5)

    def compute_confidence_with_sampling(self, x, num_samples=None, direction='AtoB'):
        """
        Compute confidence using multiple forward passes with dropout/noise.

        Args:
            x: Input image tensor
            num_samples: Number of samples for variance estimation
            direction: 'AtoB' for H&E->IHC, 'BtoA' for IHC->H&E

        Returns:
            mean_output: Mean prediction across samples
            confidence_map: Confidence map based on variance or worst-case error
            all_outputs: All sampled outputs
        """
        if num_samples is None:
            num_samples = self.opt.num_latent_samples

        if direction == 'AtoB':
            netG_forward = self.netG_A
            netG_backward = self.netG_B
        else:
            netG_forward = self.netG_B
            netG_backward = self.netG_A

        # Enable dropout if specified
        if self.opt.use_dropout_inference:
            netG_forward.train()
            netG_backward.train()
        else:
            netG_forward.eval()
            netG_backward.eval()

        outputs = []
        recon_errors = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Forward pass
                fake = netG_forward(x, layers=[])
                outputs.append(fake)

                # Backward pass for reconstruction
                recon = netG_backward(fake, layers=[])

                # Compute reconstruction error
                error = torch.abs(recon - x).mean(dim=1, keepdim=True)
                recon_errors.append(error)

        # Stack outputs and errors
        outputs = torch.stack(outputs, dim=0)  # [N, B, C, H, W]
        recon_errors = torch.stack(recon_errors, dim=0)  # [N, B, 1, H, W]

        # Compute mean output
        mean_output = outputs.mean(dim=0)

        # Compute confidence based on mode
        if self.opt.confidence_mode == 'variance':
            # Variance across samples
            variance = outputs.var(dim=0).mean(dim=1, keepdim=True)
            # High variance = low confidence
            confidence_map = 1 - torch.sigmoid(variance * 10 - 1)
        elif self.opt.confidence_mode == 'worst_case':
            # Maximum reconstruction error across samples
            max_error = recon_errors.max(dim=0)[0]
            confidence_map = 1 - torch.sigmoid(max_error * 5 - 2.5)
        else:
            # Mean reconstruction error
            mean_error = recon_errors.mean(dim=0)
            confidence_map = 1 - torch.sigmoid(mean_error * 5 - 2.5)

        # Reset to eval mode
        netG_forward.eval()
        netG_backward.eval()

        return mean_output, confidence_map, outputs

    def compute_D_loss(self):
        """Compute discriminator losses for both directions"""
        # D_A: Discriminate real IHC vs fake IHC
        pred_real_B = self.netD_A(self.real_B)
        pred_fake_B = self.netD_A(self.fake_B.detach())
        self.loss_D_A_real = self.criterionGAN(pred_real_B, True).mean()
        self.loss_D_A_fake = self.criterionGAN(pred_fake_B, False).mean()
        self.loss_D_A = (self.loss_D_A_real + self.loss_D_A_fake) * 0.5

        # D_B: Discriminate real H&E vs fake H&E
        pred_real_A = self.netD_B(self.real_A)
        pred_fake_A = self.netD_B(self.fake_A.detach())
        self.loss_D_B_real = self.criterionGAN(pred_real_A, True).mean()
        self.loss_D_B_fake = self.criterionGAN(pred_fake_A, False).mean()
        self.loss_D_B = (self.loss_D_B_real + self.loss_D_B_fake) * 0.5

        return self.loss_D_A + self.loss_D_B

    def compute_G_loss(self):
        """Compute generator losses for both directions"""
        # GAN losses
        if self.opt.lambda_GAN > 0.0:
            pred_fake_B = self.netD_A(self.fake_B)
            pred_fake_A = self.netD_B(self.fake_A)
            self.loss_G_A = self.criterionGAN(pred_fake_B, True).mean() * self.opt.lambda_GAN
            self.loss_G_B = self.criterionGAN(pred_fake_A, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_A = 0.0
            self.loss_G_B = 0.0

        # Cycle consistency losses
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_cycle
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_cycle_B

        # NCE losses for both directions
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE_A = self.calculate_NCE_loss(self.real_A, self.fake_B, self.netG_A, self.netF_A)
            self.loss_NCE_B = self.calculate_NCE_loss(self.real_B, self.fake_A, self.netG_B, self.netF_B)
        else:
            self.loss_NCE_A = 0.0
            self.loss_NCE_B = 0.0

        # Gaussian Pyramid losses
        if self.opt.lambda_gp > 0:
            self.loss_GP_A = self._compute_gp_loss(self.fake_B, self.real_B)
            self.loss_GP_B = self._compute_gp_loss(self.fake_A, self.real_A)
        else:
            self.loss_GP_A = 0.0
            self.loss_GP_B = 0.0

        # ASP losses
        if self.opt.lambda_asp > 0:
            self.loss_ASP_A = self._compute_asp_loss(self.real_B, self.fake_B, self.netG_A, self.netF_A)
            self.loss_ASP_B = self._compute_asp_loss(self.real_A, self.fake_A, self.netG_B, self.netF_B)
        else:
            self.loss_ASP_A = 0.0
            self.loss_ASP_B = 0.0

        # Total loss
        self.loss_G = (self.loss_G_A + self.loss_G_B +
                       self.loss_cycle_A + self.loss_cycle_B +
                       self.loss_NCE_A + self.loss_NCE_B +
                       self.loss_GP_A + self.loss_GP_B +
                       self.loss_ASP_A + self.loss_ASP_B)

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt, netG, netF):
        """Calculate NCE loss for content preservation"""
        feat_src = netG(src, self.nce_layers, encode_only=True)
        feat_tgt = netG(tgt, self.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = netF(feat_src, self.opt.num_patches, None)
        feat_q_pool, _ = netF(feat_tgt, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.criterionNCE(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / len(feat_src)

    def _compute_gp_loss(self, fake, real):
        """Compute Gaussian Pyramid reconstruction loss"""
        p_fake = self.P(fake)
        p_real = self.P(real)
        loss_pyramid = [self.criterionGP(pf, pr) for pf, pr in zip(p_fake, p_real)]
        loss_pyramid = [l * w for l, w in zip(loss_pyramid, self.gp_weights)]
        return torch.mean(torch.stack(loss_pyramid)) * self.opt.lambda_gp

    def _compute_asp_loss(self, real, fake, netG, netF):
        """Compute Adaptive Supervised PatchNCE loss"""
        feat_real = netG(real, self.nce_layers, encode_only=True)
        feat_fake = netG(fake, self.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = netF(feat_real, self.opt.num_patches, None)
        feat_q_pool, _ = netF(feat_fake, self.opt.num_patches, sample_ids)

        total_asp_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.criterionASP(f_q, f_k, self.current_epoch) * self.opt.lambda_asp
            total_asp_loss += loss.mean()

        return total_asp_loss / len(feat_real)

    def optimize_parameters(self):
        """Optimize generator and discriminator parameters"""
        # Forward pass
        self.forward()

        # Update discriminators
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # Update generators
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        if hasattr(self, 'optimizer_F'):
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if hasattr(self, 'optimizer_F'):
            self.optimizer_F.step()

    def get_current_visuals(self):
        """Return visualization images including confidence maps"""
        from collections import OrderedDict
        visual_ret = OrderedDict()

        for name in self.visual_names:
            if isinstance(name, str) and hasattr(self, name):
                value = getattr(self, name)
                if value is not None:
                    # Convert confidence maps to RGB for visualization
                    if 'confidence' in name:
                        value = self._colorize_confidence(value)
                    visual_ret[name] = value

        return visual_ret

    def _colorize_confidence(self, conf_map):
        """Convert confidence map to colorized RGB visualization"""
        # Expand to 3 channels for visualization
        # Green = high confidence, Red = low confidence
        conf_map = conf_map.clamp(0, 1)

        r = 1 - conf_map
        g = conf_map
        b = torch.zeros_like(conf_map)

        return torch.cat([r, g, b], dim=1) * 2 - 1  # Scale to [-1, 1]

    def generate_with_confidence(self, x, direction='AtoB', apply_mask=False):
        """
        Generate translation with confidence map.

        Args:
            x: Input image
            direction: Translation direction
            apply_mask: Whether to mask low-confidence regions

        Returns:
            output: Generated image (optionally masked)
            confidence_map: Pixel-level confidence
            masked_output: Output with low-confidence regions masked (if apply_mask=True)
        """
        self.eval()

        with torch.no_grad():
            if self.opt.num_latent_samples > 1:
                output, confidence_map, _ = self.compute_confidence_with_sampling(
                    x, direction=direction)
            else:
                if direction == 'AtoB':
                    output = self.netG_A(x, layers=[])
                    rec = self.netG_B(output, layers=[])
                else:
                    output = self.netG_B(x, layers=[])
                    rec = self.netG_A(output, layers=[])

                error = torch.abs(rec - x).mean(dim=1, keepdim=True)
                confidence_map = 1 - torch.sigmoid(error * 5 - 2.5)

        result = {'output': output, 'confidence_map': confidence_map}

        if apply_mask:
            mask = (confidence_map > self.opt.confidence_threshold).float()
            # Expand mask to match image channels
            mask = mask.expand_as(output)
            masked_output = output * mask
            result['masked_output'] = masked_output
            result['mask'] = mask

        return result
