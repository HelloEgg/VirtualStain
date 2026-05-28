import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks


def _norm_layer(norm_type, channels):
    if norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    if norm_type == 'instance':
        return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)
    return nn.Identity()


def _spectral(module, enabled=True):
    return nn.utils.spectral_norm(module) if enabled else module


class SelfAttention2d(nn.Module):
    """Self-attention block with learnable residual scaling."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 1)
        self.query = nn.Conv2d(channels, hidden, kernel_size=1)
        self.key = nn.Conv2d(channels, hidden, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        query = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key(x).view(b, -1, h * w)
        attention = torch.bmm(query, key) / (key.shape[1] ** 0.5)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(b, c, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_type='instance', use_spectral=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            _spectral(nn.Conv2d(channels, channels, kernel_size=3), use_spectral),
            _norm_layer(norm_type, channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            _spectral(nn.Conv2d(channels, channels, kernel_size=3), use_spectral),
            _norm_layer(norm_type, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='instance', use_spectral=False):
        super().__init__()
        self.block = nn.Sequential(
            _spectral(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                use_spectral,
            ),
            _norm_layer(norm_type, out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='instance', use_spectral=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            _spectral(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                use_spectral,
            ),
            _norm_layer(norm_type, out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class VirtualStainGenerator(nn.Module):
    """Encoder-decoder generator from the manuscript method."""

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_type='instance',
        n_blocks=9,
        skip_alpha=0.1,
        use_spectral=False,
    ):
        super().__init__()
        self.skip_alpha = skip_alpha
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            _spectral(nn.Conv2d(input_nc, ngf, kernel_size=7), use_spectral),
            _norm_layer(norm_type, ngf),
            nn.ReLU(True),
        )
        self.down1 = DownBlock(ngf, ngf * 2, norm_type, use_spectral)
        self.down2 = DownBlock(ngf * 2, ngf * 4, norm_type, use_spectral)

        blocks = []
        for idx in range(n_blocks):
            blocks.append(ResidualBlock(ngf * 4, norm_type, use_spectral))
            if (idx + 1) % 3 == 0:
                blocks.append(SelfAttention2d(ngf * 4))
        self.transformer = nn.Sequential(*blocks)

        self.up1 = UpBlock(ngf * 4, ngf * 2, norm_type, use_spectral)
        self.up2 = UpBlock(ngf * 2, ngf, norm_type, use_spectral)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            _spectral(nn.Conv2d(ngf, output_nc, kernel_size=7), use_spectral),
            nn.Tanh(),
        )

    def forward(self, x):
        enc0 = self.stem(x)
        enc1 = self.down1(enc0)
        enc2 = self.down2(enc1)
        z = self.transformer(enc2)

        y = self.up1(z)
        if y.shape[2:] != enc1.shape[2:]:
            y = F.interpolate(y, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        y = y + self.skip_alpha * enc1

        y = self.up2(y)
        if y.shape[2:] != enc0.shape[2:]:
            y = F.interpolate(y, size=enc0.shape[2:], mode='bilinear', align_corners=False)
        y = y + self.skip_alpha * enc0
        return self.out(y)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_type='instance', use_spectral=True):
        super().__init__()
        self.net = nn.Sequential(
            _spectral(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), use_spectral),
            nn.LeakyReLU(0.2, True),
            _spectral(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1), use_spectral),
            _norm_layer(norm_type, ndf * 2),
            nn.LeakyReLU(0.2, True),
            _spectral(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1), use_spectral),
            _norm_layer(norm_type, ndf * 4),
            nn.LeakyReLU(0.2, True),
            _spectral(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1), use_spectral),
            _norm_layer(norm_type, ndf * 8),
            nn.LeakyReLU(0.2, True),
            _spectral(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1), use_spectral),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleDiscriminator(nn.Module):
    """Three-scale discriminator operating at 1, 1/2, and 1/4 resolution."""

    def __init__(self, input_nc, ndf=64, norm_type='instance', use_spectral=True):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(input_nc, ndf, norm_type, use_spectral),
            PatchDiscriminator(input_nc, ndf, norm_type, use_spectral),
            PatchDiscriminator(input_nc, ndf, norm_type, use_spectral),
        ])

    def forward(self, x):
        outputs = []
        current = x
        for idx, discriminator in enumerate(self.discriminators):
            if idx > 0:
                current = F.avg_pool2d(current, kernel_size=2, stride=2, count_include_pad=False)
            outputs.append(discriminator(current))
        return outputs


class MILPathologyExtractor(nn.Module):
    """MIL-PPIE module for predicting IHC positivity from H&E crops."""

    def __init__(self, input_nc=3, crop_size=96, num_crops=9, feature_dim=256):
        super().__init__()
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.attention_v = nn.Linear(feature_dim, 128)
        self.attention_w = nn.Linear(128, 1, bias=False)
        self.classifier = nn.Linear(feature_dim, 2)

    def _random_crops(self, x):
        b, c, h, w = x.shape
        crop = self.crop_size
        if h < crop or w < crop:
            target_h = max(h, crop)
            target_w = max(w, crop)
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            h, w = target_h, target_w

        crops = []
        for _ in range(self.num_crops):
            top = torch.randint(0, h - crop + 1, (1,), device=x.device).item()
            left = torch.randint(0, w - crop + 1, (1,), device=x.device).item()
            crops.append(x[:, :, top:top + crop, left:left + crop])
        return torch.stack(crops, dim=1).view(b * self.num_crops, c, crop, crop)

    def forward(self, x):
        b = x.shape[0]
        crops = self._random_crops(x)
        features = self.encoder(crops).flatten(1).view(b, self.num_crops, -1)
        attention = self.attention_w(torch.tanh(self.attention_v(features))).squeeze(-1)
        attention = F.softmax(attention, dim=1).unsqueeze(-1)
        pooled = torch.sum(attention * features, dim=1)
        return self.classifier(pooled)


class IHCWeakClassifier(nn.Module):
    def __init__(self, input_nc=3, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(feature_dim),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x):
        features = self.net(x).flatten(1)
        return self.classifier(features)


class ConfusionDiscriminator(nn.Module):
    """Chooses the generated IHC image from a candidate pool."""

    def __init__(self, input_nc=3, ndf=64, use_spectral=True):
        super().__init__()
        self.encoder = nn.Sequential(
            _spectral(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), use_spectral),
            nn.LeakyReLU(0.2, True),
            _spectral(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1), use_spectral),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            _spectral(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1), use_spectral),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.score = nn.Linear(ndf * 4, 1)

    def forward(self, candidates):
        b, n, c, h, w = candidates.shape
        flat = candidates.view(b * n, c, h, w)
        features = self.encoder(flat).flatten(1)
        scores = self.score(features).view(b, n)
        return scores


class VirtualStainModel(BaseModel):
    """H&E <-> IHC generation model described in the VirtualStain manuscript."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            dataset_mode='unaligned',
            normG='instance',
            normD='instance',
            weight_norm='spectral',
            pool_size=0,
        )
        parser.add_argument('--generator_blocks', type=int, default=9, help='number of residual blocks in each generator')
        parser.add_argument('--skip_alpha', type=float, default=0.1, help='decoder skip connection weight')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for adversarial loss')
        parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle consistency')
        parser.add_argument('--lambda_identity', type=float, default=5.0, help='weight for identity loss')
        parser.add_argument('--lambda_patho', type=float, default=1.0, help='weight for pathology consistency')
        parser.add_argument('--lambda_topo', type=float, default=1.0, help='weight for topology preservation')
        parser.add_argument('--lambda_confusion', type=float, default=1.0, help='weight for confusion discriminator loss')
        parser.add_argument('--lambda_ihc', type=float, default=1.0, help='weight for weak IHC classifier training')
        parser.add_argument('--lambda_ppie', type=float, default=1.0, help='weight for HE_MIL_PPIE self-supervision')
        parser.add_argument('--dab_threshold', type=float, default=0.35, help='DAB mean threshold for weak IHC labels')
        parser.add_argument('--mil_crops', type=int, default=9, help='number of random H&E crops for MIL-PPIE')
        parser.add_argument('--mil_crop_size', type=int, default=96, help='MIL-PPIE crop size')
        parser.add_argument('--confusion_pool_size', type=int, default=4, help='candidate pool size for confusion discriminator')
        parser.add_argument('--topology_percentile', type=float, default=80.0, help='percentile threshold for nuclei centers')
        parser.add_argument('--topology_nms_radius', type=int, default=2, help='radius for nuclei non-maximum suppression')
        parser.add_argument('--topology_blur_steps', type=int, default=4, help='average-pooling steps for soft nuclei maps')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = [
            'D_A', 'D_B', 'G_AB', 'G_BA', 'cycle_A', 'cycle_B',
            'idt_A', 'idt_B', 'patho', 'topo', 'D_conf', 'conf',
            'ihc', 'ppie',
        ]
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        if self.isTrain:
            self.model_names = ['G_AB', 'G_BA', 'D_A', 'D_B', 'PPIE', 'IHC', 'D_conf']
        else:
            self.model_names = ['G_AB', 'G_BA']

        use_spectral = opt.weight_norm == 'spectral'
        self.netG_AB = networks.init_net(
            VirtualStainGenerator(
                opt.input_nc, opt.output_nc, opt.ngf, opt.normG,
                opt.generator_blocks, opt.skip_alpha, use_spectral=False,
            ),
            opt.init_type, opt.init_gain, opt.gpu_ids,
        )
        self.netG_BA = networks.init_net(
            VirtualStainGenerator(
                opt.output_nc, opt.input_nc, opt.ngf, opt.normG,
                opt.generator_blocks, opt.skip_alpha, use_spectral=False,
            ),
            opt.init_type, opt.init_gain, opt.gpu_ids,
        )

        if self.isTrain:
            self.netD_A = networks.init_net(
                MultiScaleDiscriminator(opt.input_nc, opt.ndf, opt.normD, use_spectral),
                opt.init_type, opt.init_gain, opt.gpu_ids,
            )
            self.netD_B = networks.init_net(
                MultiScaleDiscriminator(opt.output_nc, opt.ndf, opt.normD, use_spectral),
                opt.init_type, opt.init_gain, opt.gpu_ids,
            )
            self.netPPIE = networks.init_net(
                MILPathologyExtractor(opt.input_nc, opt.mil_crop_size, opt.mil_crops),
                opt.init_type, opt.init_gain, opt.gpu_ids,
            )
            self.netIHC = networks.init_net(
                IHCWeakClassifier(opt.output_nc),
                opt.init_type, opt.init_gain, opt.gpu_ids,
            )
            self.netD_conf = networks.init_net(
                ConfusionDiscriminator(opt.output_nc, opt.ndf, use_spectral),
                opt.init_type, opt.init_gain, opt.gpu_ids,
            )

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()
            self.criterionCE = nn.CrossEntropyLoss()
            self.criterionKL = nn.KLDivLoss(reduction='batchmean')

            self.optimizer_G = torch.optim.Adam(
                list(self.netG_AB.parameters()) + list(self.netG_BA.parameters()),
                lr=opt.lr, betas=(opt.beta1, opt.beta2),
            )
            self.optimizer_D = torch.optim.Adam(
                list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, opt.beta2),
            )
            self.optimizer_aux = torch.optim.Adam(
                list(self.netPPIE.parameters()) + list(self.netIHC.parameters()),
                lr=opt.lr, betas=(opt.beta1, opt.beta2),
            )
            self.optimizer_conf = torch.optim.Adam(
                self.netD_conf.parameters(),
                lr=opt.lr, betas=(opt.beta1, opt.beta2),
            )
            self.optimizers.extend([
                self.optimizer_G, self.optimizer_D, self.optimizer_aux, self.optimizer_conf,
            ])

    def set_input(self, input_data):
        self.real_A = input_data['A'].to(self.device)
        self.real_B = input_data['B'].to(self.device)
        self.image_paths = input_data['A_paths']

    def forward(self):
        self.fake_B = self.netG_AB(self.real_A)
        self.rec_A = self.netG_BA(self.fake_B)
        self.fake_A = self.netG_BA(self.real_B)
        self.rec_B = self.netG_AB(self.fake_A)

    def data_dependent_initialize(self, data):
        self.set_input(data)
        self.forward()

    def _gan_loss(self, predictions, target_is_real):
        if isinstance(predictions, (list, tuple)):
            losses = [self.criterionGAN(pred, target_is_real).mean() for pred in predictions]
            return sum(losses) / len(losses)
        return self.criterionGAN(predictions, target_is_real).mean()

    def _weak_ihc_labels(self, ihc):
        dab = self._dab_map(ihc)
        scores = dab.flatten(1).mean(dim=1)
        return (scores >= self.opt.dab_threshold).long()

    def _dab_map(self, x):
        x01 = (x + 1.0) * 0.5
        r = x01[:, 0:1]
        g = x01[:, 1:2]
        b = x01[:, 2:3]
        return (0.6 * r + 0.5 * g - 0.2 * b).clamp(0.0, 1.0)

    def _hematoxylin_map(self, x):
        x01 = (x + 1.0) * 0.5
        r = x01[:, 0:1]
        g = x01[:, 1:2]
        b = x01[:, 2:3]
        return (0.6 * b + 0.4 * r - 0.3 * g).clamp(0.0, 1.0)

    def _nuclei_centers(self, x, stain, hard=False):
        if stain == 'he':
            channel = self._hematoxylin_map(x)
        else:
            channel = self._dab_map(x)

        b = channel.shape[0]
        threshold = torch.quantile(
            channel.detach().flatten(1),
            self.opt.topology_percentile / 100.0,
            dim=1,
        ).view(b, 1, 1, 1)
        radius = self.opt.topology_nms_radius
        pooled = F.max_pool2d(channel, kernel_size=2 * radius + 1, stride=1, padding=radius)
        if hard:
            centers = (channel >= threshold) & (channel >= pooled)
            return centers.float()

        soft_threshold = torch.sigmoid((channel - threshold) * 20.0)
        local_peak = torch.sigmoid((channel - pooled.detach() + 1e-3) * 50.0)
        return soft_threshold * local_peak

    def _soft_nuclei_map(self, centers):
        soft = centers
        for _ in range(self.opt.topology_blur_steps):
            soft = F.avg_pool2d(soft, kernel_size=3, stride=1, padding=1)
        return soft

    def _topology_loss(self, real_he, fake_ihc):
        centers_real = self._nuclei_centers(real_he, 'he', hard=True).detach()
        centers_fake = self._nuclei_centers(fake_ihc, 'ihc', hard=False)
        count_real = centers_real.flatten(1).mean(dim=1)
        count_fake = centers_fake.flatten(1).mean(dim=1)
        count_loss = torch.mean(torch.abs(count_real - count_fake))

        soft_real = self._soft_nuclei_map(centers_real)
        soft_fake = self._soft_nuclei_map(centers_fake)
        soft_loss = F.l1_loss(soft_real, soft_fake)

        sorted_real = torch.sort(soft_real.flatten(1), dim=1)[0]
        sorted_fake = torch.sort(soft_fake.flatten(1), dim=1)[0]
        wasserstein = torch.sqrt(torch.mean((sorted_real - sorted_fake) ** 2) + 1e-8)
        return count_loss + soft_loss + wasserstein

    def _confusion_candidates(self, fake):
        pool_size = max(2, self.opt.confusion_pool_size)
        candidates = [fake]
        for idx in range(pool_size - 1):
            candidates.append(torch.roll(self.real_B, shifts=idx + 1, dims=0))
        return torch.stack(candidates, dim=1)

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_real = self._gan_loss(pred_real, True)
        loss_fake = self._gan_loss(pred_fake, False)
        return 0.5 * (loss_real + loss_fake)

    def backward_D(self):
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, self.fake_B)
        loss_D = self.loss_D_A + self.loss_D_B
        loss_D.backward()

    def backward_confusion_D(self):
        candidates = self._confusion_candidates(self.fake_B.detach())
        scores = self.netD_conf(candidates)
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        self.loss_D_conf = self.criterionCE(scores, labels)
        self.loss_D_conf.backward()

    def backward_aux(self):
        labels = self._weak_ihc_labels(self.real_B)
        logits_ihc = self.netIHC(self.real_B)
        self.loss_ihc = self.criterionCE(logits_ihc, labels) * self.opt.lambda_ihc

        perm = torch.randperm(self.real_B.shape[0], device=self.real_B.device)
        target = F.softmax(self.netIHC(self.real_B[perm]).detach(), dim=1)
        log_he = F.log_softmax(self.netPPIE(self.real_A), dim=1)
        self.loss_ppie = self.criterionKL(log_he, target) * self.opt.lambda_ppie

        aux_loss = self.loss_ihc + self.loss_ppie
        aux_loss.backward()

    def backward_G(self):
        self.idt_B = self.netG_AB(self.real_B)
        self.idt_A = self.netG_BA(self.real_A)

        self.loss_G_AB = self._gan_loss(self.netD_B(self.fake_B), True)
        self.loss_G_BA = self._gan_loss(self.netD_A(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_cycle
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_cycle
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A) * self.opt.lambda_identity
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B) * self.opt.lambda_identity

        p_he = F.softmax(self.netPPIE(self.real_A), dim=1).detach()
        log_p_ihc_fake = F.log_softmax(self.netIHC(self.fake_B), dim=1)
        self.loss_patho = self.criterionKL(log_p_ihc_fake, p_he) * self.opt.lambda_patho

        self.loss_topo = self._topology_loss(self.real_A, self.fake_B) * self.opt.lambda_topo

        confusion_scores = self.netD_conf(self._confusion_candidates(self.fake_B))
        prob_fake = F.softmax(confusion_scores, dim=1)[:, 0]
        self.loss_conf = (-torch.log(1.0 - prob_fake + 1e-6)).mean() * self.opt.lambda_confusion

        loss_gan = (self.loss_G_AB + self.loss_G_BA) * self.opt.lambda_GAN
        loss_G = (
            loss_gan + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A +
            self.loss_idt_B + self.loss_patho + self.loss_topo + self.loss_conf
        )
        loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD_conf, True)
        self.optimizer_conf.zero_grad()
        self.backward_confusion_D()
        self.optimizer_conf.step()

        self.set_requires_grad([self.netPPIE, self.netIHC], True)
        self.optimizer_aux.zero_grad()
        self.backward_aux()
        self.optimizer_aux.step()

        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_conf], False)
        self.set_requires_grad([self.netPPIE, self.netIHC], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_conf], True)
        self.set_requires_grad([self.netPPIE, self.netIHC], True)
