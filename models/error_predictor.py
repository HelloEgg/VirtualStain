"""
Learned Error Predictor for Confidence-Aware Virtual Staining

The key insight: Instead of relying on proxy signals (cycle consistency, discriminator),
we directly learn to predict WHERE the model will make errors.

Training:
1. Generate fake_IHC = G(H&E)
2. Compute actual error = |fake_IHC - real_IHC|
3. Train ErrorPredictor(H&E, fake_IHC) to predict this error

At test time:
- ErrorPredictor predicts likely error regions without needing GT
- High predicted error = Low confidence = "I don't know this"

This works because:
- Certain H&E patterns are ambiguous (could map to brown or blue)
- The predictor learns to recognize these ambiguous patterns
- The predictor also learns generator's systematic failure modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init


class ErrorPredictorNetwork(nn.Module):
    """
    Network that predicts pixel-wise error given H&E input and generated IHC.

    Architecture: U-Net style encoder-decoder with skip connections
    Input: Concatenated [H&E (3ch), Generated IHC (3ch)] = 6 channels
    Output: Predicted error map (1 channel)
    """

    def __init__(self, input_nc=6, output_nc=1, ngf=64, n_downsampling=4,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        # Encoder
        self.enc1 = self._make_encoder_block(input_nc, ngf, norm_layer, first=True)
        self.enc2 = self._make_encoder_block(ngf, ngf * 2, norm_layer)
        self.enc3 = self._make_encoder_block(ngf * 2, ngf * 4, norm_layer)
        self.enc4 = self._make_encoder_block(ngf * 4, ngf * 8, norm_layer)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, padding=1),
            norm_layer(ngf * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, padding=1),
            norm_layer(ngf * 8),
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connections
        self.dec4 = self._make_decoder_block(ngf * 8 + ngf * 8, ngf * 4, norm_layer, use_dropout)
        self.dec3 = self._make_decoder_block(ngf * 4 + ngf * 4, ngf * 2, norm_layer, use_dropout)
        self.dec2 = self._make_decoder_block(ngf * 2 + ngf * 2, ngf, norm_layer, use_dropout)
        self.dec1 = self._make_decoder_block(ngf + ngf, ngf, norm_layer, use_dropout)

        # Output layer - predicts error magnitude
        self.output = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf // 2, output_nc, kernel_size=1),
            nn.Sigmoid()  # Error in [0, 1] range
        )

        self._init_weights()

    def _make_encoder_block(self, in_ch, out_ch, norm_layer, first=False):
        if first:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _make_decoder_block(self, in_ch, out_ch, norm_layer, use_dropout=False):
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0)

    def forward(self, he_input, generated_ihc):
        """
        Args:
            he_input: H&E image [B, 3, H, W]
            generated_ihc: Generated IHC image [B, 3, H, W]

        Returns:
            predicted_error: Predicted error map [B, 1, H, W]
        """
        # Concatenate inputs
        x = torch.cat([he_input, generated_ihc], dim=1)  # [B, 6, H, W]

        # Encoder
        e1 = self.enc1(x)      # [B, 64, H/2, W/2]
        e2 = self.enc2(e1)     # [B, 128, H/4, W/4]
        e3 = self.enc3(e2)     # [B, 256, H/8, W/8]
        e4 = self.enc4(e3)     # [B, 512, H/16, W/16]

        # Bottleneck
        b = self.bottleneck(e4)  # [B, 512, H/16, W/16]

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([b, e4], dim=1))    # [B, 256, H/8, W/8]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))   # [B, 128, H/4, W/4]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))   # [B, 64, H/2, W/2]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))   # [B, 64, H, W]

        # Output
        predicted_error = self.output(d1)  # [B, 1, H, W]

        return predicted_error


class ErrorPredictorLight(nn.Module):
    """
    Lightweight error predictor - faster but less accurate.
    Uses only H&E input (doesn't need generated IHC).

    This learns: "Given this H&E pattern, how uncertain should the model be?"
    """

    def __init__(self, input_nc=3, output_nc=1, ngf=32):
        super().__init__()

        self.net = nn.Sequential(
            # Encoder
            nn.Conv2d(input_nc, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),

            # Decoder
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, output_nc, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, he_input, generated_ihc=None):
        """Only uses H&E input."""
        return self.net(he_input)


class DualInputErrorPredictor(nn.Module):
    """
    Most powerful error predictor using both:
    1. H&E input features (what patterns are ambiguous?)
    2. Generated IHC features (what did the model produce?)
    3. Cross-attention between them (where do they mismatch conceptually?)
    """

    def __init__(self, input_nc=3, ngf=64):
        super().__init__()

        # Separate encoders for H&E and generated IHC
        self.he_encoder = self._make_encoder(input_nc, ngf)
        self.ihc_encoder = self._make_encoder(input_nc, ngf)

        # Cross-attention module
        self.cross_attention = CrossAttentionBlock(ngf * 4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def _make_encoder(self, input_nc, ngf):
        return nn.Sequential(
            nn.Conv2d(input_nc, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, he_input, generated_ihc):
        # Encode both inputs
        he_feat = self.he_encoder(he_input)
        ihc_feat = self.ihc_encoder(generated_ihc)

        # Cross-attention
        he_attended, ihc_attended = self.cross_attention(he_feat, ihc_feat)

        # Concatenate and decode
        combined = torch.cat([he_attended, ihc_attended], dim=1)
        predicted_error = self.decoder(combined)

        return predicted_error


class CrossAttentionBlock(nn.Module):
    """Cross-attention between H&E and IHC features."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.query_he = nn.Conv2d(channels, channels // 8, 1)
        self.key_ihc = nn.Conv2d(channels, channels // 8, 1)
        self.value_ihc = nn.Conv2d(channels, channels, 1)

        self.query_ihc = nn.Conv2d(channels, channels // 8, 1)
        self.key_he = nn.Conv2d(channels, channels // 8, 1)
        self.value_he = nn.Conv2d(channels, channels, 1)

        self.gamma_he = nn.Parameter(torch.zeros(1))
        self.gamma_ihc = nn.Parameter(torch.zeros(1))

    def forward(self, he_feat, ihc_feat):
        B, C, H, W = he_feat.shape

        # H&E attends to IHC
        q_he = self.query_he(he_feat).view(B, -1, H * W).permute(0, 2, 1)
        k_ihc = self.key_ihc(ihc_feat).view(B, -1, H * W)
        v_ihc = self.value_ihc(ihc_feat).view(B, -1, H * W)

        attn_he = F.softmax(torch.bmm(q_he, k_ihc), dim=-1)
        out_he = torch.bmm(v_ihc, attn_he.permute(0, 2, 1)).view(B, C, H, W)
        he_attended = self.gamma_he * out_he + he_feat

        # IHC attends to H&E
        q_ihc = self.query_ihc(ihc_feat).view(B, -1, H * W).permute(0, 2, 1)
        k_he = self.key_he(he_feat).view(B, -1, H * W)
        v_he = self.value_he(he_feat).view(B, -1, H * W)

        attn_ihc = F.softmax(torch.bmm(q_ihc, k_he), dim=-1)
        out_ihc = torch.bmm(v_he, attn_ihc.permute(0, 2, 1)).view(B, C, H, W)
        ihc_attended = self.gamma_ihc * out_ihc + ihc_feat

        return he_attended, ihc_attended


def define_error_predictor(predictor_type='standard', input_nc=3, ngf=64, gpu_ids=[]):
    """
    Factory function to create error predictor.

    Args:
        predictor_type: 'standard', 'light', or 'dual'
        input_nc: Number of input channels (usually 3 for RGB)
        ngf: Number of generator filters
        gpu_ids: GPU IDs for DataParallel

    Returns:
        Error predictor network
    """
    if predictor_type == 'light':
        net = ErrorPredictorLight(input_nc=input_nc, ngf=ngf // 2)
    elif predictor_type == 'dual':
        net = DualInputErrorPredictor(input_nc=input_nc, ngf=ngf)
    else:
        net = ErrorPredictorNetwork(input_nc=input_nc * 2, ngf=ngf)

    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = nn.DataParallel(net, gpu_ids)

    return net
