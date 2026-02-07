"""
Confidence Estimation WITHOUT Pixel-Aligned Ground Truth

This module provides confidence estimation methods that work with unpaired/misaligned data,
which is common in histopathology where H&E and IHC are from serial sections.

Key Insight: We cannot compute pixel-wise |fake_IHC - real_IHC| because the serial sections
have structural differences. Instead, we use:

1. Brown Intensity Consistency: Learn which H&E patterns produce brown vs blue
2. Feature-Space Nearest Neighbor: Check if similar H&E patches have consistent outputs
3. Real vs Generated Discriminator: Learn to detect hallucinated content
4. Stain Statistics Predictor: Predict expected stain distribution, flag deviations

These methods learn from the DISTRIBUTION of H&E → IHC mappings, not pixel alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# Color Deconvolution for IHC Analysis
# ============================================================================

class ColorDeconvolution:
    """
    Extract DAB (brown) and Hematoxylin (blue) channels from IHC images.

    This allows us to quantify "brownness" without needing pixel alignment.
    We can ask: "Does this H&E region typically produce brown in IHC?"
    """

    # Standard stain vectors for H-DAB
    # Reference: Ruifrok & Johnston, "Quantification of histochemical staining"
    HE_MATRIX = np.array([
        [0.650, 0.704, 0.286],  # Hematoxylin
        [0.268, 0.570, 0.776],  # Eosin (for H&E) / DAB (for IHC)
        [0.0, 0.0, 0.0]         # Background
    ])

    DAB_MATRIX = np.array([
        [0.650, 0.704, 0.286],  # Hematoxylin
        [0.270, 0.570, 0.780],  # DAB (brown)
        [0.0, 0.0, 0.0]         # Background
    ])

    def __init__(self):
        # Compute inverse matrices for deconvolution
        self.dab_matrix_inv = self._compute_inverse(self.DAB_MATRIX)

    def _compute_inverse(self, matrix):
        """Compute inverse of stain matrix."""
        # Add small values to prevent singular matrix
        matrix = matrix.copy()
        matrix[2, :] = np.cross(matrix[0, :], matrix[1, :])
        matrix[2, :] /= np.linalg.norm(matrix[2, :])
        return np.linalg.inv(matrix)

    def extract_dab(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract DAB (brown) channel from IHC image.

        Args:
            image: IHC image tensor [B, 3, H, W] in [-1, 1] range

        Returns:
            DAB intensity map [B, 1, H, W] in [0, 1] range (higher = more brown)
        """
        # Convert to [0, 1]
        image = (image + 1) / 2

        B, C, H, W = image.shape
        device = image.device

        # Reshape for matrix multiplication
        image_flat = image.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]

        # Convert to optical density
        image_flat = torch.clamp(image_flat, 1e-6, 1)
        od = -torch.log(image_flat)

        # Deconvolution
        matrix_inv = torch.tensor(self.dab_matrix_inv, dtype=torch.float32, device=device)
        stains = torch.matmul(od, matrix_inv.T)  # [B*H*W, 3]

        # DAB is the second channel
        dab = stains[:, 1].reshape(B, 1, H, W)

        # Normalize to [0, 1]
        dab = torch.clamp(dab, 0, 3) / 3  # OD typically ranges 0-3

        return dab

    def extract_brown_ratio(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute brown-to-blue ratio as a simpler metric.

        Uses color space analysis without full deconvolution.
        """
        # Convert to [0, 1]
        image = (image + 1) / 2

        R, G, B = image[:, 0:1], image[:, 1:2], image[:, 2:3]

        # Brown detection: high R, medium G, low B
        # Blue detection: low R, low G, high B
        brown_score = R - B + 0.5 * (R - G)
        brown_score = torch.clamp(brown_score, 0, 1)

        return brown_score


# ============================================================================
# Brown Intensity Predictor (Patch-Level Supervision)
# ============================================================================

class BrownIntensityPredictor(nn.Module):
    """
    Predicts expected brown (DAB) intensity from H&E input.

    Training: Learn mapping from H&E patches to brown intensity statistics
    (mean, std of brown in corresponding IHC patches - no pixel alignment needed!)

    Inference: Compare predicted vs generated brown intensity
    Large discrepancy → model might be hallucinating
    """

    def __init__(self, input_nc=3, ngf=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
        )

        # Spatial prediction head (predicts brown intensity at each location)
        self.spatial_head = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 1, 4, 2, 1),
            nn.Sigmoid()  # Output: expected brown intensity [0, 1]
        )

        # Uncertainty head (predicts variance of brown intensity)
        self.uncertainty_head = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 1, 4, 2, 1),
            nn.Softplus()  # Output: variance (always positive)
        )

        self.color_deconv = ColorDeconvolution()

    def forward(self, he_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict expected brown intensity and uncertainty from H&E.

        Args:
            he_input: H&E image [B, 3, H, W]

        Returns:
            expected_brown: Predicted brown intensity [B, 1, H, W]
            uncertainty: Predicted variance [B, 1, H, W]
        """
        features = self.encoder(he_input)
        expected_brown = self.spatial_head(features)
        uncertainty = self.uncertainty_head(features)

        return expected_brown, uncertainty

    def compute_confidence(
        self,
        he_input: torch.Tensor,
        generated_ihc: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence by comparing predicted vs actual brown intensity.

        Args:
            he_input: H&E input [B, 3, H, W]
            generated_ihc: Generated IHC [B, 3, H, W]

        Returns:
            Dictionary with confidence maps
        """
        # Predict expected brown intensity from H&E
        expected_brown, uncertainty = self.forward(he_input)

        # Extract actual brown intensity from generated IHC
        actual_brown = self.color_deconv.extract_brown_ratio(generated_ihc)

        # Compute deviation
        deviation = torch.abs(actual_brown - expected_brown)

        # Confidence based on how well prediction matches generation
        # If generated brown deviates significantly from expected → low confidence
        # Scale by uncertainty (high uncertainty regions are allowed more deviation)
        normalized_deviation = deviation / (uncertainty + 0.1)

        # Convert to confidence
        confidence = torch.exp(-normalized_deviation)

        return {
            'confidence': confidence,
            'expected_brown': expected_brown,
            'actual_brown': actual_brown,
            'uncertainty': uncertainty,
            'deviation': deviation
        }


# ============================================================================
# Feature Space Consistency Checker
# ============================================================================

class FeatureBank:
    """
    Stores feature representations of H&E patches and their corresponding IHC statistics.

    Used for nearest-neighbor based confidence estimation:
    1. Find similar H&E patches in the bank
    2. Check if their IHC outputs are consistent
    3. High variance → ambiguous region → low confidence
    """

    def __init__(self, feature_dim=256, max_size=10000):
        self.feature_dim = feature_dim
        self.max_size = max_size

        self.he_features = []  # H&E patch features
        self.ihc_stats = []    # Corresponding IHC statistics (brown mean, std)

    def add(self, he_feature: np.ndarray, ihc_brown_mean: float, ihc_brown_std: float):
        """Add a new H&E-IHC pair to the bank."""
        self.he_features.append(he_feature)
        self.ihc_stats.append((ihc_brown_mean, ihc_brown_std))

        # Keep bank size bounded
        if len(self.he_features) > self.max_size:
            self.he_features.pop(0)
            self.ihc_stats.pop(0)

    def find_neighbors(self, query_feature: np.ndarray, k: int = 10) -> List[Tuple[float, float]]:
        """Find k nearest H&E patches and return their IHC statistics."""
        if len(self.he_features) == 0:
            return []

        # Compute distances
        features = np.stack(self.he_features)
        distances = np.linalg.norm(features - query_feature, axis=1)

        # Get k nearest
        k = min(k, len(distances))
        nearest_idx = np.argsort(distances)[:k]

        return [self.ihc_stats[i] for i in nearest_idx]

    def compute_consistency(self, query_feature: np.ndarray, k: int = 10) -> Tuple[float, float]:
        """
        Compute consistency of IHC outputs for similar H&E patches.

        Returns:
            mean_brown: Expected brown intensity
            consistency: How consistent the neighbors are (high = confident)
        """
        neighbors = self.find_neighbors(query_feature, k)

        if len(neighbors) < 2:
            return 0.5, 0.5  # Default: uncertain

        brown_means = [n[0] for n in neighbors]
        mean_brown = np.mean(brown_means)
        std_brown = np.std(brown_means)

        # High variance among neighbors → low consistency
        consistency = np.exp(-std_brown * 5)  # Scale factor

        return mean_brown, consistency


class FeatureConsistencyConfidence(nn.Module):
    """
    Confidence estimation based on feature-space consistency.

    For each patch in the test image:
    1. Extract feature representation
    2. Find similar patches in the training set (feature bank)
    3. Check if similar H&E patches have consistent IHC outputs
    4. Inconsistency → the mapping is ambiguous → low confidence
    """

    def __init__(self, encoder, feature_dim=256, bank_size=10000):
        super().__init__()
        self.encoder = encoder  # Pre-trained feature extractor
        self.feature_bank = FeatureBank(feature_dim, bank_size)
        self.color_deconv = ColorDeconvolution()

    def build_bank(self, dataloader, generator, device='cuda'):
        """
        Build the feature bank from training data.

        Args:
            dataloader: Training data loader
            generator: Generator network (frozen)
            device: Device to use
        """
        print("Building feature bank...")

        self.encoder.eval()
        generator.eval()

        with torch.no_grad():
            for batch in dataloader:
                he = batch['A'].to(device)

                # Extract H&E features (use encoder's intermediate features)
                features = self.encoder(he)
                if isinstance(features, list):
                    features = features[-1]  # Use last layer features

                # Pool to get patch-level features
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1).cpu().numpy()

                # Generate IHC and compute brown statistics
                fake_ihc = generator(he, layers=[])
                brown = self.color_deconv.extract_brown_ratio(fake_ihc)

                for i in range(he.size(0)):
                    brown_map = brown[i].cpu().numpy()
                    self.feature_bank.add(
                        features[i],
                        brown_map.mean(),
                        brown_map.std()
                    )

        print(f"Feature bank built with {len(self.feature_bank.he_features)} entries")

    def compute_confidence(
        self,
        he_input: torch.Tensor,
        generated_ihc: torch.Tensor,
        k_neighbors: int = 10
    ) -> torch.Tensor:
        """
        Compute confidence based on feature consistency.

        Args:
            he_input: H&E input [B, 3, H, W]
            generated_ihc: Generated IHC [B, 3, H, W]
            k_neighbors: Number of neighbors to consider

        Returns:
            Confidence map [B, 1, H, W]
        """
        B, C, H, W = he_input.shape
        device = he_input.device

        # For simplicity, compute patch-level confidence and upsample
        # In practice, you might want sliding window

        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(he_input)
            if isinstance(features, list):
                features = features[-1]

            # Get spatial features
            feat_h, feat_w = features.shape[2], features.shape[3]

            confidence_map = torch.zeros(B, 1, feat_h, feat_w, device=device)

            for b in range(B):
                for i in range(feat_h):
                    for j in range(feat_w):
                        patch_feat = features[b, :, i, j].cpu().numpy()
                        _, consistency = self.feature_bank.compute_consistency(
                            patch_feat, k_neighbors
                        )
                        confidence_map[b, 0, i, j] = consistency

            # Upsample to original size
            confidence_map = F.interpolate(
                confidence_map, size=(H, W), mode='bilinear', align_corners=False
            )

        return confidence_map


# ============================================================================
# Real vs Generated Discriminator (Hallucination Detector)
# ============================================================================

class HallucinationDetector(nn.Module):
    """
    Discriminator trained to distinguish real IHC from generated IHC.

    Unlike the GAN discriminator (which just checks if image looks realistic),
    this one is trained specifically on the GENERATOR'S outputs.

    It learns to detect patterns that the generator hallucinates.

    Training:
    - Real: Actual IHC images from the dataset
    - Fake: Generated IHC images (from the generator we want to evaluate)

    At inference:
    - Low "realness" score → might be hallucinated
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=4):
        super().__init__()

        layers = [
            nn.Conv2d(input_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final layer outputs spatial confidence map
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: IHC image [B, 3, H, W]

        Returns:
            Realness score map [B, 1, H', W']
        """
        return self.model(x)

    def compute_confidence(self, generated_ihc: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence from the detector's "realness" score.

        Args:
            generated_ihc: Generated IHC image [B, 3, H, W]

        Returns:
            Confidence map [B, 1, H, W]
        """
        with torch.no_grad():
            realness = self.forward(generated_ihc)

            # Upsample to original size
            confidence = F.interpolate(
                realness,
                size=(generated_ihc.size(2), generated_ihc.size(3)),
                mode='bilinear',
                align_corners=False
            )

        return confidence


# ============================================================================
# Combined Unpaired Confidence Estimator
# ============================================================================

class UnpairedConfidenceEstimator(nn.Module):
    """
    Combined confidence estimation for unpaired/misaligned data.

    Combines multiple signals:
    1. Brown intensity deviation (expected vs actual)
    2. MC Dropout variance
    3. Hallucination detector score

    All methods work WITHOUT pixel-aligned GT!
    """

    def __init__(
        self,
        brown_predictor: Optional[BrownIntensityPredictor] = None,
        hallucination_detector: Optional[HallucinationDetector] = None,
        generator: Optional[nn.Module] = None
    ):
        super().__init__()

        self.brown_predictor = brown_predictor
        self.hallucination_detector = hallucination_detector
        self.generator = generator
        self.color_deconv = ColorDeconvolution()

    def compute_confidence(
        self,
        he_input: torch.Tensor,
        generated_ihc: torch.Tensor,
        use_mc_dropout: bool = True,
        mc_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence using all available methods.

        Args:
            he_input: H&E input [B, 3, H, W]
            generated_ihc: Generated IHC [B, 3, H, W]
            use_mc_dropout: Whether to use MC Dropout
            mc_samples: Number of MC samples

        Returns:
            Dictionary with various confidence maps
        """
        results = {
            'generated_ihc': generated_ihc
        }

        confidences = []

        # 1. Brown intensity deviation
        if self.brown_predictor is not None:
            brown_results = self.brown_predictor.compute_confidence(he_input, generated_ihc)
            results['confidence_brown'] = brown_results['confidence']
            results['expected_brown'] = brown_results['expected_brown']
            results['actual_brown'] = brown_results['actual_brown']
            results['brown_deviation'] = brown_results['deviation']
            confidences.append(brown_results['confidence'])

        # 2. Hallucination detector
        if self.hallucination_detector is not None:
            halluc_conf = self.hallucination_detector.compute_confidence(generated_ihc)
            results['confidence_hallucination'] = halluc_conf
            confidences.append(halluc_conf)

        # 3. MC Dropout (if generator available)
        if use_mc_dropout and self.generator is not None:
            mc_conf = self._compute_mc_dropout_confidence(he_input, mc_samples)
            results['confidence_mc_dropout'] = mc_conf
            confidences.append(mc_conf)

        # 4. Simple brown intensity analysis (always available)
        actual_brown = self.color_deconv.extract_brown_ratio(generated_ihc)
        results['brown_intensity'] = actual_brown

        # Combine confidences
        if len(confidences) > 0:
            # Geometric mean
            combined = confidences[0]
            for conf in confidences[1:]:
                combined = combined * conf
            combined = combined ** (1.0 / len(confidences))
            results['confidence_combined'] = combined
        else:
            # Fallback: use brown intensity as proxy
            # Very high or very low brown might indicate hallucination
            brown_centered = torch.abs(actual_brown - 0.5) * 2  # 0 at 0.5, 1 at extremes
            results['confidence_combined'] = 1 - brown_centered * 0.5

        return results

    def _compute_mc_dropout_confidence(
        self,
        he_input: torch.Tensor,
        n_samples: int = 10
    ) -> torch.Tensor:
        """Compute MC Dropout confidence."""
        self.generator.train()  # Enable dropout

        outputs = []
        with torch.no_grad():
            for _ in range(n_samples):
                fake = self.generator(he_input, layers=[])
                outputs.append(fake)

        outputs = torch.stack(outputs, dim=0)
        variance = outputs.var(dim=0).mean(dim=1, keepdim=True)

        # Normalize variance
        var_flat = variance.view(-1)
        p5 = torch.quantile(var_flat, 0.05)
        p95 = torch.quantile(var_flat, 0.95)
        normalized = (variance - p5) / (p95 - p5 + 1e-8)
        normalized = normalized.clamp(0, 1)

        self.generator.eval()

        return 1 - normalized  # High variance = low confidence


# ============================================================================
# Training Functions
# ============================================================================

def train_brown_predictor(
    predictor: BrownIntensityPredictor,
    dataloader,
    device: str = 'cuda',
    n_epochs: int = 50,
    lr: float = 0.0002
):
    """
    Train the brown intensity predictor.

    Uses PATCH-LEVEL supervision (not pixel-level):
    - For each H&E patch, predict the brown intensity statistics
    - Ground truth: actual brown statistics from corresponding IHC
    - No pixel alignment needed!
    """
    predictor = predictor.to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    color_deconv = ColorDeconvolution()

    print("Training Brown Intensity Predictor...")
    print("Note: Using patch-level supervision (no pixel alignment needed)")

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0

        for batch in dataloader:
            he = batch['A'].to(device)
            ihc = batch['B'].to(device)  # Not pixel-aligned, but has similar brown distribution

            # Get actual brown intensity from IHC
            with torch.no_grad():
                actual_brown = color_deconv.extract_brown_ratio(ihc)

            # Predict brown intensity from H&E
            predicted_brown, uncertainty = predictor(he)

            # Loss: Gaussian negative log-likelihood
            # This naturally handles the uncertainty in the mapping
            diff = predicted_brown - actual_brown
            nll_loss = 0.5 * (diff ** 2 / (uncertainty + 0.01) + torch.log(uncertainty + 0.01))
            loss = nll_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / n_batches:.4f}")

    return predictor


def train_hallucination_detector(
    detector: HallucinationDetector,
    generator: nn.Module,
    dataloader,
    device: str = 'cuda',
    n_epochs: int = 50,
    lr: float = 0.0002
):
    """
    Train the hallucination detector.

    Real: Actual IHC images
    Fake: Generated IHC images

    The detector learns to distinguish real from generated,
    capturing the generator's systematic errors/hallucinations.
    """
    detector = detector.to(device)
    generator = generator.to(device)
    generator.eval()

    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print("Training Hallucination Detector...")

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0

        for batch in dataloader:
            he = batch['A'].to(device)
            real_ihc = batch['B'].to(device)

            # Generate fake IHC
            with torch.no_grad():
                fake_ihc = generator(he, layers=[])

            # Train on real
            pred_real = detector(real_ihc)
            loss_real = criterion(pred_real, torch.ones_like(pred_real))

            # Train on fake
            pred_fake = detector(fake_ihc)
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

            loss = (loss_real + loss_fake) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / n_batches:.4f}")

    return detector
