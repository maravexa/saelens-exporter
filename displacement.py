"""SAE decomposition and persona displacement monitoring.

Takes raw residual stream activations and decomposes them through
a pretrained sparse autoencoder from SAELens. Computes displacement
vectors relative to a baseline distribution to detect persona drift
under adversarial probing.

This is the core of the garak-axis integration: garak generates the
adversarial prompts, this module measures how the model's internal
representations shift in response.
"""

from dataclasses import dataclass, field
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from sae_lens import SAE

from .activation import ActivationResult

logger = logging.getLogger(__name__)


@dataclass
class DisplacementResult:
    """Displacement metrics for a single prompt."""

    prompt: str
    # Cosine distance from baseline centroid in SAE feature space
    displacement_magnitude: float
    # Per-feature activations (sparse — most will be zero)
    feature_activations: torch.Tensor
    # Top-k active features by magnitude
    top_features: list[tuple[int, float]] = field(default_factory=list)
    # L0 norm: number of active features
    active_feature_count: int = 0


@dataclass
class BaselineProfile:
    """Baseline activation distribution from clean prompts."""

    centroid: torch.Tensor          # Mean SAE feature vector
    std: torch.Tensor               # Per-feature standard deviation
    n_samples: int = 0
    # Feature indices with high variance under clean prompts
    # (these are noisy and should be downweighted in displacement)
    high_variance_features: list[int] = field(default_factory=list)


class DisplacementAnalyzer:
    """Decomposes activations through SAE and tracks displacement.

    Lifecycle:
        1. __init__ + load() — load SAE weights
        2. calibrate_baseline() — establish clean reference distribution
        3. analyze() — compute displacement for adversarial prompts
    """

    def __init__(
        self,
        sae_release: str,
        sae_id: str,
        device: str = "cuda",
        alert_threshold: float = 0.35,
        top_k: int = 20,
    ):
        self.sae_release = sae_release
        self.sae_id = sae_id
        self.device = device
        self.alert_threshold = alert_threshold
        self.top_k = top_k

        self.sae: Optional[SAE] = None
        self.baseline: Optional[BaselineProfile] = None

    def load(self) -> None:
        """Load pretrained SAE from SAELens."""
        self.sae = SAE.from_pretrained(
            release=self.sae_release,
            sae_id=self.sae_id,
            device=self.device,
        )

    def _encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode residual stream activations into SAE feature space.

        Args:
            activations: shape (seq_len, hidden_dim)

        Returns:
            SAE features: shape (sae_hidden_dim,) — mean-pooled over sequence
        """
        if self.sae is None:
            raise RuntimeError("SAE not loaded. Call load() first.")

        # Encode each token position, then mean-pool
        # This gives us a single feature vector representing the
        # overall activation pattern for the full prompt
        features = self.sae.encode(activations)  # (seq_len, sae_dim)
        pooled = features.mean(dim=0)             # (sae_dim,)
        return pooled

    def calibrate_baseline(
        self, activation_results: list[ActivationResult]
    ) -> BaselineProfile:
        """Establish baseline distribution from clean prompt activations.

        Run this with a diverse set of benign prompts before scanning.
        The baseline centroid becomes the reference point for
        displacement measurement.
        """
        feature_vectors = []
        for result in activation_results:
            fv = self._encode(result.activations)
            feature_vectors.append(fv)

        stacked = torch.stack(feature_vectors)  # (n_prompts, sae_dim)
        centroid = stacked.mean(dim=0)
        std = stacked.std(dim=0)

        # Features with high variance under clean prompts are unreliable
        # displacement signals — flag them for downweighting
        variance_threshold = std.mean() + 2 * std.std()
        high_var = (std > variance_threshold).nonzero(as_tuple=True)[0]

        self.baseline = BaselineProfile(
            centroid=centroid,
            std=std,
            n_samples=len(activation_results),
            high_variance_features=high_var.tolist(),
        )

        logger.info(
            "Baseline calibrated: %d samples, %d high-variance features masked",
            self.baseline.n_samples,
            len(self.baseline.high_variance_features),
        )

        return self.baseline

    def analyze(self, activation_result: ActivationResult) -> DisplacementResult:
        """Compute displacement for a single prompt vs baseline.

        This is the hot path during garak scanning — called once per
        adversarial probe.
        """
        if self.baseline is None:
            raise RuntimeError("No baseline. Call calibrate_baseline() first.")

        features = self._encode(activation_result.activations)

        # Cosine distance from baseline centroid
        cos_sim = F.cosine_similarity(
            features.unsqueeze(0),
            self.baseline.centroid.unsqueeze(0),
        ).item()
        displacement = 1.0 - cos_sim

        # Top-k active features by magnitude
        values, indices = torch.topk(features.abs(), self.top_k)
        top_features = [
            (idx.item(), features[idx].item())
            for idx, val in zip(indices, values)
        ]

        active_count = (features.abs() > 1e-6).sum().item()

        return DisplacementResult(
            prompt=activation_result.prompt,
            displacement_magnitude=displacement,
            feature_activations=features.detach().cpu(),
            top_features=top_features,
            active_feature_count=active_count,
        )

    def analyze_batch(
        self, results: list[ActivationResult]
    ) -> list[DisplacementResult]:
        """Analyze a batch of activation results."""
        return [self.analyze(r) for r in results]

    def is_displaced(self, result: DisplacementResult) -> bool:
        """Check if displacement exceeds alert threshold."""
        return result.displacement_magnitude > self.alert_threshold
