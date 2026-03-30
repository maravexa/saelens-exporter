"""Prometheus metrics via textfile collector pattern.

Writes .prom files to a directory that node_exporter or the Go
control plane scrapes. Python never binds a port, never serves HTTP.

Metric naming follows Prometheus conventions:
    saelens_displacement_magnitude          — gauge, per-probe
    saelens_displacement_alert_total        — counter, threshold breaches
    saelens_active_feature_count            — gauge, L0 of SAE features
    saelens_baseline_samples_total          — gauge, baseline size
    saelens_scan_prompts_total              — counter, prompts analyzed
    saelens_scan_duration_seconds           — histogram, per-prompt latency
    saelens_top_feature_activation          — gauge, top-k feature magnitudes
"""

import os
import time
import tempfile
import logging
from pathlib import Path
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from .displacement import DisplacementResult

logger = logging.getLogger(__name__)


class MetricsWriter:
    """Writes Prometheus metrics to textfile collector directory.

    Uses atomic write (write to tmp, rename) to prevent partial reads
    by the scraper.
    """

    def __init__(
        self,
        textfile_dir: str,
        namespace: str = "saelens",
        flush_interval: float = 30.0,
    ):
        self.textfile_dir = Path(textfile_dir)
        self.namespace = namespace
        self.flush_interval = flush_interval
        self._last_flush: float = 0.0

        # Dedicated registry — don't pollute the default global one
        self.registry = CollectorRegistry()

        # --- Gauges (current state) ---
        self.displacement_magnitude = Gauge(
            f"{namespace}_displacement_magnitude",
            "Cosine displacement from baseline centroid",
            ["probe_id"],
            registry=self.registry,
        )

        self.active_feature_count = Gauge(
            f"{namespace}_active_feature_count",
            "Number of active SAE features (L0 norm)",
            ["probe_id"],
            registry=self.registry,
        )

        self.baseline_samples = Gauge(
            f"{namespace}_baseline_samples_total",
            "Number of clean prompts in baseline calibration",
            registry=self.registry,
        )

        self.top_feature_activation = Gauge(
            f"{namespace}_top_feature_activation",
            "Activation magnitude of top SAE features",
            ["feature_index", "probe_id"],
            registry=self.registry,
        )

        # --- Counters (monotonic) ---
        self.alert_total = Counter(
            f"{namespace}_displacement_alert_total",
            "Total displacement threshold breaches",
            registry=self.registry,
        )

        self.prompts_total = Counter(
            f"{namespace}_scan_prompts_total",
            "Total prompts analyzed",
            registry=self.registry,
        )

        # --- Histograms ---
        self.scan_duration = Histogram(
            f"{namespace}_scan_duration_seconds",
            "Time to extract activations and compute displacement",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )

    def record(
        self,
        result: DisplacementResult,
        probe_id: str,
        duration: float,
        is_alert: bool,
    ) -> None:
        """Record metrics for a single displacement result."""
        self.displacement_magnitude.labels(probe_id=probe_id).set(
            result.displacement_magnitude
        )
        self.active_feature_count.labels(probe_id=probe_id).set(
            result.active_feature_count
        )
        self.prompts_total.inc()
        self.scan_duration.observe(duration)

        if is_alert:
            self.alert_total.inc()

        # Top features — labeled by feature index for Grafana filtering
        for feat_idx, feat_val in result.top_features[:10]:
            self.top_feature_activation.labels(
                feature_index=str(feat_idx),
                probe_id=probe_id,
            ).set(feat_val)

        # Auto-flush on interval
        now = time.monotonic()
        if now - self._last_flush >= self.flush_interval:
            self.flush()

    def set_baseline_size(self, n: int) -> None:
        """Update baseline sample count after calibration."""
        self.baseline_samples.set(n)

    def flush(self) -> None:
        """Atomic write of all metrics to the textfile collector dir."""
        self.textfile_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.textfile_dir / f"{self.namespace}.prom"

        metrics_bytes = generate_latest(self.registry)

        # Atomic write: tmp file + rename prevents partial reads
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.textfile_dir), suffix=".prom.tmp"
        )
        try:
            os.write(fd, metrics_bytes)
            os.close(fd)
            os.rename(tmp_path, str(output_path))
            self._last_flush = time.monotonic()
            logger.debug("Flushed metrics to %s", output_path)
        except Exception:
            os.close(fd) if not os.get_inheritable(fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def cleanup(self) -> None:
        """Remove .prom file on shutdown."""
        output_path = self.textfile_dir / f"{self.namespace}.prom"
        if output_path.exists():
            output_path.unlink()
            logger.info("Cleaned up %s", output_path)
