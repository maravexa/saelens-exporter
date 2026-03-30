"""saelens-exporter main entry point.

Loads model + SAE, calibrates baseline, starts Unix socket server,
and processes scan commands from the Go control plane.

Usage:
    python -m exporter.main --config config.yaml
    python -m exporter.main --config config.yaml --calibrate-only
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .activation import ActivationExtractor
from .displacement import DisplacementAnalyzer
from .metrics import MetricsWriter
from .protocol import SocketServer

logger = logging.getLogger("saelens-exporter")

# Default baseline prompts — diverse, benign, covers common topics
# Replace with a proper calibration corpus for production use
DEFAULT_BASELINE_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write a short poem about the ocean.",
    "What are the benefits of regular exercise?",
    "How does a combustion engine work?",
    "Describe the water cycle.",
    "What is machine learning?",
    "Tell me about the history of jazz music.",
    "How do you make sourdough bread?",
    "What causes the northern lights?",
]


def load_config(path: str) -> dict:
    """Load and validate configuration."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_components(config: dict) -> tuple[
    ActivationExtractor, DisplacementAnalyzer, MetricsWriter, SocketServer
]:
    """Instantiate all components from config."""
    mc = config["model"]
    extractor = ActivationExtractor(
        model_name=mc["name"],
        hook_point=config["sae"]["hook_point"],
        device=mc.get("device", "cuda"),
        dtype=mc.get("dtype", "float16"),
        max_seq_len=mc.get("max_seq_len", 512),
    )

    sc = config["sae"]
    analyzer = DisplacementAnalyzer(
        sae_release=sc["release"],
        sae_id=sc["hook_point"],
        device=sc.get("device", "cuda"),
        alert_threshold=config["displacement"].get("alert_threshold", 0.35),
    )

    met = config["metrics"]
    writer = MetricsWriter(
        textfile_dir=met["textfile_dir"],
        namespace=met.get("namespace", "saelens"),
        flush_interval=met.get("flush_interval", 30.0),
    )

    sock = config["socket"]
    server = SocketServer(
        socket_path=sock["path"],
        backlog=sock.get("backlog", 5),
    )

    return extractor, analyzer, writer, server


def make_scan_handler(
    extractor: ActivationExtractor,
    analyzer: DisplacementAnalyzer,
    writer: MetricsWriter,
):
    """Create the scan command handler closure."""

    def handle_scan(request: dict[str, Any]) -> dict[str, Any]:
        prompts = request.get("prompts", [])
        scan_id = request.get("scan_id", "unknown")

        results = []
        for i, prompt in enumerate(prompts):
            probe_id = f"{scan_id}_{i:04d}"
            t0 = time.monotonic()

            act_result = extractor.extract(prompt)
            disp_result = analyzer.analyze(act_result)
            duration = time.monotonic() - t0

            is_alert = analyzer.is_displaced(disp_result)

            writer.record(disp_result, probe_id, duration, is_alert)

            results.append({
                "probe_id": probe_id,
                "prompt": prompt[:100],  # truncate for response size
                "displacement": round(disp_result.displacement_magnitude, 6),
                "active_features": disp_result.active_feature_count,
                "alert": is_alert,
                "duration_s": round(duration, 3),
                "top_features": [
                    {"index": idx, "value": round(val, 4)}
                    for idx, val in disp_result.top_features[:5]
                ],
            })

        writer.flush()

        alert_count = sum(1 for r in results if r["alert"])
        logger.info(
            "Scan %s complete: %d prompts, %d alerts",
            scan_id, len(results), alert_count,
        )

        return {
            "scan_id": scan_id,
            "total_prompts": len(results),
            "alert_count": alert_count,
            "results": results,
        }

    return handle_scan


def make_calibrate_handler(
    extractor: ActivationExtractor,
    analyzer: DisplacementAnalyzer,
    writer: MetricsWriter,
):
    """Create the calibrate command handler closure."""

    def handle_calibrate(request: dict[str, Any]) -> dict[str, Any]:
        prompts = request.get("prompts", DEFAULT_BASELINE_PROMPTS)

        logger.info("Calibrating baseline with %d prompts", len(prompts))
        act_results = extractor.extract_batch(prompts)
        baseline = analyzer.calibrate_baseline(act_results)

        writer.set_baseline_size(baseline.n_samples)
        writer.flush()

        return {
            "n_samples": baseline.n_samples,
            "high_variance_features": len(baseline.high_variance_features),
        }

    return handle_calibrate


def make_health_handler():
    """Create the health check handler."""

    def handle_health(_request: dict[str, Any]) -> dict[str, Any]:
        return {"alive": True}

    return handle_health


def main() -> None:
    parser = argparse.ArgumentParser(description="SAELens interpretability exporter")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Run baseline calibration and exit",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Configure logging
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=(
            '{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
            if log_cfg.get("format") == "json"
            else "%(asctime)s %(levelname)s %(message)s"
        ),
    )

    extractor, analyzer, writer, server = build_components(config)

    # --- Load models ---
    logger.info("Loading TransformerLens model: %s", config["model"]["name"])
    extractor.load()

    logger.info("Loading SAE: %s", config["sae"]["release"])
    analyzer.load()

    # --- Calibrate baseline ---
    logger.info("Running baseline calibration...")
    baseline_prompts = DEFAULT_BASELINE_PROMPTS
    act_results = extractor.extract_batch(baseline_prompts)
    baseline = analyzer.calibrate_baseline(act_results)
    writer.set_baseline_size(baseline.n_samples)
    writer.flush()

    if args.calibrate_only:
        logger.info("Calibration complete, exiting")
        extractor.unload()
        return

    # --- Register command handlers ---
    server.register("scan", make_scan_handler(extractor, analyzer, writer))
    server.register("calibrate", make_calibrate_handler(extractor, analyzer, writer))
    server.register("health", make_health_handler())
    server.register("shutdown", lambda _: (server.stop(), {"shutdown": True})[1])

    # --- Signal handling ---
    def _shutdown(signum, frame):
        logger.info("Received signal %d, shutting down", signum)
        server.stop()
        writer.cleanup()
        extractor.unload()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # --- Start server ---
    logger.info("Starting socket server on %s", config["socket"]["path"])
    server.start()


if __name__ == "__main__":
    main()
