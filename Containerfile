# Sandboxed Python worker for saelens-exporter
#
# Runs as non-root, no shell in final image, no network capability.
# Communicates exclusively via Unix domain socket (mounted volume)
# and textfile collector directory (mounted volume).
#
# Build:
#   podman build -t saelens-exporter:latest .
#
# Run (pair with Go exporter):
#   podman run --rm \
#     --device /dev/kfd --device /dev/dri \
#     -v /tmp/saelens-exporter.sock:/tmp/saelens-exporter.sock \
#     -v /var/lib/prometheus/textfile_collector:/var/lib/prometheus/textfile_collector \
#     --network=none \
#     --cap-drop=ALL \
#     --security-opt no-new-privileges \
#     saelens-exporter:latest

# --- Build stage: install deps ---
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage: minimal image ---
FROM python:3.11-slim

# Non-root user
RUN useradd --system --no-create-home --shell /usr/sbin/nologin exporter

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY exporter/ ./exporter/
COPY config.yaml .

# Own the textfile dir mount point
RUN mkdir -p /var/lib/prometheus/textfile_collector && \
    chown exporter:exporter /var/lib/prometheus/textfile_collector

USER exporter

ENTRYPOINT ["python", "-m", "exporter.main", "--config", "config.yaml"]
