# PRISM-LBS Docker Image for Publication-Ready Reproducibility
# Base: NVIDIA CUDA 12.6 with Ubuntu 22.04
# Target: Nature Communications / Bioinformatics publication standards
#
# Build:
#   docker build -t prism-lbs:1.0 -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -v $(pwd)/data:/data prism-lbs:1.0 detect /data/structure.pdb
#
# Reproduce benchmarks:
#   docker run --gpus all prism-lbs:1.0 validate --all

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    cmake \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Create workspace
WORKDIR /prism

# Copy source code
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY foundation/ ./foundation/
COPY kernels/ ./kernels/
COPY models/ ./models/

# Build release binaries
RUN cargo build --release --features cuda -p prism-lbs

# Verify PTX kernels exist
RUN ls -la kernels/ptx/*.ptx || echo "Warning: PTX kernels not found"

# =============================================================================
# Runtime Stage
# =============================================================================
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for validation
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    requests \
    tqdm

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Create app directory
WORKDIR /app

# Copy binaries from builder
COPY --from=builder /prism/target/release/prism-lbs /app/bin/
COPY --from=builder /prism/target/release/prism-lbs-benchmark /app/bin/
COPY --from=builder /prism/target/release/prism-lbs-train /app/bin/

# Copy PTX kernels
COPY --from=builder /prism/kernels/ptx/ /app/kernels/ptx/

# Copy models
COPY --from=builder /prism/models/ /app/models/

# Copy validation scripts
COPY benchmark/ /app/benchmark/
COPY scripts/ /app/scripts/

# Add binaries to PATH
ENV PATH="/app/bin:${PATH}"

# Set kernel path
ENV PRISM_PTX_PATH=/app/kernels/ptx
ENV PRISM_MODEL_PATH=/app/models

# Create data directories
RUN mkdir -p /data /results

# Default entrypoint
ENTRYPOINT ["prism-lbs"]
CMD ["--help"]

# Labels for publication
LABEL org.opencontainers.image.title="PRISM-LBS"
LABEL org.opencontainers.image.description="GPU-accelerated Ligand Binding Site Prediction"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="PRISM Research Team <IS@Delfictus.com>"
LABEL org.opencontainers.image.source="https://github.com/prism-ai/prism-lbs"
LABEL org.opencontainers.image.licenses="MIT"

# =============================================================================
# Validation Image (Extended)
# =============================================================================
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS validation

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install comprehensive Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    requests \
    tqdm \
    matplotlib \
    seaborn \
    biopython

WORKDIR /app

# Copy from builder
COPY --from=builder /prism/target/release/prism-lbs /app/bin/
COPY --from=builder /prism/kernels/ptx/ /app/kernels/ptx/
COPY --from=builder /prism/models/ /app/models/

# Copy validation infrastructure
COPY benchmark/ /app/benchmark/
COPY scripts/ /app/scripts/

ENV PATH="/app/bin:${PATH}"
ENV PRISM_PTX_PATH=/app/kernels/ptx
ENV PRISM_MODEL_PATH=/app/models
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Validation entrypoint
ENTRYPOINT ["python3", "/app/benchmark/validate_all.py"]
CMD ["--all", "--output", "/results"]
