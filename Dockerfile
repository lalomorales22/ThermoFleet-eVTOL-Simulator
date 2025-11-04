# ThermoFleet-eVTOL-Simulator Docker Image
# Multi-stage build for optimized image size

# Stage 1: Base image with dependencies
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Stage 2: Dependencies installation
FROM base as dependencies

WORKDIR /tmp

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 3: Final image
FROM base

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash flyingcar && \
    mkdir -p /app /data /logs && \
    chown -R flyingcar:flyingcar /app /data /logs

# Copy installed dependencies from previous stage
COPY --from=dependencies /root/.local /home/flyingcar/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=flyingcar:flyingcar . /app/

# Set PATH to include user-installed packages
ENV PATH=/home/flyingcar/.local/bin:$PATH

# Switch to non-root user
USER flyingcar

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/logs /app/data

# Expose ports
EXPOSE 8501 8265 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "main.py", "--mode=headless", "--agents=10"]

# Alternative commands (can be overridden):
# Training: docker run flyingcarrl python train.py --algo PPO
# Dashboard: docker run -p 8501:8501 flyingcarrl streamlit run dashboard.py
# Profile: docker run flyingcarrl python scripts/profile_training.py
