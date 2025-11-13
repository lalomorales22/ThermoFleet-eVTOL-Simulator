# .env File Update Instructions

## Add CPU/GPU Device Selection

Add these lines to both `.env` and `.env.example` in the **Performance Settings** section:

### Replace this section:
```bash
# ===========================
# Performance Settings
# ===========================
# GPU device ID (for multi-GPU systems)
CUDA_DEVICE=0

# Number of parallel workers
NUM_WORKERS=4

# Batch size for training
BATCH_SIZE=256
```

### With this updated section:
```bash
# ===========================
# Performance Settings
# ===========================
# Compute Device: cpu or cuda (GPU)
# Uncomment ONE of the following lines:
DEVICE=cpu
# DEVICE=cuda

# GPU device ID (for multi-GPU systems, only used if DEVICE=cuda)
CUDA_DEVICE=0

# Number of parallel workers
NUM_WORKERS=4

# Batch size for training
BATCH_SIZE=256

# JAX Device Selection (for thermodynamic computing)
# Options: cpu, gpu
JAX_PLATFORM=cpu
# JAX_PLATFORM=gpu
```

## Usage in Training Scripts

The training scripts will automatically detect these environment variables:

```python
import os
device = os.getenv('DEVICE', 'cpu')  # Defaults to 'cpu'
```

When you get GPU access, simply change:
```bash
DEVICE=cpu
# DEVICE=cuda
```

To:
```bash
# DEVICE=cpu
DEVICE=cuda
```

And for JAX-based thermodynamic computing:
```bash
# JAX_PLATFORM=cpu
JAX_PLATFORM=gpu
```

