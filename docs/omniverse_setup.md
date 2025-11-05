# NVIDIA Omniverse and Isaac Lab Setup Guide

This guide provides detailed instructions for installing and configuring NVIDIA Omniverse and Isaac Lab for ThermoFleet-eVTOL-Simulator development.

## Table of Contents
- [System Requirements](#system-requirements)
- [Omniverse Installation](#omniverse-installation)
- [Isaac Lab Setup](#isaac-lab-setup)
- [Cesium Integration](#cesium-integration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX series (RTX 3060 or better recommended)
  - Minimum: RTX 2060 with 6GB VRAM
  - Recommended: RTX 3080/4080 with 10GB+ VRAM
  - Optimal: RTX A6000/A100 for large-scale training
- **CPU**: 8+ cores recommended
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 100GB+ SSD space for Omniverse and cache

### Software
- **OS**:
  - Ubuntu 20.04/22.04 LTS (primary support)
  - Windows 10/11 (full support)
  - macOS (limited support, no Omniverse)
- **NVIDIA Driver**: Latest (535+ recommended)
- **CUDA**: 12.0 or newer
- **Python**: 3.10-3.12

### Check Your System
```bash
# Check GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Check Python version
python --version
```

## Omniverse Installation

### Step 1: Download Omniverse Launcher

1. Visit [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)
2. Click "Download Omniverse"
3. Create/sign in with NVIDIA account (free)
4. Download the launcher for your OS

**Ubuntu:**
```bash
# Make installer executable
chmod +x omniverse-launcher-linux.AppImage

# Run installer
./omniverse-launcher-linux.AppImage
```

**Windows:**
```powershell
# Run the downloaded .exe
omniverse-launcher-win.exe
```

### Step 2: Install Omniverse Components

Through the launcher, install:

1. **Omniverse Cache** (required)
   - Stores 3D assets locally
   - Recommended location: Default

2. **Omniverse Nucleus** (optional but recommended)
   - Local asset server
   - Useful for team collaboration

3. **Isaac Sim** (required)
   - The core simulation platform
   - Latest stable version (2023.1.1+)
   - Installation size: ~20GB

**Installation Steps:**
```
Launcher → Exchange Tab → Search "Isaac Sim" → Install
```

Wait for download (may take 30-60 minutes depending on connection).

### Step 3: Verify Omniverse Installation

```bash
# Ubuntu - Launch Isaac Sim from terminal
~/.local/share/ov/pkg/isaac_sim-*/isaac-sim.sh

# Windows - Launch from Start Menu or:
# C:\Users\[username]\AppData\Local\ov\pkg\isaac_sim-*\isaac-sim.bat
```

If a viewport opens with a sample scene, installation succeeded!

## Isaac Lab Setup

Isaac Lab is the lightweight RL framework built on Isaac Sim.

### Step 1: Clone Isaac Lab Repository

```bash
# Choose installation directory
cd ~/Projects  # or your preferred location

# Clone the repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Checkout latest stable release (or main for latest)
git checkout main
```

### Step 2: Install Dependencies

Isaac Lab provides installation scripts:

**Ubuntu:**
```bash
# Make install script executable
chmod +x isaaclab.sh

# Run installation (creates conda environment)
./isaaclab.sh --install

# This will:
# - Create conda env: isaaclab
# - Install Python dependencies
# - Link Isaac Sim
```

**Alternative: Manual Setup**
```bash
# Create conda environment
conda create -n isaaclab python=3.10
conda activate isaaclab

# Install dependencies
pip install -e .
```

### Step 3: Set Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Isaac Sim installation path
export ISAAC_SIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2023.1.1"

# Isaac Lab path
export ISAACLAB_PATH="${HOME}/Projects/IsaacLab"

# Add to Python path
export PYTHONPATH="${ISAACLAB_PATH}:${PYTHONPATH}"

# Omniverse cache
export OMNI_CACHE_PATH="${HOME}/.cache/ov"
```

Reload:
```bash
source ~/.bashrc
```

### Step 4: Verify Isaac Lab Installation

```bash
# Activate environment
conda activate isaaclab

# Test basic import
python -c "import omni.isaac.lab; print('Isaac Lab imported successfully!')"

# Run test environment
cd ${ISAACLAB_PATH}
python source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-v0 --headless
```

If training starts, Isaac Lab is working!

## Cesium Integration

### Step 1: Install Cesium for Omniverse

1. Visit [Cesium for Omniverse](https://cesium.com/platform/cesium-for-omniverse/)
2. Download the extension (requires Cesium account - free)
3. Extract to Omniverse extensions folder:

```bash
# Ubuntu
~/.local/share/ov/pkg/isaac_sim-*/exts/

# Windows
C:\Users\[username]\AppData\Local\ov\pkg\isaac_sim-*\exts\
```

### Step 2: Enable Extension in Omniverse

1. Launch Isaac Sim
2. `Window` → `Extensions`
3. Search for "Cesium"
4. Toggle `Enable`

### Step 3: Configure Cesium Ion Token

1. Get token from [Cesium Ion](https://cesium.com/ion/)
2. In Isaac Sim:
   - `Cesium` → `Cesium Settings`
   - Paste your token
3. Or add to `.env` file (see project `.env.example`)

### Step 4: Test Cesium

1. Create new stage in Isaac Sim
2. `Cesium` → `Add Blank 3D Tiles Tileset`
3. In properties, set:
   - Ion Asset ID: `1` (Google Photorealistic 3D Tiles)
   - Token: (your token)
4. Viewport should show Google Earth tiles

## Verification

Run full verification script:

```bash
# From ThermoFleet-eVTOL-Simulator project root
python scripts/verify_setup.py
```

This checks:
- ✓ NVIDIA GPU detected
- ✓ CUDA available
- ✓ Omniverse installed
- ✓ Isaac Lab accessible
- ✓ Cesium configured
- ✓ Python dependencies

## Troubleshooting

### Issue: "No NVIDIA GPU found"

**Solution:**
```bash
# Check if GPU is detected
lspci | grep -i nvidia

# Update drivers
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Issue: "Isaac Sim won't launch"

**Possible causes:**
1. Insufficient VRAM
   - Close other GPU apps
   - Reduce simulation quality settings

2. Driver incompatibility
   ```bash
   # Check driver version
   nvidia-smi

   # Update if needed
   sudo apt update && sudo apt upgrade nvidia-driver-535
   ```

3. Vulkan issues (Linux)
   ```bash
   sudo apt install vulkan-tools
   vulkaninfo
   ```

### Issue: "Cesium tiles not loading"

**Solutions:**
1. Check internet connection (tiles stream from cloud)
2. Verify token is valid at cesium.com/ion/tokens
3. Check firewall/proxy settings
4. Clear cache:
   ```bash
   rm -rf ~/.cache/cesium
   ```

### Issue: "ImportError: No module named 'omni.isaac.lab'"

**Solution:**
```bash
# Ensure environment variables are set
echo $ISAACLAB_PATH
echo $PYTHONPATH

# If empty, add to ~/.bashrc and reload
source ~/.bashrc

# Reinstall Isaac Lab
cd $ISAACLAB_PATH
pip install -e . --force-reinstall
```

### Issue: "Out of memory errors"

**Solutions:**
1. Reduce number of agents
2. Use headless mode:
   ```bash
   python train.py --headless
   ```
3. Decrease sensor resolution
4. Close other applications

### Issue: "Slow simulation performance"

**Optimizations:**
1. Enable headless mode (no rendering)
2. Reduce physics timestep (less accurate but faster)
3. Disable unnecessary sensors
4. Use GPU acceleration for RL training
5. Batch episodes in parallel

## Performance Benchmarks

Expected performance on different hardware:

| GPU | Agents | FPS (Visual) | Steps/sec (Headless) |
|-----|--------|--------------|----------------------|
| RTX 3060 | 100 | 30-40 | 50K |
| RTX 3080 | 500 | 40-50 | 200K |
| RTX 4090 | 1000 | 50-60 | 500K |
| A100 | 5000+ | N/A | 2M+ |

*FPS = Frames per second in visual mode*
*Steps/sec = Simulation steps per second in headless mode*

## Development Workflow

Typical workflow after setup:

1. **Start development session:**
   ```bash
   conda activate isaaclab
   cd /path/to/ThermoFleet-eVTOL-Simulator
   ```

2. **Visual debugging:**
   ```bash
   # Launch with GUI
   python main.py --mode=visual --agents=10
   ```

3. **Training (headless):**
   ```bash
   # Headless for faster training
   python train.py --mode=headless --agents=1000 --episodes=10000
   ```

4. **Monitor with dashboard:**
   ```bash
   # In separate terminal
   streamlit run dashboard.py
   ```

## Next Steps

After successful setup:

1. ✓ Initialize database:
   ```bash
   python scripts/init_db.py
   ```

2. ✓ Configure Cesium:
   ```bash
   python scripts/setup_cesium.py
   ```

3. ✓ Create first arena:
   ```bash
   python scripts/setup_arena.py --name NYC_Test --location "New York City"
   ```

4. ✓ Read [Phase 2 documentation](../README.md#phase-2-vehicle-modeling-and-spawning) for vehicle modeling

## Additional Resources

- [NVIDIA Omniverse Docs](https://docs.omniverse.nvidia.com/)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Cesium for Omniverse Tutorial](https://cesium.com/learn/omniverse/)
- [PhysX Documentation](https://nvidia-omniverse.github.io/PhysX/)

## Support

If you encounter issues:

1. Check [Isaac Lab Discussions](https://github.com/isaac-sim/IsaacLab/discussions)
2. Search [NVIDIA Forums](https://forums.developer.nvidia.com/c/omniverse/)
3. Review [Cesium Community](https://community.cesium.com/)
4. Open issue in [ThermoFleet-eVTOL-Simulator repo](https://github.com/lalomorales22/ThermoFleet-eVTOL-Simulator/issues)

---

**Note**: Installation paths and versions may vary. This guide is based on Isaac Sim 2023.1.1 and Isaac Lab as of January 2025. Check official docs for latest updates.
