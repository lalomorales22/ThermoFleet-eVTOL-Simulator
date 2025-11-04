# FlyingCarRL: Autonomous eVTOL Training Simulator

![Project Banner](docs/images/banner.png)
*(Note: In a real repo, replace with a generated banner image showing a 3D urban airspace with eVTOL vehicles in flight.)*

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technical Specifications](#technical-specifications)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Development Roadmap and Tasks](#development-roadmap-and-tasks)
- [Usage Guide](#usage-guide)
- [Data Management and Database](#data-management-and-database)
- [Testing and Evaluation](#testing-and-evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

**FlyingCarRL** is an open-source, end-to-end simulation platform designed to develop, train, and evaluate neural networks for autonomous flying cars—specifically electric Vertical Takeoff and Landing (eVTOL) vehicles operating in low-altitude urban airspace (400-500 feet). Inspired by advancements in reinforcement learning (RL) for embodied AI, such as NVIDIA's Project GR00T for humanoids, this project adapts similar scalable simulation techniques to aerial mobility.

At its core, FlyingCarRL creates a photorealistic, physics-accurate 3D virtual world using real-time Earth data (e.g., from Google Maps 3D Tiles via Cesium). Users can spawn fleets of customizable eVTOL vehicles (varying in size, shape, weight, and capabilities) into dynamic "arenas" representing urban environments. These vehicles learn via RL to navigate, avoid obstacles, optimize energy use, and coordinate in multi-agent scenarios—all while adhering to simulated airspace regulations.

The project serves dual purposes:
- **Research and Development Tool**: For AI enthusiasts, researchers, or startups prototyping autonomous aerial systems without real-world hardware risks or FAA approvals.
- **Educational Platform**: Demonstrates integration of game engines, geospatial tech, physics simulation, and ML frameworks for 3D RL training.

**What it does:**
- Simulates thousands of eVTOL agents in parallel for efficient RL training.
- Integrates real-world geospatial data for "as-close-to-real-time" urban mapping.
- Logs comprehensive data (sensors, trajectories, rewards) to a database for analysis and iteration.
- Provides a visual frontend for monitoring simulations, debugging, and visualizing learned policies.

By 2025 standards, this aligns with the growing "low-altitude economy" (e.g., urban air mobility initiatives by FAA and companies like Joby Aviation), emphasizing safe, scalable AI for flying vehicles.

## Key Features

- **Scalable Multi-Agent Simulation**: Train 1,000+ eVTOLs simultaneously in GPU-accelerated environments, handling fleet coordination and emergent behaviors.
- **Customizable Vehicles**: Define and spawn multiple vehicle types (e.g., small scout: 100kg, agile; medium passenger: 500kg, balanced; large cargo: 1,000kg, heavy-lift) with variable physics parameters.
- **Photorealistic 3D World**: Stream real-time Earth data (terrain, buildings, elevation) at 400-500 ft altitude, with dynamic elements like weather, wind, and traffic.
- **RL Training Pipeline**: End-to-end support for policies like navigation, collision avoidance, and landing, using optimized libraries for fast iteration.
- **Data Logging and Analytics**: Store all simulation data (e.g., positions, sensor feeds, episode outcomes) in a flexible database for querying, visualization, and model improvement.
- **Visual Frontend**: Real-time viewer with dashboards for spawning agents, monitoring metrics, and replaying episodes.
- **Modular Extensions**: Integrate with external tools for advanced physics (e.g., turbulence models) or hybrid ground-air scenarios.
- **Safety and Compliance Sim**: Enforce virtual geofencing, no-fly zones, and energy constraints to mimic real aviation regs.

## Technical Specifications

- **Simulation Scale**: Up to 10,000 parallel agents on high-end GPUs (e.g., NVIDIA A100); fallback to 100-500 on consumer hardware.
- **Altitude Constraints**: Fixed to 400-500 ft (Z-axis limits in sim coordinates); dynamic adjustments for takeoff/landing transitions.
- **Physics Fidelity**: Real-time aerodynamics (thrust, drag, lift via PhysX/Omniverse); wind turbulence (Gaussian models); battery simulation (energy drain based on mass/thrust).
- **Sensor Emulation**: RGB/Depth cameras, LiDAR point clouds, IMU (accelerometers/gyros), GPS with noise injection for realism.
- **RL Metrics**: Rewards for path efficiency, collision-free flights, altitude adherence; penalties for violations or crashes.
- **Performance Targets**: 1M+ simulation steps/sec in headless mode; 60 FPS in visual mode.
- **Data Volume**: Per episode (10-min sim): ~1GB raw sensor data; aggregated logs ~10MB per 100 agents.
- **Compatibility**: Windows/Linux (primary); macOS partial (no full Omniverse support).
- **Security**: Local-only by default; optional cloud integration with encrypted data transfer.

## Architecture Overview

The system is modular, divided into:

1. **Environment Layer**: NVIDIA Isaac Lab/Omniverse for core sim, extended with Cesium for geospatial mapping and AirSim APIs for aerial dynamics.
2. **Agent Layer**: Custom eVTOL blueprints (USD assets) with RL interfaces (e.g., Gym/PettingZoo env wrappers).
3. **Training Layer**: PufferLib/RLlib for optimized RL algorithms; PyTorch backend for neural nets.
4. **Data Layer**: SQLite (dev) or MySQL (prod) for logging; optional export to CSV/Parquet for ML pipelines.
5. **UI/Frontend Layer**: Omniverse Viewer + custom Python dashboards (e.g., via Streamlit) for interaction.
6. **Orchestration**: Python scripts for spawning, training loops, and DB integration.

**High-level flow**: User defines arena/vehicles → Spawn agents → Run RL episodes → Log data → Analyze/Iterate.

## Tech Stack

- **Simulation Engine**: NVIDIA Isaac Lab (with Omniverse, PhysX for physics).
- **Geospatial Integration**: Cesium for Unreal/Unity (Google 3D Tiles streaming).
- **Aerial Extensions**: AirSim/Pegasus Simulator for eVTOL models.
- **RL Frameworks**: Puffer.ai's PufferLib, Stable Baselines3, Ray RLlib.
- **ML Backend**: PyTorch 2.x for neural nets.
- **Database**: SQLite (lightweight, file-based) for ease; MySQL for scalability.
- **Programming**: Python 3.12+ (core); C++ for custom Omniverse extensions.
- **UI Tools**: Omniverse Composer, Streamlit/Dash for web dashboards.
- **Version Control**: Git (with GitHub Actions for CI/CD).
- **Hardware**: NVIDIA GPU required (RTX 30-series min); CPU fallback slow.

## Prerequisites

- NVIDIA GPU with CUDA 12+ (e.g., RTX 3060 or better).
- Python 3.12 installed.
- NVIDIA Omniverse Launcher (free account).
- Git for cloning repo.
- Optional: Docker for containerized MySQL.

## Installation and Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/FlyingCarRL.git
   cd FlyingCarRL
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Omniverse**:
   - Download from [NVIDIA site](https://www.nvidia.com/en-us/omniverse/)
   - Install Isaac Lab via their [GitHub instructions](https://github.com/isaac-sim/IsaacLab)
   - See [docs/omniverse_setup.md](docs/omniverse_setup.md) for detailed steps

4. **Configure Cesium**:
   - Sign up for [Cesium ion](https://cesium.com/ion/)
   - Add API key to `.env` (copy from `.env.example`)
   - Run: `python scripts/setup_cesium.py`

5. **Initialize DB**:
   ```bash
   python scripts/init_db.py
   ```
   (creates SQLite file or MySQL schema)

6. **Launch sim**:
   - Visual mode: `python main.py --mode=visual`
   - Headless training: `python main.py --mode=headless`

## Development Roadmap and Tasks

This section acts as a comprehensive tasks file, outlining all steps to build the project from scratch. Tasks are broken into phases with subtasks, dependencies, estimated effort (in person-days for a solo dev), and milestones. Assume intermediate programming/ML knowledge; scale up for teams.

### Phase 1: Project Setup and Core Environment (2-5 days) ✅

- **Task 1.1**: Initialize Git repo and structure folders (src/, scripts/, docs/, assets/).
  - Subtasks: Add .gitignore; commit initial README skeleton.
- **Task 1.2**: Install and configure NVIDIA Omniverse and Isaac Lab.
  - Subtasks: Follow NVIDIA docs; test basic humanoid sim to verify GPU accel.
  - Dependencies: NVIDIA account.
- **Task 1.3**: Integrate Cesium for geospatial data.
  - Subtasks: Add Cesium plugin; script to load Google 3D Tiles for a test city (e.g., NYC); constrain Z to 400-500 ft.
  - Effort: 1 day.
- **Task 1.4**: Set up basic arena environment.
  - Subtasks: Create USD scene with urban terrain; add procedural elements (buildings from OSM data).
  - **Milestone**: Render a static 3D map viewable in Omniverse.

### Phase 2: Vehicle Modeling and Spawning (3-7 days) ✅

- **Task 2.1**: Define eVTOL blueprints.
  - Subtasks: Create 3 vehicle types (small: 100kg, sphere-like; medium: 500kg, winged; large: 1,000kg, boxy) using USD assets; parameterize mass, shape, thrust vectors.
- **Task 2.2**: Implement physics customizations.
  - Subtasks: Add aerodynamics (drag/lift equations); wind/turbulence models (using PhysX forces); battery sim (linear drain formula).
  - Dependencies: Phase 1.
- **Task 2.3**: Develop spawning system.
  - Subtasks: Python API to spawn N agents (user-defined count/types); "enter arena" logic (random start positions at 400 ft).
  - Effort: 2 days.
- **Task 2.4**: Add sensor emulation.
  - Subtasks: Integrate cameras/LiDAR/IMU; generate noisy data streams.
  - **Milestone**: Spawn and manually control 10 vehicles in sim.

### Phase 3: RL Integration and Training Pipeline (5-10 days) ✅

- **Task 3.1**: Wrap sim as RL environment.
  - Subtasks: Use Gym/PettingZoo; define obs (sensors), actions (controls), rewards (safe flight at altitude).
- **Task 3.2**: Integrate PufferLib.
  - Subtasks: Clone repo; add wrappers for compatibility; optimize for multi-agent.
  - Dependencies: Phase 2.
- **Task 3.3**: Implement training loops.
  - Subtasks: Scripts for supervised/RL training; support PPO/DDPG algorithms; handle multi-vehicle coordination.
  - Effort: 3 days.
- **Task 3.4**: Add edge cases.
  - Subtasks: Script scenarios (bird strikes, GPS failures, weather changes).
  - **Milestone**: Train a basic policy for single-vehicle navigation.

### Phase 4: Frontend and UI Development (4-8 days) ✅

- **Task 4.1**: Build visual frontend.
  - Subtasks: Use Omniverse Viewer; add camera controls, agent highlighting.
- **Task 4.2**: Create dashboard.
  - Subtasks: Streamlit app for spawning, metric displays (rewards, trajectories); integrate live sim feeds.
  - Dependencies: Phase 3.
- **Task 4.3**: Implement replay system.
  - Subtasks: Save/load episodes; visualize paths in 3D.
  - Effort: 2 days.
- **Task 4.4**: Add user controls.
  - Subtasks: UI to select vehicle types, arena params, start/stop training.
  - **Milestone**: Interactive demo with 100 agents viewable in real-time.

### Phase 5: Data Logging and Database Integration (3-6 days) ✅

- **Task 5.1**: Design DB schema. ✅
  - Subtasks: Tables for episodes (id, vehicle_type, positions JSON, rewards), sensors (timestamps, data blobs), metrics.
- **Task 5.2**: Implement logging hooks. ✅
  - Subtasks: In sim loop, export data to SQLite (default); add MySQL switch for large-scale.
  - Dependencies: Phase 4.
- **Task 5.3**: Add querying tools. ✅
  - Subtasks: Scripts to query DB (e.g., avg reward per vehicle type); export to Pandas for analysis.
  - Effort: 2 days.
- **Task 5.4**: Handle data volume. ✅
  - Subtasks: Compression for blobs; indexing for fast queries; migration script from SQLite to MySQL.
  - **Milestone**: Log and query data from a full training run.

### Phase 6: Testing, Optimization, and Deployment (5-10 days) ✅

- **Task 6.1**: Unit/integration tests. ✅
  - Subtasks: Test spawning, physics, RL convergence using pytest.
  - Completed: Comprehensive test suite with 50+ test cases, pytest configuration, fixtures.
- **Task 6.2**: Performance optimization. ✅
  - Subtasks: Profile GPU usage; enable distributed training (Ray cluster).
  - Completed: Profiling tools, GPU memory tracking, distributed training with Ray.
  - Dependencies: All prior.
- **Task 6.3**: Add compliance features. ✅
  - Subtasks: Virtual geofencing; simulate FAA rules (e.g., corridor paths).
  - Completed: No-fly zones, geofencing, altitude/speed limits, violation tracking.
  - Effort: 3 days.
- **Task 6.4**: Documentation and CI/CD. ✅
  - Subtasks: Expand this README; set up GitHub Actions for builds/tests.
  - Completed: GitHub Actions workflows, Phase 6 documentation, deployment guides.
- **Task 6.5**: Cloud deployment option. ✅
  - Subtasks: Dockerize; AWS/GCP scripts for GPU instances.
  - Completed: Dockerfile, Docker Compose, AWS EC2 scripts, GCP deployment scripts.
  - **Milestone**: Release v1.0 with end-to-end training example.

**Total Estimated Effort**: 22-46 person-days. Prioritize phases sequentially; iterate based on testing.

## Usage Guide

### Training

Train eVTOL agents with different algorithms:

```bash
# Basic training
python train.py --algo=PPO --vehicle-type=medium --total-timesteps=1000000

# Training with custom settings
python main.py --mode=training --algo=DDPG --vehicle-type=large --timesteps=500000

# Headless simulation
python main.py --mode=headless --agents=100 --episodes=1000 --db=sqlite
```

### Visualization

Launch the interactive dashboard:

```bash
# Visual mode with Omniverse
python main.py --mode=visual --agents=50 --arena=NYC_Manhattan

# Streamlit dashboard
streamlit run dashboard.py
```

### Database Analysis

Query and analyze training data:

```bash
# Show database statistics
python scripts/analyze_db.py --stats

# View recent episodes
python scripts/analyze_db.py --episodes --vehicle-type=medium --limit=10

# View performance metrics
python scripts/analyze_db.py --performance --vehicle-type=large

# Export training progress
python scripts/analyze_db.py --training-progress --vehicle-type=small --output=progress.csv

# Export episodes to CSV
python scripts/analyze_db.py --export-csv --output=episodes.csv --vehicle-type=medium

# Custom SQL query
python scripts/analyze_db.py --query="SELECT * FROM episodes WHERE total_reward > 100"
```

### Database Migration

Migrate from SQLite to MySQL for production:

```bash
# Dry run (see what would be migrated)
python scripts/migrate_db.py --from-sqlite --to-mysql --dry-run

# Actual migration
python scripts/migrate_db.py --from-sqlite --to-mysql --batch-size=1000

# Verify migration
python scripts/migrate_db.py --from-sqlite --to-mysql --verify-only
```

## Data Management and Database

The FlyingCarRL platform includes comprehensive database integration for logging and analyzing training data.

### Database Features

- **Dual Database Support**: SQLite (development) and MySQL (production)
- **Comprehensive Logging**: Episodes, metrics, sensor data, and training runs
- **Data Compression**: Automatic compression for trajectory and sensor data
- **Query Tools**: Command-line tools for analyzing training performance
- **Migration Support**: Easy migration from SQLite to MySQL for scaling

### Database Schema

The database includes the following tables:

- **vehicles**: Vehicle type definitions (small, medium, large)
- **arenas**: Simulation environments (NYC, etc.)
- **episodes**: Training episode records with rewards and outcomes
- **metrics**: Per-timestep metrics (position, velocity, battery, etc.)
- **sensor_logs**: Optional sensor data logs (camera, LiDAR, IMU, GPS)
- **training_runs**: Training run metadata and hyperparameters

### Using Database Logging

Integrate database logging into your training scripts:

```python
from src.database import DatabaseLogger
from src.database.callbacks import DatabaseLoggingCallback

# Create logger
db_logger = DatabaseLogger(db_type='sqlite')

# Use with Stable Baselines3
from stable_baselines3 import PPO

callback = DatabaseLoggingCallback(
    db_logger=db_logger,
    training_run_name="my_training_run",
    algorithm="PPO",
    vehicle_type="medium",
    hyperparameters={'learning_rate': 3e-4, 'batch_size': 64}
)

model.learn(total_timesteps=1000000, callback=callback)
```

### Performance Considerations

- **SQLite**: Suitable for up to 100GB datasets, single-user development
- **MySQL**: Recommended for production, supports concurrent access and replication
- **Compression**: Reduces storage by 50-70% for trajectory and sensor data
- **Indexing**: Optimized indexes on episode_number, vehicle_id, and timestamps

## Testing and Evaluation

### Running Tests

FlyingCarRL includes a comprehensive test suite with 50+ test cases:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -v -m unit              # Unit tests only
pytest tests/ -v -m integration       # Integration tests only
pytest tests/ -v -m "not slow"        # Skip slow tests

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

### Performance Profiling

Profile training performance and identify bottlenecks:

```bash
# Basic profiling
python scripts/profile_training.py --algo PPO --timesteps 10000

# GPU memory profiling
python scripts/profile_training.py --algo DDPG --vehicle-type large --profile-memory

# Distributed training
python scripts/distributed_training.py --num-workers 4 --num-gpus 2 --algo PPO
```

### Compliance Testing

Test compliance with aviation regulations:

```python
from src.utils.compliance import ComplianceManager, NoFlyZone

manager = ComplianceManager()
manager.create_default_zones(arena_bounds)

# Check compliance
result = manager.check_compliance(position, velocity, nearby_vehicles)
print(f"Compliant: {result['compliant']}")
print(f"Violations: {result['violations']}")
```

### Evaluation Metrics

- **Collision rate**: Target <1%
- **Altitude compliance**: Target >95%
- **Speed compliance**: Target >98%
- **Training convergence**: Target <10k episodes
- **No-fly zone violations**: Target 0
- **Minimum separation violations**: Target <0.1%

### Benchmarks

Compare policies vs. baselines:
- Random flight
- Rule-based navigation
- Pre-trained models

### Tools

- **pytest**: Automated testing
- **WandB**: Experiment tracking and logging
- **Ray Dashboard**: Distributed training monitoring (port 8265)
- **Streamlit Dashboard**: Real-time visualization (port 8501)
- **GitHub Actions**: Continuous integration

### Cloud Deployment

Deploy to AWS or GCP for large-scale training:

```bash
# AWS EC2 deployment
cd deploy/aws
bash deploy_ec2.sh

# GCP deployment
cd deploy/gcp
bash deploy_gce.sh

# Docker deployment (any cloud)
docker-compose up -d
```

See `deploy/README.md` for detailed deployment instructions.

## Contributing

Fork, PR with tasks completed. Follow code style (PEP8). Issues welcome for feature requests (e.g., add more vehicle types).

## License

MIT License. Free to use/modify, with attribution.

## Acknowledgments

Inspired by [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab), [Puffer.ai](https://puffer.ai/), [AirSim](https://github.com/microsoft/AirSim), [Cesium](https://cesium.com/). Thanks to open-source communities advancing RL and simulation tech.
