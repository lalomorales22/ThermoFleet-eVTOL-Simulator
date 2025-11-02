# FlyingCarRL UI Guide

**Phase 4: Frontend and UI Development - Complete**

This guide covers all the user interface and visualization components available in FlyingCarRL.

## Table of Contents

- [Overview](#overview)
- [Dashboard](#dashboard)
- [Visualization Tools](#visualization-tools)
- [Replay System](#replay-system)
- [Omniverse Viewer](#omniverse-viewer)
- [Control Components](#control-components)
- [Usage Examples](#usage-examples)

## Overview

Phase 4 introduces a comprehensive frontend system for FlyingCarRL with the following components:

- **Interactive Dashboard**: Web-based Streamlit interface for monitoring and control
- **Visualization Library**: Plotly-based plotting tools for trajectories and metrics
- **Replay System**: Record, save, and playback simulation episodes
- **Omniverse Viewer**: Integration with NVIDIA Omniverse for 3D visualization
- **Control Components**: Programmatic interfaces for simulation control

## Dashboard

### Launching the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your web browser at `http://localhost:8501`.

### Dashboard Modes

The dashboard has four modes accessible from the sidebar:

#### 1. Live Simulation

Monitor and control running simulations in real-time.

**Features:**
- Arena selection and loading
- Vehicle spawning controls
- Fleet configuration (type, count, formation)
- 3D trajectory visualization
- Live metrics display

**Usage:**
1. Select an arena from the dropdown
2. Click "Load Arena"
3. Configure vehicle type and spawn count
4. Choose formation (grid, line, random)
5. Click "Spawn Fleet" to add vehicles
6. Monitor the 3D visualization and metrics

#### 2. Training Monitor

Track RL training progress with comprehensive metrics.

**Features:**
- Training configuration
- Real-time metrics dashboard
- Episode statistics table
- Data export capabilities

**Displayed Metrics:**
- Episode rewards (with moving average)
- Collision rate
- Altitude compliance
- Episode length

**Usage:**
1. Configure training parameters (algorithm, vehicle type, etc.)
2. Monitor the metrics dashboard during training
3. Download training data as CSV for further analysis

#### 3. Replay Viewer

Visualize previously recorded simulation episodes.

**Features:**
- Replay selection from saved episodes
- Episode metadata display
- 3D trajectory playback
- Energy consumption analysis

**Usage:**
1. Select a replay from the dropdown
2. View episode metadata and statistics
3. Load the replay to visualize
4. Use the frame slider for playback control

#### 4. Configuration

Manage system settings and view component status.

**Sections:**
- Viewer Settings (resolution, FPS, rendering options)
- Arena Configuration (view and edit arena parameters)
- Vehicle Types (view vehicle specifications)
- System Info (component status and versions)

### Control Panel

The sidebar contains simulation controls:

- **▶️ Start**: Begin simulation
- **⏸️ Pause**: Pause simulation
- **■ Stop**: Stop simulation
- **⟲ Reset**: Reset simulation state

Status indicators show current simulation state:
- 🟢 **RUNNING**: Simulation is active
- 🟠 **PAUSED**: Simulation is paused
- 🔴 **STOPPED**: Simulation is stopped

## Visualization Tools

The visualization module (`src/ui/visualization.py`) provides plotting functions for various data types.

### Plot Trajectories

```python
from src.ui.visualization import plot_trajectories
import numpy as np

# Create sample trajectory data (n_vehicles, n_timesteps, 3)
positions = np.random.randn(5, 100, 3) * 100
positions[:, :, 2] = 450  # Set altitude

# Plot
fig = plot_trajectories(
    positions,
    vehicle_ids=[0, 1, 2, 3, 4],
    title="eVTOL Trajectories",
    show_altitude_bounds=True
)

fig.show()
```

**Features:**
- 3D trajectory lines
- Start/end markers
- Altitude bounds visualization
- Interactive hover information

### Plot Rewards

```python
from src.ui.visualization import plot_rewards

# Training rewards
rewards = np.random.randn(1000).cumsum()

# Plot with moving average
fig = plot_rewards(
    rewards,
    window_size=100,
    title="Training Rewards"
)

fig.show()
```

### Metrics Dashboard

```python
from src.ui.visualization import plot_metrics_dashboard

metrics = {
    'rewards': episode_rewards,
    'collisions': collision_flags,
    'altitude_violations': altitude_violation_flags,
    'episode_lengths': lengths
}

fig = plot_metrics_dashboard(metrics)
fig.show()
```

### Energy Consumption

```python
from src.ui.visualization import plot_energy_consumption

fig = plot_energy_consumption(
    timestamps,
    battery_levels,
    thrust_values
)

fig.show()
```

## Replay System

The replay system allows recording and playback of simulation episodes.

### Recording Episodes

```python
from src.ui.replay_system import ReplayRecorder
import numpy as np

# Initialize recorder
recorder = ReplayRecorder(
    save_dir="replays",
    compress=True,
    save_sensor_data=False
)

# Start recording
recorder.start_recording(
    episode_id="my_episode",
    vehicle_type="medium",
    num_agents=10,
    arena="NYC_Manhattan",
    notes="Test flight"
)

# Record frames during simulation
for timestep in range(1000):
    recorder.record_frame(
        timestep=timestep,
        positions=agent_positions,
        velocities=agent_velocities,
        actions=agent_actions,
        rewards=agent_rewards,
        battery_levels=agent_battery_levels
    )

# Stop and save
filepath = recorder.stop_recording(
    success=True,
    collisions=0,
    altitude_violations=5
)

print(f"Saved to: {filepath}")
```

### Playing Back Episodes

```python
from src.ui.replay_system import load_episode

# Load replay
player = load_episode("replays/my_episode.pkl.gz")

# Get metadata
print(f"Episode: {player.metadata.episode_id}")
print(f"Duration: {player.metadata.duration}s")
print(f"Reward: {player.metadata.total_reward}")

# Get statistics
stats = player.get_statistics()
print(stats)

# Get trajectories
all_trajectories = player.get_all_trajectories()  # (n_agents, n_frames, 3)
agent_0_trajectory = player.get_trajectory(agent_id=0)  # (n_frames, 3)

# Frame-by-frame playback
player.reset()
while True:
    frame = player.next_frame()
    if frame is None:
        break

    # Access frame data
    print(f"Timestep: {frame.timestep}")
    print(f"Positions: {frame.positions}")
    print(f"Rewards: {frame.rewards}")
```

### Listing Replays

```python
from src.ui.replay_system import list_replays

# Get all replays
replays = list_replays(replay_dir="replays")

for replay in replays:
    print(f"{replay['episode_id']}: {replay['total_reward']:.2f}")
```

## Omniverse Viewer

Integration with NVIDIA Omniverse for photorealistic 3D visualization.

**Note**: Full functionality requires NVIDIA Omniverse installation. The current implementation provides a mock interface for development.

### Basic Usage

```python
from src.ui.omniverse_viewer import OmniverseViewer, ViewerConfig

# Create configuration
config = ViewerConfig(
    window_width=1920,
    window_height=1080,
    fps=60,
    enable_shadows=True,
    enable_ray_tracing=False
)

# Initialize viewer
viewer = OmniverseViewer(config=config, headless=False)
viewer.initialize()

# Load arena
viewer.load_arena("NYC_Manhattan")

# Spawn agents
viewer.spawn_agent(
    agent_id=0,
    position=np.array([0, 0, 450]),
    vehicle_type="medium"
)

# Update agent position
viewer.update_agent(
    agent_id=0,
    position=new_position,
    velocity=velocity
)

# Set camera
viewer.set_camera(
    position=np.array([100, 100, 500]),
    target=np.array([0, 0, 450])
)

# Follow agent
viewer.follow_agent(agent_id=0)

# Cleanup
viewer.shutdown()
```

### Visualizing Replays

```python
from src.ui.omniverse_viewer import visualize_replay

# Visualize a saved replay
visualize_replay(
    replay_file="replays/my_episode.pkl.gz",
    playback_speed=1.0
)
```

## Control Components

Programmatic control interfaces for simulations.

### Simulation Controller

```python
from src.ui.controls import SimulationController

controller = SimulationController()

# Control simulation
controller.start()
controller.pause()
controller.resume()
controller.stop()
controller.reset()

# Check state
if controller.is_running():
    print("Simulation is running")

# Register callbacks
def on_start():
    print("Simulation started!")

controller.register_callback('start', on_start)
```

### Vehicle Spawner

```python
from src.ui.controls import VehicleSpawner, ARENAS

arena = ARENAS['NYC_Manhattan']
spawner = VehicleSpawner(arena)

# Spawn single vehicle
vehicle_id = spawner.spawn_single(
    vehicle_type='medium',
    position=np.array([0, 0, 450])
)

# Spawn fleet
fleet_ids = spawner.spawn_fleet(
    count=10,
    vehicle_type='medium',
    formation='grid'
)

# Spawn mixed fleet
mixed_ids = spawner.spawn_mixed_fleet(
    counts={
        'small': 5,
        'medium': 10,
        'large': 3
    },
    formation='random'
)

# Get active vehicles
active = spawner.get_active_vehicles()

# Despawn
spawner.despawn(vehicle_id)
spawner.despawn_all()
```

### Arena Selector

```python
from src.ui.controls import ArenaSelector

selector = ArenaSelector()

# List arenas
arenas = selector.list_arenas()
print(arenas)  # ['NYC_Manhattan', 'SF_Downtown', 'LA_Airport', 'Test_Grid']

# Select arena
arena = selector.select_arena('NYC_Manhattan')

# Get current arena
current = selector.get_current_arena()

# Create custom arena
custom = selector.create_custom_arena(
    name='Custom_City',
    location='Custom Location',
    bounds=(-2000, 2000, -2000, 2000),
    wind_enabled=True,
    wind_speed=10.0
)

# Save/load arena config
selector.save_arena('Custom_City', 'configs/custom_arena.json')
loaded = ArenaSelector.load_arena('configs/custom_arena.json')
```

## Usage Examples

### Example 1: Training with Visualization

```python
from src.ui.replay_system import ReplayRecorder
from src.ui.visualization import plot_rewards
from src.training.ppo_trainer import PPOTrainer

# Initialize recorder
recorder = ReplayRecorder()

# Train
trainer = PPOTrainer(vehicle_type='medium', n_envs=8)

# Record best episodes
# (implement callback in trainer)

trainer.train()

# Visualize results
fig = plot_rewards(trainer.episode_rewards)
fig.show()
```

### Example 2: Headless Batch Simulation

```bash
# Run multiple episodes in headless mode
python main.py --mode=headless --agents=10 --episodes=100 --arena=NYC_Manhattan
```

### Example 3: Interactive Exploration

```bash
# Launch dashboard for interactive exploration
streamlit run dashboard.py
```

Then:
1. Switch to "Live Simulation" mode
2. Select arena and spawn vehicles
3. Monitor trajectories in real-time

### Example 4: Replay Analysis

```python
from src.ui.replay_system import list_replays, load_episode
from src.ui.visualization import plot_trajectories

# Find best episode
replays = list_replays()
best = max(replays, key=lambda x: x['total_reward'])

# Load and analyze
player = load_episode(f"replays/{best['episode_id']}.pkl.gz")
trajectories = player.get_all_trajectories()

# Visualize
fig = plot_trajectories(trajectories, title=f"Best Episode: {best['total_reward']:.2f}")
fig.show()

# Get statistics
stats = player.get_statistics()
print(f"Distance traveled: {stats['total_distance']:.2f} m")
print(f"Mean reward: {stats['mean_reward']:.2f}")
```

## Command Line Interface

### Visual Mode

```bash
python main.py --mode=visual --agents=10 --arena=NYC_Manhattan --vehicle-type=medium
```

### Headless Mode

```bash
python main.py --mode=headless --agents=100 --episodes=1000 --arena=SF_Downtown
```

### Training Mode

```bash
python main.py --mode=training --algo=PPO --vehicle-type=medium --timesteps=1000000
```

## Tips and Best Practices

1. **Performance**: Use headless mode for large-scale simulations
2. **Recording**: Disable sensor data recording (`save_sensor_data=False`) to reduce file size
3. **Visualization**: Limit trajectory plots to 10-20 vehicles for better performance
4. **Training**: Use the dashboard to monitor training progress in real-time
5. **Replays**: Compress replays to save disk space (`compress=True`)

## Troubleshooting

### Dashboard not loading

```bash
# Install streamlit
pip install streamlit

# Run with verbose logging
streamlit run dashboard.py --logger.level=debug
```

### Plotly figures not displaying

```bash
# Install required packages
pip install plotly kaleido
```

### Omniverse integration issues

The Omniverse viewer currently runs in mock mode for development. For full 3D visualization:

1. Install [NVIDIA Omniverse](https://www.nvidia.com/omniverse/)
2. Install Isaac Lab extensions
3. See `docs/omniverse_setup.md` for detailed setup

### Replay files not found

Ensure the `replays/` directory exists:

```bash
mkdir -p replays
```

## Further Reading

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [NVIDIA Omniverse](https://docs.omniverse.nvidia.com/)
- [FlyingCarRL README](../README.md)

## Support

For issues or questions:
- GitHub Issues: [Report an issue](https://github.com/yourusername/FlyingCarRL/issues)
- Documentation: See `docs/` directory
- Examples: See `examples/` directory (coming in Phase 5)

---

**Phase 4 Complete** ✅

All frontend and UI components are now fully implemented and documented.
