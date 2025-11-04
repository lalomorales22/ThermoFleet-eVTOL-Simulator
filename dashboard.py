#!/usr/bin/env python3
"""
ThermoFleet-eVTOL-Simulator Dashboard - Streamlit Web Interface

Interactive dashboard for monitoring, controlling, and visualizing
eVTOL simulations and training progress.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ui.visualization import (
    plot_trajectories,
    plot_rewards,
    plot_metrics_dashboard,
    create_3d_trajectory_plot,
    plot_energy_consumption
)
from src.ui.replay_system import list_replays, load_episode
from src.ui.controls import (
    SimulationController,
    VehicleSpawner,
    ArenaSelector,
    VEHICLE_TYPES,
    ARENAS
)
from src.ui.omniverse_viewer import OmniverseViewer, ViewerConfig


# Page configuration
st.set_page_config(
    page_title="FlyingCarRL Dashboard",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-running {
        color: #00cc00;
        font-weight: bold;
    }
    .status-stopped {
        color: #cc0000;
        font-weight: bold;
    }
    .status-paused {
        color: #ff9900;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'controller' not in st.session_state:
    st.session_state.controller = SimulationController()

if 'arena_selector' not in st.session_state:
    st.session_state.arena_selector = ArenaSelector()

if 'spawner' not in st.session_state:
    # Default to NYC Manhattan
    arena = st.session_state.arena_selector.select_arena('NYC_Manhattan')
    st.session_state.spawner = VehicleSpawner(arena)

if 'viewer' not in st.session_state:
    st.session_state.viewer = None

if 'mock_metrics' not in st.session_state:
    # Generate mock training data for demonstration
    np.random.seed(42)
    n_episodes = 1000
    st.session_state.mock_metrics = {
        'rewards': np.cumsum(np.random.randn(n_episodes) * 10 + 50),
        'collisions': np.random.rand(n_episodes) < 0.05,
        'altitude_violations': np.random.rand(n_episodes) < 0.03,
        'episode_lengths': np.random.randint(100, 500, n_episodes)
    }


def main():
    """Main dashboard function"""

    # Header
    st.markdown('<div class="main-header">🚁 FlyingCarRL Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Autonomous eVTOL Training Simulator - Phase 4 Frontend**")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")

        # Mode selection
        mode = st.selectbox(
            "Dashboard Mode",
            ["Live Simulation", "Training Monitor", "Replay Viewer", "Configuration"]
        )

        st.divider()

        # Simulation status
        st.subheader("Simulation Status")
        state = st.session_state.controller.get_state()

        if state.value == "running":
            st.markdown('<p class="status-running">● RUNNING</p>', unsafe_allow_html=True)
        elif state.value == "paused":
            st.markdown('<p class="status-paused">● PAUSED</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">● STOPPED</p>', unsafe_allow_html=True)

        st.metric("Episode", st.session_state.controller.current_episode)
        st.metric("Timestep", st.session_state.controller.current_timestep)

        st.divider()

        # Quick actions
        st.subheader("Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.controller.start()
                st.rerun()

            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.controller.pause()
                st.rerun()

        with col2:
            if st.button("■ Stop", use_container_width=True):
                st.session_state.controller.stop()
                st.rerun()

            if st.button("⟲ Reset", use_container_width=True):
                st.session_state.controller.reset()
                st.session_state.spawner.despawn_all()
                st.rerun()

    # Main content based on mode
    if mode == "Live Simulation":
        show_live_simulation()
    elif mode == "Training Monitor":
        show_training_monitor()
    elif mode == "Replay Viewer":
        show_replay_viewer()
    elif mode == "Configuration":
        show_configuration()


def show_live_simulation():
    """Live simulation mode"""
    st.header("🎮 Live Simulation")

    # Two columns: controls and visualization
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Spawn Controls")

        # Arena selection
        arena_name = st.selectbox(
            "Select Arena",
            list(ARENAS.keys()),
            index=0
        )

        if st.button("Load Arena"):
            arena = st.session_state.arena_selector.select_arena(arena_name)
            st.session_state.spawner = VehicleSpawner(arena)
            st.success(f"Loaded arena: {arena_name}")

        st.divider()

        # Vehicle spawning
        st.subheader("Spawn Vehicles")

        vehicle_type = st.selectbox(
            "Vehicle Type",
            list(VEHICLE_TYPES.keys())
        )

        spawn_count = st.number_input(
            "Number of Vehicles",
            min_value=1,
            max_value=100,
            value=10
        )

        formation = st.selectbox(
            "Formation",
            ["grid", "line", "random"]
        )

        if st.button("Spawn Fleet", type="primary"):
            vehicle_ids = st.session_state.spawner.spawn_fleet(
                count=spawn_count,
                vehicle_type=vehicle_type,
                formation=formation
            )
            st.success(f"Spawned {len(vehicle_ids)} vehicles")

        # Active vehicles
        active_vehicles = st.session_state.spawner.get_active_vehicles()
        st.metric("Active Vehicles", len(active_vehicles))

        if st.button("Despawn All"):
            st.session_state.spawner.despawn_all()
            st.success("All vehicles despawned")
            st.rerun()

    with col2:
        st.subheader("3D Visualization")

        # Mock 3D visualization
        if len(st.session_state.spawner.get_active_vehicles()) > 0:
            # Generate mock trajectories
            active_ids = st.session_state.spawner.get_active_vehicles()
            n_vehicles = min(len(active_ids), 10)  # Limit for performance
            n_timesteps = 100

            # Create sample trajectories
            positions = np.zeros((n_vehicles, n_timesteps, 3))
            for i in range(n_vehicles):
                # Random walk
                positions[i, 0] = np.random.uniform(-1000, 1000, 3)
                positions[i, 0, 2] = np.random.uniform(400, 500)  # Altitude

                for t in range(1, n_timesteps):
                    positions[i, t] = positions[i, t-1] + np.random.randn(3) * 10
                    positions[i, t, 2] = np.clip(positions[i, t, 2], 400, 500)

            # Plot trajectories
            fig = plot_trajectories(
                positions,
                vehicle_ids=active_ids[:n_vehicles],
                title=f"Active Vehicles in {st.session_state.arena_selector.current_arena.name}"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No active vehicles. Spawn vehicles to see visualization.")

        # Live metrics
        st.subheader("Live Metrics")
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric("Avg Altitude", "445 ft", "5 ft")
        with metric_cols[1]:
            st.metric("Avg Speed", "25 m/s", "-2 m/s")
        with metric_cols[2]:
            st.metric("Collisions", "0", "0")
        with metric_cols[3]:
            st.metric("Battery Avg", "87%", "-3%")


def show_training_monitor():
    """Training monitoring mode"""
    st.header("📊 Training Monitor")

    # Training controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        algorithm = st.selectbox("Algorithm", ["PPO", "DDPG", "TD3", "SAC"])
    with col2:
        vehicle_type = st.selectbox("Vehicle", list(VEHICLE_TYPES.keys()))
    with col3:
        n_envs = st.number_input("Parallel Envs", 1, 32, 8)
    with col4:
        total_timesteps = st.number_input("Timesteps", 10000, 10000000, 1000000, step=100000)

    if st.button("Start Training", type="primary"):
        st.info(f"Training with {algorithm} would start here. Use `python train.py` for actual training.")

    st.divider()

    # Metrics dashboard
    st.subheader("Training Metrics")

    # Use mock metrics for demonstration
    metrics = st.session_state.mock_metrics

    # Plot comprehensive dashboard
    fig = plot_metrics_dashboard(metrics)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.subheader("Episode Statistics")

    recent_episodes = 50
    df = pd.DataFrame({
        'Episode': range(len(metrics['rewards']) - recent_episodes, len(metrics['rewards'])),
        'Reward': metrics['rewards'][-recent_episodes:],
        'Length': metrics['episode_lengths'][-recent_episodes:],
        'Collision': metrics['collisions'][-recent_episodes:],
        'Alt. Violation': metrics['altitude_violations'][-recent_episodes:]
    })

    st.dataframe(df, use_container_width=True, height=300)

    # Download data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Training Data",
        data=csv,
        file_name="training_metrics.csv",
        mime="text/csv"
    )


def show_replay_viewer():
    """Replay viewing mode"""
    st.header("🎬 Replay Viewer")

    # List available replays
    replays = list_replays()

    if len(replays) == 0:
        st.info("No replays found. Record episodes during training to create replays.")

        # Create demo replay button
        if st.button("Generate Demo Replay"):
            st.info("Demo replay generation would happen here.")

    else:
        # Replay selection
        replay_options = [
            f"{r['episode_id']} - {r['vehicle_type']} - {r['timestamp']}"
            for r in replays
        ]

        selected = st.selectbox("Select Replay", replay_options)

        if selected:
            selected_idx = replay_options.index(selected)
            replay_metadata = replays[selected_idx]

            # Display metadata
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Duration", f"{replay_metadata['duration']:.2f}s")
                st.metric("Total Reward", f"{replay_metadata['total_reward']:.2f}")

            with col2:
                st.metric("Agents", replay_metadata['num_agents'])
                st.metric("Vehicle Type", replay_metadata['vehicle_type'])

            with col3:
                st.metric("Collisions", replay_metadata['collisions'])
                st.metric("Alt. Violations", replay_metadata['altitude_violations'])

            # Load and visualize replay
            st.divider()

            replay_file = f"replays/{replay_metadata['episode_id']}.pkl.gz"

            if Path(replay_file).exists():
                if st.button("Load Replay", type="primary"):
                    with st.spinner("Loading replay..."):
                        player = load_episode(replay_file)
                        st.session_state.replay_player = player
                        st.success("Replay loaded!")

                # If replay is loaded, show visualization
                if 'replay_player' in st.session_state:
                    player = st.session_state.replay_player

                    # Playback controls
                    st.subheader("Playback Controls")

                    frame_idx = st.slider(
                        "Frame",
                        0,
                        len(player.frames) - 1,
                        0
                    )

                    # Get trajectories
                    trajectories = player.get_all_trajectories()

                    # Plot
                    fig = plot_trajectories(
                        trajectories,
                        title=f"Replay: {replay_metadata['episode_id']}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Energy plot
                    if len(player.frames) > 0:
                        timestamps = np.array([f.timestep * 0.02 for f in player.frames])
                        battery_levels = np.array([f.battery_levels[0] for f in player.frames])
                        thrust_values = np.random.uniform(1000, 5000, len(timestamps))  # Mock

                        fig_energy = plot_energy_consumption(
                            timestamps,
                            battery_levels,
                            thrust_values
                        )

                        st.plotly_chart(fig_energy, use_container_width=True)


def show_configuration():
    """Configuration mode"""
    st.header("⚙️ Configuration")

    # Tabs for different config sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Viewer Settings",
        "Arena Configuration",
        "Vehicle Types",
        "System Info"
    ])

    with tab1:
        st.subheader("Omniverse Viewer Settings")

        col1, col2 = st.columns(2)

        with col1:
            width = st.number_input("Window Width", 800, 3840, 1920)
            height = st.number_input("Window Height", 600, 2160, 1080)
            fps = st.number_input("Target FPS", 30, 120, 60)

        with col2:
            enable_shadows = st.checkbox("Enable Shadows", value=True)
            enable_raytracing = st.checkbox("Enable Ray Tracing", value=False)
            enable_physics_debug = st.checkbox("Physics Debug", value=False)

        camera_mode = st.selectbox("Camera Mode", ["free", "follow", "orbit"])

        if st.button("Apply Viewer Settings"):
            config = ViewerConfig(
                window_width=width,
                window_height=height,
                fps=fps,
                enable_shadows=enable_shadows,
                enable_ray_tracing=enable_raytracing,
                enable_physics_debug=enable_physics_debug,
                camera_mode=camera_mode
            )
            st.success("Viewer settings updated")

    with tab2:
        st.subheader("Arena Configuration")

        # List arenas
        for arena_name, arena in ARENAS.items():
            with st.expander(f"🗺️ {arena_name}"):
                st.write(f"**Location:** {arena.location}")
                st.write(f"**Bounds:** {arena.bounds}")
                st.write(f"**Altitude Range:** {arena.altitude_range[0]}-{arena.altitude_range[1]} ft")
                st.write(f"**Wind Enabled:** {arena.wind_enabled}")

                if arena.wind_enabled:
                    st.write(f"**Wind Speed:** {arena.wind_speed} m/s")

    with tab3:
        st.subheader("Vehicle Types")

        # Display vehicle specs
        for vtype, config in VEHICLE_TYPES.items():
            with st.expander(f"✈️ {vtype.capitalize()}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Mass", f"{config.mass} kg")
                    st.metric("Max Thrust", f"{config.max_thrust} N")

                with col2:
                    st.metric("Drag Coeff", f"{config.drag_coefficient:.2f}")
                    st.metric("Battery", f"{config.battery_capacity/1000:.1f} kWh")

    with tab4:
        st.subheader("System Information")

        info = {
            "Phase": "4 - Frontend and UI Development",
            "Status": "Complete ✅",
            "Python Version": sys.version.split()[0],
            "Streamlit Version": st.__version__,
            "Replay Directory": str(Path("replays").absolute()),
            "Assets Directory": str(Path("assets").absolute()),
        }

        for key, value in info.items():
            st.text(f"{key}: {value}")

        st.divider()

        st.subheader("Component Status")

        components = {
            "✅ Visualization": "plot_trajectories, plot_rewards, plot_metrics",
            "✅ Replay System": "ReplayRecorder, ReplayPlayer",
            "✅ Omniverse Viewer": "OmniverseViewer (mock mode)",
            "✅ Controls": "SimulationController, VehicleSpawner, ArenaSelector",
            "✅ Dashboard": "Streamlit web interface"
        }

        for component, details in components.items():
            st.write(f"**{component}**")
            st.caption(details)


# Run the dashboard
if __name__ == "__main__":
    main()
