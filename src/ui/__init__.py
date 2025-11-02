"""
FlyingCarRL UI Module

This module provides frontend and visualization components for the FlyingCarRL simulation.

Components:
- visualization: Plotting utilities for trajectories and metrics
- replay_system: Episode recording and playback
- omniverse_viewer: 3D visualization integration
- controls: User interaction components
"""

__version__ = "1.0.0"

from .visualization import (
    plot_trajectories,
    plot_rewards,
    plot_metrics_dashboard,
    create_3d_trajectory_plot
)

from .replay_system import (
    ReplayRecorder,
    ReplayPlayer,
    save_episode,
    load_episode
)

from .omniverse_viewer import (
    OmniverseViewer,
    ViewerConfig
)

from .controls import (
    SimulationController,
    VehicleSpawner,
    ArenaSelector
)

__all__ = [
    # Visualization
    'plot_trajectories',
    'plot_rewards',
    'plot_metrics_dashboard',
    'create_3d_trajectory_plot',

    # Replay
    'ReplayRecorder',
    'ReplayPlayer',
    'save_episode',
    'load_episode',

    # Viewer
    'OmniverseViewer',
    'ViewerConfig',

    # Controls
    'SimulationController',
    'VehicleSpawner',
    'ArenaSelector',
]
