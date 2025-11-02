"""
Visualization utilities for FlyingCarRL

Provides plotting functions for trajectories, metrics, and real-time monitoring.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import pandas as pd


def plot_trajectories(
    positions: np.ndarray,
    vehicle_ids: Optional[List[int]] = None,
    colors: Optional[List[str]] = None,
    title: str = "eVTOL Trajectories",
    show_altitude_bounds: bool = True
) -> go.Figure:
    """
    Plot 3D trajectories of eVTOL vehicles.

    Args:
        positions: Array of shape (n_vehicles, n_timesteps, 3) with XYZ coordinates
        vehicle_ids: List of vehicle IDs for labeling
        colors: List of colors for each vehicle
        title: Plot title
        show_altitude_bounds: Whether to show 400-500ft altitude bounds

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    n_vehicles = positions.shape[0]
    if vehicle_ids is None:
        vehicle_ids = list(range(n_vehicles))

    if colors is None:
        colors = px.colors.qualitative.Plotly * (n_vehicles // 10 + 1)

    # Plot each vehicle's trajectory
    for i, (traj, vid) in enumerate(zip(positions, vehicle_ids)):
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines+markers',
            name=f'Vehicle {vid}',
            line=dict(color=colors[i], width=2),
            marker=dict(size=2),
            hovertemplate=(
                f'Vehicle {vid}<br>'
                'X: %{x:.2f}<br>'
                'Y: %{y:.2f}<br>'
                'Z (altitude): %{z:.2f} ft<br>'
                '<extra></extra>'
            )
        ))

        # Add start and end markers
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            z=[traj[0, 2]],
            mode='markers',
            name=f'Start {vid}',
            marker=dict(size=8, color='green', symbol='diamond'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter3d(
            x=[traj[-1, 0]],
            y=[traj[-1, 1]],
            z=[traj[-1, 2]],
            mode='markers',
            name=f'End {vid}',
            marker=dict(size=8, color='red', symbol='square'),
            showlegend=False
        ))

    # Add altitude bounds
    if show_altitude_bounds:
        # Create a semi-transparent plane at 400 and 500 ft
        x_range = [positions[:, :, 0].min(), positions[:, :, 0].max()]
        y_range = [positions[:, :, 1].min(), positions[:, :, 1].max()]

        # Lower bound (400 ft)
        fig.add_trace(go.Surface(
            x=x_range,
            y=y_range,
            z=np.ones((2, 2)) * 400,
            opacity=0.1,
            colorscale='Reds',
            showscale=False,
            name='400 ft bound',
            hoverinfo='skip'
        ))

        # Upper bound (500 ft)
        fig.add_trace(go.Surface(
            x=x_range,
            y=y_range,
            z=np.ones((2, 2)) * 500,
            opacity=0.1,
            colorscale='Blues',
            showscale=False,
            name='500 ft bound',
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Altitude (feet)',
            aspectmode='auto'
        ),
        showlegend=True,
        height=700
    )

    return fig


def plot_rewards(
    rewards: np.ndarray,
    window_size: int = 100,
    title: str = "Training Rewards"
) -> go.Figure:
    """
    Plot training rewards with moving average.

    Args:
        rewards: Array of episode rewards
        window_size: Window size for moving average
        title: Plot title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    episodes = np.arange(len(rewards))

    # Raw rewards
    fig.add_trace(go.Scatter(
        x=episodes,
        y=rewards,
        mode='markers',
        name='Episode Reward',
        marker=dict(size=3, opacity=0.5),
        hovertemplate='Episode: %{x}<br>Reward: %{y:.2f}<extra></extra>'
    ))

    # Moving average
    if len(rewards) >= window_size:
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        fig.add_trace(go.Scatter(
            x=episodes,
            y=moving_avg,
            mode='lines',
            name=f'{window_size}-Episode MA',
            line=dict(color='red', width=2),
            hovertemplate='Episode: %{x}<br>Avg Reward: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Episode',
        yaxis_title='Reward',
        hovermode='closest',
        height=500
    )

    return fig


def plot_metrics_dashboard(
    metrics: Dict[str, np.ndarray],
    episode_range: Optional[Tuple[int, int]] = None
) -> go.Figure:
    """
    Create a comprehensive metrics dashboard.

    Args:
        metrics: Dictionary with keys like 'rewards', 'collisions', 'altitude_violations', etc.
        episode_range: Optional (start, end) episode range to display

    Returns:
        Plotly figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Episode Rewards',
            'Collision Rate',
            'Altitude Compliance',
            'Episode Length'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    if episode_range:
        start, end = episode_range
        for key in metrics:
            metrics[key] = metrics[key][start:end]

    episodes = np.arange(len(metrics.get('rewards', [])))

    # Rewards
    if 'rewards' in metrics:
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=metrics['rewards'],
                mode='lines',
                name='Rewards',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

    # Collision rate
    if 'collisions' in metrics:
        collision_rate = pd.Series(metrics['collisions']).rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=collision_rate * 100,  # Convert to percentage
                mode='lines',
                name='Collision %',
                line=dict(color='red')
            ),
            row=1, col=2
        )

    # Altitude compliance
    if 'altitude_violations' in metrics:
        compliance = (1 - np.array(metrics['altitude_violations'])) * 100
        compliance_avg = pd.Series(compliance).rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=compliance_avg,
                mode='lines',
                name='Compliance %',
                line=dict(color='green')
            ),
            row=2, col=1
        )

    # Episode length
    if 'episode_lengths' in metrics:
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=metrics['episode_lengths'],
                mode='lines',
                name='Length',
                line=dict(color='purple')
            ),
            row=2, col=2
        )

    # Update axes
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Compliance (%)", row=2, col=1)
    fig.update_yaxes(title_text="Steps", row=2, col=2)

    fig.update_layout(
        title_text="Training Metrics Dashboard",
        showlegend=False,
        height=800
    )

    return fig


def create_3d_trajectory_plot(
    positions: np.ndarray,
    timestamps: np.ndarray,
    vehicle_type: str = "medium",
    show_velocity: bool = True
) -> go.Figure:
    """
    Create detailed 3D trajectory plot with velocity vectors.

    Args:
        positions: Array of shape (n_timesteps, 3) with XYZ coordinates
        timestamps: Array of timestamps
        vehicle_type: Type of vehicle ('small', 'medium', 'large')
        show_velocity: Whether to show velocity vectors

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Main trajectory
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines',
        name='Trajectory',
        line=dict(
            color=timestamps,
            colorscale='Viridis',
            width=4,
            colorbar=dict(title="Time (s)")
        ),
        hovertemplate=(
            'Time: %{marker.color:.2f}s<br>'
            'X: %{x:.2f} m<br>'
            'Y: %{y:.2f} m<br>'
            'Z: %{z:.2f} ft<br>'
            '<extra></extra>'
        )
    ))

    # Velocity vectors
    if show_velocity and len(positions) > 1:
        # Calculate velocity vectors
        velocities = np.diff(positions, axis=0)
        dt = np.diff(timestamps)
        velocities = velocities / dt[:, np.newaxis]

        # Sample every N points to avoid clutter
        sample_rate = max(1, len(positions) // 20)
        sampled_pos = positions[:-1:sample_rate]
        sampled_vel = velocities[::sample_rate]

        # Normalize for visualization
        vel_scale = 10.0
        vel_arrows = sampled_vel / (np.linalg.norm(sampled_vel, axis=1, keepdims=True) + 1e-6) * vel_scale

        for pos, vel in zip(sampled_pos, vel_arrows):
            fig.add_trace(go.Scatter3d(
                x=[pos[0], pos[0] + vel[0]],
                y=[pos[1], pos[1] + vel[1]],
                z=[pos[2], pos[2] + vel[2]],
                mode='lines',
                line=dict(color='orange', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Start and end markers
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]],
        y=[positions[0, 1]],
        z=[positions[0, 2]],
        mode='markers+text',
        name='Start',
        marker=dict(size=12, color='green', symbol='diamond'),
        text=['START'],
        textposition='top center'
    ))

    fig.add_trace(go.Scatter3d(
        x=[positions[-1, 0]],
        y=[positions[-1, 1]],
        z=[positions[-1, 2]],
        mode='markers+text',
        name='End',
        marker=dict(size=12, color='red', symbol='square'),
        text=['END'],
        textposition='top center'
    ))

    fig.update_layout(
        title=f"eVTOL Trajectory - {vehicle_type.capitalize()} Vehicle",
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Altitude (feet)',
            aspectmode='auto',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        height=800
    )

    return fig


def plot_energy_consumption(
    timestamps: np.ndarray,
    battery_levels: np.ndarray,
    thrust_values: np.ndarray
) -> go.Figure:
    """
    Plot energy consumption over time.

    Args:
        timestamps: Array of timestamps
        battery_levels: Battery percentage over time
        thrust_values: Thrust values over time

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Battery Level', 'Thrust Output'),
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # Battery level
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=battery_levels * 100,
            mode='lines',
            name='Battery %',
            line=dict(color='green', width=2),
            fill='tozeroy'
        ),
        row=1, col=1
    )

    # Thrust
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=thrust_values,
            mode='lines',
            name='Thrust',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Battery (%)", row=1, col=1)
    fig.update_yaxes(title_text="Thrust (N)", row=2, col=1)

    fig.update_layout(
        title_text="Energy Consumption Analysis",
        showlegend=False,
        height=600
    )

    return fig
