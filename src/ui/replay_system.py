"""
Replay System for FlyingCarRL

Handles recording, saving, loading, and playback of simulation episodes.
"""

import json
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime


@dataclass
class EpisodeMetadata:
    """Metadata for a recorded episode"""
    episode_id: str
    timestamp: str
    vehicle_type: str
    num_agents: int
    duration: float
    total_reward: float
    collisions: int
    altitude_violations: int
    arena: str
    success: bool
    notes: str = ""


@dataclass
class EpisodeFrame:
    """Single frame of episode data"""
    timestep: int
    positions: np.ndarray  # Shape: (n_agents, 3)
    velocities: np.ndarray  # Shape: (n_agents, 3)
    actions: np.ndarray  # Shape: (n_agents, action_dim)
    rewards: np.ndarray  # Shape: (n_agents,)
    battery_levels: np.ndarray  # Shape: (n_agents,)
    sensor_data: Optional[Dict[str, Any]] = None


class ReplayRecorder:
    """
    Records simulation episodes for later playback.
    """

    def __init__(
        self,
        save_dir: str = "replays",
        compress: bool = True,
        save_sensor_data: bool = False
    ):
        """
        Initialize replay recorder.

        Args:
            save_dir: Directory to save replay files
            compress: Whether to compress replay files
            save_sensor_data: Whether to save detailed sensor data (increases file size)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        self.save_sensor_data = save_sensor_data

        self.recording = False
        self.current_episode = None
        self.frames = []
        self.metadata = None

    def start_recording(
        self,
        episode_id: Optional[str] = None,
        vehicle_type: str = "medium",
        num_agents: int = 1,
        arena: str = "NYC_Manhattan",
        notes: str = ""
    ):
        """
        Start recording a new episode.

        Args:
            episode_id: Unique episode ID (auto-generated if None)
            vehicle_type: Type of vehicle being recorded
            num_agents: Number of agents in episode
            arena: Arena name
            notes: Optional notes about this episode
        """
        if self.recording:
            self.stop_recording()

        self.recording = True
        self.frames = []

        # Generate episode ID if not provided
        if episode_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_id = f"episode_{timestamp}"

        self.current_episode = episode_id

        # Initialize metadata
        self.metadata = EpisodeMetadata(
            episode_id=episode_id,
            timestamp=datetime.now().isoformat(),
            vehicle_type=vehicle_type,
            num_agents=num_agents,
            duration=0.0,
            total_reward=0.0,
            collisions=0,
            altitude_violations=0,
            arena=arena,
            success=False,
            notes=notes
        )

        print(f"Started recording episode: {episode_id}")

    def record_frame(
        self,
        timestep: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        battery_levels: np.ndarray,
        sensor_data: Optional[Dict[str, Any]] = None
    ):
        """
        Record a single frame of the episode.

        Args:
            timestep: Current timestep
            positions: Agent positions
            velocities: Agent velocities
            actions: Actions taken
            rewards: Rewards received
            battery_levels: Battery levels
            sensor_data: Optional sensor data
        """
        if not self.recording:
            raise RuntimeError("Not currently recording. Call start_recording() first.")

        # Create frame
        frame = EpisodeFrame(
            timestep=timestep,
            positions=positions.copy(),
            velocities=velocities.copy(),
            actions=actions.copy(),
            rewards=rewards.copy(),
            battery_levels=battery_levels.copy(),
            sensor_data=sensor_data if self.save_sensor_data else None
        )

        self.frames.append(frame)

        # Update metadata
        self.metadata.total_reward += rewards.sum()

    def stop_recording(
        self,
        success: bool = False,
        collisions: int = 0,
        altitude_violations: int = 0
    ) -> str:
        """
        Stop recording and save the episode.

        Args:
            success: Whether the episode was successful
            collisions: Number of collisions
            altitude_violations: Number of altitude violations

        Returns:
            Path to saved replay file
        """
        if not self.recording:
            return None

        self.recording = False

        # Update final metadata
        if len(self.frames) > 0:
            self.metadata.duration = self.frames[-1].timestep * 0.02  # Assuming 50Hz
        self.metadata.success = success
        self.metadata.collisions = collisions
        self.metadata.altitude_violations = altitude_violations

        # Save to file
        filepath = self.save_episode(
            self.current_episode,
            self.frames,
            self.metadata
        )

        print(f"Stopped recording. Saved to: {filepath}")
        print(f"  Duration: {self.metadata.duration:.2f}s")
        print(f"  Total Reward: {self.metadata.total_reward:.2f}")
        print(f"  Frames: {len(self.frames)}")

        # Reset
        self.frames = []
        self.metadata = None
        self.current_episode = None

        return filepath

    def save_episode(
        self,
        episode_id: str,
        frames: List[EpisodeFrame],
        metadata: EpisodeMetadata
    ) -> str:
        """
        Save episode to disk.

        Args:
            episode_id: Episode identifier
            frames: List of frames
            metadata: Episode metadata

        Returns:
            Path to saved file
        """
        # Prepare data
        episode_data = {
            'metadata': asdict(metadata),
            'frames': []
        }

        # Convert frames to serializable format
        for frame in frames:
            frame_dict = {
                'timestep': frame.timestep,
                'positions': frame.positions.tolist(),
                'velocities': frame.velocities.tolist(),
                'actions': frame.actions.tolist(),
                'rewards': frame.rewards.tolist(),
                'battery_levels': frame.battery_levels.tolist(),
                'sensor_data': frame.sensor_data
            }
            episode_data['frames'].append(frame_dict)

        # Save to file
        if self.compress:
            filepath = self.save_dir / f"{episode_id}.pkl.gz"
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(episode_data, f)
        else:
            filepath = self.save_dir / f"{episode_id}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(episode_data, f)

        # Also save metadata as JSON for easy inspection
        metadata_path = self.save_dir / f"{episode_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        return str(filepath)


class ReplayPlayer:
    """
    Plays back recorded episodes.
    """

    def __init__(self, replay_file: str):
        """
        Initialize replay player.

        Args:
            replay_file: Path to replay file
        """
        self.replay_file = Path(replay_file)
        self.episode_data = None
        self.metadata = None
        self.frames = None
        self.current_frame = 0

        self.load_episode()

    def load_episode(self):
        """Load episode from disk"""
        if not self.replay_file.exists():
            raise FileNotFoundError(f"Replay file not found: {self.replay_file}")

        # Load data
        if self.replay_file.suffix == '.gz':
            with gzip.open(self.replay_file, 'rb') as f:
                self.episode_data = pickle.load(f)
        else:
            with open(self.replay_file, 'rb') as f:
                self.episode_data = pickle.load(f)

        # Extract metadata and frames
        self.metadata = EpisodeMetadata(**self.episode_data['metadata'])

        # Convert frames back to EpisodeFrame objects
        self.frames = []
        for frame_dict in self.episode_data['frames']:
            frame = EpisodeFrame(
                timestep=frame_dict['timestep'],
                positions=np.array(frame_dict['positions']),
                velocities=np.array(frame_dict['velocities']),
                actions=np.array(frame_dict['actions']),
                rewards=np.array(frame_dict['rewards']),
                battery_levels=np.array(frame_dict['battery_levels']),
                sensor_data=frame_dict.get('sensor_data')
            )
            self.frames.append(frame)

        print(f"Loaded replay: {self.metadata.episode_id}")
        print(f"  Duration: {self.metadata.duration:.2f}s")
        print(f"  Frames: {len(self.frames)}")
        print(f"  Vehicle Type: {self.metadata.vehicle_type}")
        print(f"  Agents: {self.metadata.num_agents}")

    def get_frame(self, index: int) -> EpisodeFrame:
        """Get frame by index"""
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"Frame index {index} out of range [0, {len(self.frames)})")
        return self.frames[index]

    def get_current_frame(self) -> EpisodeFrame:
        """Get current frame"""
        return self.get_frame(self.current_frame)

    def next_frame(self) -> Optional[EpisodeFrame]:
        """Advance to next frame"""
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            return self.get_current_frame()
        return None

    def prev_frame(self) -> Optional[EpisodeFrame]:
        """Go to previous frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            return self.get_current_frame()
        return None

    def reset(self):
        """Reset playback to beginning"""
        self.current_frame = 0

    def get_trajectory(self, agent_id: int = 0) -> np.ndarray:
        """
        Get full trajectory for a specific agent.

        Args:
            agent_id: Agent index

        Returns:
            Array of shape (n_frames, 3) with positions
        """
        positions = np.array([frame.positions[agent_id] for frame in self.frames])
        return positions

    def get_all_trajectories(self) -> np.ndarray:
        """
        Get trajectories for all agents.

        Returns:
            Array of shape (n_agents, n_frames, 3)
        """
        n_agents = self.metadata.num_agents
        n_frames = len(self.frames)

        trajectories = np.zeros((n_agents, n_frames, 3))
        for i, frame in enumerate(self.frames):
            trajectories[:, i, :] = frame.positions

        return trajectories

    def get_rewards(self) -> np.ndarray:
        """
        Get rewards over time.

        Returns:
            Array of shape (n_frames, n_agents)
        """
        rewards = np.array([frame.rewards for frame in self.frames])
        return rewards

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get episode statistics.

        Returns:
            Dictionary with statistics
        """
        trajectories = self.get_all_trajectories()
        rewards = self.get_rewards()

        stats = {
            'episode_id': self.metadata.episode_id,
            'duration': self.metadata.duration,
            'total_reward': self.metadata.total_reward,
            'mean_reward': rewards.mean(),
            'max_reward': rewards.max(),
            'min_reward': rewards.min(),
            'num_frames': len(self.frames),
            'num_agents': self.metadata.num_agents,
            'vehicle_type': self.metadata.vehicle_type,
            'arena': self.metadata.arena,
            'success': self.metadata.success,
            'collisions': self.metadata.collisions,
            'altitude_violations': self.metadata.altitude_violations,
        }

        # Calculate distance traveled
        distances = []
        for agent_id in range(self.metadata.num_agents):
            traj = self.get_trajectory(agent_id)
            dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            distances.append(dist)

        stats['total_distance'] = sum(distances)
        stats['mean_distance_per_agent'] = np.mean(distances)

        return stats


# Convenience functions

def save_episode(
    episode_id: str,
    frames: List[EpisodeFrame],
    metadata: EpisodeMetadata,
    save_dir: str = "replays",
    compress: bool = True
) -> str:
    """
    Convenience function to save an episode.

    Args:
        episode_id: Episode identifier
        frames: List of frames
        metadata: Episode metadata
        save_dir: Directory to save to
        compress: Whether to compress

    Returns:
        Path to saved file
    """
    recorder = ReplayRecorder(save_dir=save_dir, compress=compress)
    return recorder.save_episode(episode_id, frames, metadata)


def load_episode(replay_file: str) -> ReplayPlayer:
    """
    Convenience function to load an episode.

    Args:
        replay_file: Path to replay file

    Returns:
        ReplayPlayer instance
    """
    return ReplayPlayer(replay_file)


def list_replays(replay_dir: str = "replays") -> List[Dict[str, Any]]:
    """
    List all available replays.

    Args:
        replay_dir: Directory containing replays

    Returns:
        List of replay metadata
    """
    replay_path = Path(replay_dir)
    if not replay_path.exists():
        return []

    replays = []
    for metadata_file in replay_path.glob("*_metadata.json"):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            replays.append(metadata)

    # Sort by timestamp (newest first)
    replays.sort(key=lambda x: x['timestamp'], reverse=True)

    return replays
