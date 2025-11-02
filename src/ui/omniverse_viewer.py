"""
NVIDIA Omniverse Viewer Integration for FlyingCarRL

Provides interface for visualizing simulations in Omniverse.
Note: Requires NVIDIA Omniverse to be installed separately.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ViewerConfig:
    """Configuration for Omniverse Viewer"""
    window_width: int = 1920
    window_height: int = 1080
    fps: int = 60
    enable_physics_debug: bool = False
    enable_wireframe: bool = False
    camera_mode: str = "free"  # free, follow, orbit
    follow_agent_id: Optional[int] = None
    enable_shadows: bool = True
    enable_ray_tracing: bool = False
    fog_enabled: bool = True
    fog_density: float = 0.001


class OmniverseViewer:
    """
    Interface for NVIDIA Omniverse visualization.

    This class provides a Python API for controlling the Omniverse viewport
    and rendering simulation episodes in 3D.

    Note: This is a simplified interface. Full Omniverse integration requires
    additional setup and the Isaac Lab extensions.
    """

    def __init__(
        self,
        config: Optional[ViewerConfig] = None,
        headless: bool = False
    ):
        """
        Initialize Omniverse viewer.

        Args:
            config: Viewer configuration
            headless: Run in headless mode (no GUI)
        """
        self.config = config or ViewerConfig()
        self.headless = headless
        self.initialized = False

        # Viewer state
        self.camera_position = np.array([0.0, 0.0, 500.0])
        self.camera_target = np.array([0.0, 0.0, 450.0])
        self.agents = {}
        self.arena_loaded = False

        # Mock Omniverse connection
        # In production, this would connect to actual Omniverse Kit
        self.stage = None
        self.viewport = None

    def initialize(self) -> bool:
        """
        Initialize Omniverse connection.

        Returns:
            True if successful
        """
        try:
            # In production, this would:
            # 1. Import omni.isaac.kit
            # 2. Create simulation app
            # 3. Initialize extensions
            # 4. Load USD stage

            print("Initializing Omniverse Viewer...")
            print(f"  Mode: {'Headless' if self.headless else 'Visual'}")
            print(f"  Resolution: {self.config.window_width}x{self.config.window_height}")
            print(f"  FPS: {self.config.fps}")

            # Mock initialization
            self.initialized = True
            print("✓ Viewer initialized (mock mode)")

            # Note: Actual implementation would look like:
            # from omni.isaac.kit import SimulationApp
            # self.simulation_app = SimulationApp({"headless": self.headless})
            # import omni.usd
            # self.stage = omni.usd.get_context().get_stage()

            return True

        except Exception as e:
            print(f"Failed to initialize Omniverse: {e}")
            print("This is expected if Omniverse is not installed.")
            print("For full 3D visualization, install NVIDIA Omniverse and Isaac Lab.")
            return False

    def load_arena(
        self,
        arena_name: str,
        terrain_file: Optional[str] = None
    ) -> bool:
        """
        Load arena/environment.

        Args:
            arena_name: Name of arena to load
            terrain_file: Optional USD file for terrain

        Returns:
            True if successful
        """
        if not self.initialized:
            print("Viewer not initialized. Call initialize() first.")
            return False

        print(f"Loading arena: {arena_name}")

        # In production, this would load USD assets:
        # - Terrain from Cesium
        # - Buildings and obstacles
        # - Skybox and lighting
        # - Ground plane with altitude markers

        self.arena_loaded = True
        print(f"✓ Arena '{arena_name}' loaded (mock mode)")

        return True

    def spawn_agent(
        self,
        agent_id: int,
        position: np.ndarray,
        vehicle_type: str = "medium",
        color: Optional[Tuple[float, float, float]] = None
    ) -> bool:
        """
        Spawn an eVTOL agent in the viewer.

        Args:
            agent_id: Unique agent identifier
            position: Initial position [x, y, z]
            vehicle_type: Type of vehicle
            color: RGB color (0-1 range)

        Returns:
            True if successful
        """
        if not self.arena_loaded:
            print("Arena not loaded. Call load_arena() first.")
            return False

        # Store agent info
        self.agents[agent_id] = {
            'position': position.copy(),
            'vehicle_type': vehicle_type,
            'color': color or (0.2, 0.5, 0.8),
            'active': True
        }

        print(f"✓ Spawned agent {agent_id} at {position}")

        # In production, this would:
        # - Load vehicle USD asset based on type
        # - Set initial transform
        # - Apply materials/colors
        # - Enable physics

        return True

    def update_agent(
        self,
        agent_id: int,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None
    ):
        """
        Update agent state.

        Args:
            agent_id: Agent identifier
            position: New position
            velocity: New velocity (for motion blur)
            rotation: New rotation (quaternion or euler)
        """
        if agent_id not in self.agents:
            return

        self.agents[agent_id]['position'] = position.copy()

        if velocity is not None:
            self.agents[agent_id]['velocity'] = velocity.copy()

        if rotation is not None:
            self.agents[agent_id]['rotation'] = rotation.copy()

        # In production, update USD prim transform

    def remove_agent(self, agent_id: int):
        """Remove agent from viewer"""
        if agent_id in self.agents:
            self.agents[agent_id]['active'] = False
            # In production, delete or hide USD prim

    def set_camera(
        self,
        position: np.ndarray,
        target: np.ndarray,
        mode: str = "free"
    ):
        """
        Set camera position and target.

        Args:
            position: Camera position
            target: Camera look-at target
            mode: Camera mode (free, follow, orbit)
        """
        self.camera_position = position.copy()
        self.camera_target = target.copy()
        self.config.camera_mode = mode

        # In production, update viewport camera

    def follow_agent(self, agent_id: int, offset: Optional[np.ndarray] = None):
        """
        Set camera to follow an agent.

        Args:
            agent_id: Agent to follow
            offset: Camera offset from agent
        """
        if agent_id not in self.agents:
            return

        self.config.camera_mode = "follow"
        self.config.follow_agent_id = agent_id

        if offset is None:
            offset = np.array([-50.0, 0.0, 30.0])

        agent_pos = self.agents[agent_id]['position']
        self.set_camera(agent_pos + offset, agent_pos, mode="follow")

    def render_frame(self) -> Optional[np.ndarray]:
        """
        Render current frame.

        Returns:
            RGB image as numpy array (or None in headless mode)
        """
        if self.headless:
            return None

        # In production, capture viewport render
        # For now, return mock image dimensions
        mock_image = np.zeros((
            self.config.window_height,
            self.config.window_width,
            3
        ), dtype=np.uint8)

        return mock_image

    def highlight_agent(
        self,
        agent_id: int,
        highlight: bool = True,
        color: Tuple[float, float, float] = (1.0, 1.0, 0.0)
    ):
        """
        Highlight an agent (e.g., for debugging).

        Args:
            agent_id: Agent to highlight
            highlight: Enable/disable highlight
            color: Highlight color
        """
        if agent_id in self.agents:
            self.agents[agent_id]['highlighted'] = highlight
            if highlight:
                self.agents[agent_id]['highlight_color'] = color

    def draw_trajectory(
        self,
        positions: np.ndarray,
        color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        width: float = 2.0
    ):
        """
        Draw trajectory path.

        Args:
            positions: Array of positions (n_points, 3)
            color: Line color
            width: Line width
        """
        # In production, create curve or line primitive in USD
        pass

    def add_marker(
        self,
        position: np.ndarray,
        label: str = "",
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        size: float = 5.0
    ):
        """
        Add visual marker.

        Args:
            position: Marker position
            label: Optional text label
            color: Marker color
            size: Marker size
        """
        # In production, create sphere or cone primitive
        pass

    def clear_markers(self):
        """Clear all markers"""
        pass

    def enable_physics_debug(self, enable: bool = True):
        """Enable/disable physics debug visualization"""
        self.config.enable_physics_debug = enable

    def enable_wireframe(self, enable: bool = True):
        """Enable/disable wireframe rendering"""
        self.config.enable_wireframe = enable

    def take_screenshot(self, filepath: str) -> bool:
        """
        Take screenshot.

        Args:
            filepath: Path to save screenshot

        Returns:
            True if successful
        """
        if self.headless:
            print("Cannot take screenshot in headless mode")
            return False

        # In production, capture and save viewport render
        print(f"Screenshot saved to: {filepath}")
        return True

    def get_agent_info(self, agent_id: int) -> Optional[Dict]:
        """Get agent information"""
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List[int]:
        """Get list of all active agent IDs"""
        return [aid for aid, info in self.agents.items() if info.get('active', False)]

    def shutdown(self):
        """Shutdown viewer and cleanup"""
        print("Shutting down Omniverse Viewer...")

        # Clear all agents
        self.agents.clear()

        # In production, cleanup Omniverse resources:
        # - Close USD stage
        # - Shutdown simulation app
        # self.simulation_app.close()

        self.initialized = False
        print("✓ Viewer shutdown complete")

    def save_config(self, filepath: str):
        """Save viewer configuration"""
        config_dict = {
            'window_width': self.config.window_width,
            'window_height': self.config.window_height,
            'fps': self.config.fps,
            'enable_physics_debug': self.config.enable_physics_debug,
            'enable_wireframe': self.config.enable_wireframe,
            'camera_mode': self.config.camera_mode,
            'enable_shadows': self.config.enable_shadows,
            'enable_ray_tracing': self.config.enable_ray_tracing,
            'fog_enabled': self.config.fog_enabled,
            'fog_density': self.config.fog_density,
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, filepath: str) -> ViewerConfig:
        """Load viewer configuration"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return ViewerConfig(**config_dict)


# Helper functions

def create_viewer(
    headless: bool = False,
    config: Optional[ViewerConfig] = None
) -> OmniverseViewer:
    """
    Create and initialize an Omniverse viewer.

    Args:
        headless: Run in headless mode
        config: Optional viewer configuration

    Returns:
        Initialized OmniverseViewer instance
    """
    viewer = OmniverseViewer(config=config, headless=headless)
    viewer.initialize()
    return viewer


def visualize_replay(
    replay_file: str,
    viewer: Optional[OmniverseViewer] = None,
    playback_speed: float = 1.0
) -> bool:
    """
    Visualize a replay file in Omniverse.

    Args:
        replay_file: Path to replay file
        viewer: Optional viewer instance (creates new if None)
        playback_speed: Playback speed multiplier

    Returns:
        True if successful
    """
    from .replay_system import load_episode

    # Load replay
    player = load_episode(replay_file)

    # Create viewer if needed
    if viewer is None:
        viewer = create_viewer(headless=False)

    # Load arena
    viewer.load_arena(player.metadata.arena)

    # Spawn agents
    first_frame = player.get_frame(0)
    n_agents = player.metadata.num_agents

    for i in range(n_agents):
        viewer.spawn_agent(
            agent_id=i,
            position=first_frame.positions[i],
            vehicle_type=player.metadata.vehicle_type
        )

    # Playback loop
    print(f"Playing back episode: {player.metadata.episode_id}")
    print(f"  Duration: {player.metadata.duration:.2f}s")
    print(f"  Speed: {playback_speed}x")

    player.reset()
    frame = player.get_current_frame()

    while frame is not None:
        # Update agent positions
        for i in range(n_agents):
            viewer.update_agent(
                agent_id=i,
                position=frame.positions[i],
                velocity=frame.velocities[i]
            )

        # Render
        viewer.render_frame()

        # Next frame
        frame = player.next_frame()

    print("✓ Playback complete")
    return True
