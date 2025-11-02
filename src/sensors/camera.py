"""
Camera Sensor Emulation

Simulates RGB and depth camera sensors with realistic noise characteristics
for eVTOL vehicles.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CameraConfig:
    """
    Camera sensor configuration.

    Attributes:
        resolution: (width, height) in pixels
        fov: Field of view in degrees
        fps: Frames per second
        noise_std: Standard deviation of Gaussian noise (0-1 normalized)
        depth_range: (min_depth, max_depth) in meters
    """
    resolution: Tuple[int, int] = (640, 480)
    fov: float = 90.0
    fps: int = 30
    noise_std: float = 0.02
    depth_range: Tuple[float, float] = (0.5, 100.0)


class CameraSensor:
    """
    RGB and Depth camera sensor with noise injection.

    Simulates camera data acquisition for autonomous vehicle perception.
    """

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize camera sensor.

        Args:
            config: Camera configuration (default: CameraConfig())
            random_seed: Random seed for reproducibility
        """
        self.config = config or CameraConfig()
        self.rng = np.random.default_rng(random_seed)

        # Frame timing
        self.frame_time = 1.0 / self.config.fps
        self.time_since_last_frame = 0.0

    def update(self, dt: float) -> bool:
        """
        Update sensor timing.

        Args:
            dt: Time step in seconds

        Returns:
            True if a new frame is ready, False otherwise
        """
        self.time_since_last_frame += dt
        if self.time_since_last_frame >= self.frame_time:
            self.time_since_last_frame = 0.0
            return True
        return False

    def capture_rgb(self, scene_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Capture RGB image with noise.

        Args:
            scene_image: Ground truth scene image (if None, generates synthetic)

        Returns:
            RGB image array (H, W, 3) with values in [0, 1]
        """
        width, height = self.config.resolution

        if scene_image is None:
            # Generate synthetic scene (placeholder for real rendering)
            scene_image = self._generate_synthetic_rgb(width, height)

        # Add Gaussian noise
        noise = self.rng.normal(0, self.config.noise_std, scene_image.shape)
        noisy_image = scene_image + noise

        # Clip to valid range [0, 1]
        noisy_image = np.clip(noisy_image, 0.0, 1.0).astype(np.float32)

        return noisy_image

    def capture_depth(self, scene_depth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Capture depth image with noise.

        Args:
            scene_depth: Ground truth depth map in meters (if None, generates synthetic)

        Returns:
            Depth image array (H, W) with values in meters
        """
        width, height = self.config.resolution

        if scene_depth is None:
            # Generate synthetic depth (placeholder for real rendering)
            scene_depth = self._generate_synthetic_depth(width, height)

        # Add depth-dependent noise (more noise at larger distances)
        depth_noise_std = self.config.noise_std * scene_depth / 10.0
        noise = self.rng.normal(0, depth_noise_std, scene_depth.shape)
        noisy_depth = scene_depth + noise

        # Clip to valid range
        min_depth, max_depth = self.config.depth_range
        noisy_depth = np.clip(noisy_depth, min_depth, max_depth).astype(np.float32)

        return noisy_depth

    def _generate_synthetic_rgb(self, width: int, height: int) -> np.ndarray:
        """
        Generate synthetic RGB scene (placeholder).

        In real implementation, this would raycast/render the 3D scene.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Synthetic RGB image (H, W, 3)
        """
        # Gradient sky + ground scene
        image = np.zeros((height, width, 3), dtype=np.float32)

        # Sky gradient (blue)
        for y in range(height // 2):
            intensity = 0.3 + 0.5 * (1 - y / (height // 2))
            image[y, :, 2] = intensity  # Blue channel

        # Ground (gray-green)
        for y in range(height // 2, height):
            image[y, :, 0] = 0.2  # Red
            image[y, :, 1] = 0.3  # Green
            image[y, :, 2] = 0.2  # Blue

        # Add random buildings/obstacles
        num_obstacles = self.rng.integers(3, 8)
        for _ in range(num_obstacles):
            x_start = self.rng.integers(0, width - 50)
            y_start = self.rng.integers(height // 2, height - 50)
            w = self.rng.integers(20, 80)
            h = self.rng.integers(30, 100)

            x_end = min(x_start + w, width)
            y_end = min(y_start + h, height)

            # Random building color (gray-ish)
            color = self.rng.uniform(0.4, 0.7, size=3)
            image[y_start:y_end, x_start:x_end] = color

        return image

    def _generate_synthetic_depth(self, width: int, height: int) -> np.ndarray:
        """
        Generate synthetic depth map (placeholder).

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Synthetic depth map (H, W) in meters
        """
        min_depth, max_depth = self.config.depth_range

        # Distance increases with vertical position (ground is closer at bottom)
        depth = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            # Sky is far, ground is closer
            distance = min_depth + (max_depth - min_depth) * (y / height)
            depth[y, :] = distance

        # Add random obstacles at closer distances
        num_obstacles = self.rng.integers(3, 8)
        for _ in range(num_obstacles):
            x_start = self.rng.integers(0, width - 50)
            y_start = self.rng.integers(height // 2, height - 50)
            w = self.rng.integers(20, 80)
            h = self.rng.integers(30, 100)

            x_end = min(x_start + w, width)
            y_end = min(y_start + h, height)

            obstacle_depth = self.rng.uniform(min_depth + 5, max_depth / 2)
            depth[y_start:y_end, x_start:x_end] = obstacle_depth

        return depth

    def get_intrinsics(self) -> np.ndarray:
        """
        Get camera intrinsic matrix.

        Returns:
            3x3 intrinsic matrix K
        """
        width, height = self.config.resolution
        fov_rad = np.deg2rad(self.config.fov)

        # Compute focal length from FOV
        focal_length = (width / 2) / np.tan(fov_rad / 2)

        # Intrinsic matrix
        K = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def __repr__(self) -> str:
        """String representation of camera sensor."""
        return (
            f"CameraSensor(resolution={self.config.resolution}, "
            f"fov={self.config.fov}°, fps={self.config.fps})"
        )
