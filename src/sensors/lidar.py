"""
LiDAR Sensor Emulation

Simulates LiDAR (Light Detection and Ranging) sensor with point cloud generation
and realistic noise characteristics for eVTOL vehicles.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LiDARConfig:
    """
    LiDAR sensor configuration.

    Attributes:
        num_channels: Number of vertical channels (e.g., 16, 32, 64)
        horizontal_resolution: Number of horizontal points per rotation
        vertical_fov: Vertical field of view in degrees (min, max)
        horizontal_fov: Horizontal field of view in degrees
        max_range: Maximum detection range in meters
        min_range: Minimum detection range in meters
        rotation_frequency: Rotation frequency in Hz
        range_noise_std: Standard deviation of range noise in meters
        angle_noise_std: Standard deviation of angle noise in degrees
    """
    num_channels: int = 16
    horizontal_resolution: int = 360
    vertical_fov: Tuple[float, float] = (-15.0, 15.0)
    horizontal_fov: float = 360.0
    max_range: float = 100.0
    min_range: float = 0.5
    rotation_frequency: float = 10.0  # 10 Hz
    range_noise_std: float = 0.02  # 2cm noise
    angle_noise_std: float = 0.1  # 0.1 degree noise


class LiDARSensor:
    """
    LiDAR sensor with point cloud generation and noise injection.

    Simulates 3D point cloud data for autonomous vehicle perception.
    """

    def __init__(
        self,
        config: Optional[LiDARConfig] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize LiDAR sensor.

        Args:
            config: LiDAR configuration (default: LiDARConfig())
            random_seed: Random seed for reproducibility
        """
        self.config = config or LiDARConfig()
        self.rng = np.random.default_rng(random_seed)

        # Scan timing
        self.scan_time = 1.0 / self.config.rotation_frequency
        self.time_since_last_scan = 0.0

        # Pre-compute scan angles
        self._compute_scan_angles()

    def _compute_scan_angles(self) -> None:
        """Pre-compute vertical and horizontal scan angles."""
        # Vertical angles (channels)
        v_min, v_max = self.config.vertical_fov
        self.vertical_angles = np.linspace(
            v_min, v_max, self.config.num_channels, dtype=np.float32
        )

        # Horizontal angles (rotation)
        h_fov = self.config.horizontal_fov
        h_start = -h_fov / 2
        h_end = h_fov / 2
        self.horizontal_angles = np.linspace(
            h_start, h_end, self.config.horizontal_resolution, dtype=np.float32
        )

    def update(self, dt: float) -> bool:
        """
        Update sensor timing.

        Args:
            dt: Time step in seconds

        Returns:
            True if a new scan is ready, False otherwise
        """
        self.time_since_last_scan += dt
        if self.time_since_last_scan >= self.scan_time:
            self.time_since_last_scan = 0.0
            return True
        return False

    def capture_scan(
        self,
        scene_ranges: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture LiDAR scan and generate point cloud.

        Args:
            scene_ranges: Ground truth range data (num_channels, horizontal_resolution)
                         If None, generates synthetic scene

        Returns:
            Tuple of (point_cloud, intensities)
            - point_cloud: (N, 3) array of [x, y, z] points in sensor frame
            - intensities: (N,) array of intensity values [0, 1]
        """
        if scene_ranges is None:
            # Generate synthetic ranges
            scene_ranges = self._generate_synthetic_ranges()

        # Add noise to ranges
        noisy_ranges = self._add_range_noise(scene_ranges)

        # Add noise to angles
        noisy_v_angles = self.vertical_angles + self.rng.normal(
            0, self.config.angle_noise_std, size=self.config.num_channels
        )
        noisy_h_angles = self.horizontal_angles + self.rng.normal(
            0, self.config.angle_noise_std, size=self.config.horizontal_resolution
        )

        # Convert to point cloud
        point_cloud, intensities = self._ranges_to_pointcloud(
            noisy_ranges, noisy_v_angles, noisy_h_angles
        )

        return point_cloud, intensities

    def _add_range_noise(self, ranges: np.ndarray) -> np.ndarray:
        """
        Add realistic noise to range measurements.

        Args:
            ranges: Ground truth ranges (num_channels, horizontal_resolution)

        Returns:
            Noisy ranges with same shape
        """
        # Range-dependent noise (more noise at larger distances)
        noise_std = self.config.range_noise_std * (1 + ranges / self.config.max_range)
        noise = self.rng.normal(0, noise_std, size=ranges.shape)

        noisy_ranges = ranges + noise

        # Clip to valid range
        noisy_ranges = np.clip(
            noisy_ranges,
            self.config.min_range,
            self.config.max_range
        )

        return noisy_ranges.astype(np.float32)

    def _ranges_to_pointcloud(
        self,
        ranges: np.ndarray,
        vertical_angles: np.ndarray,
        horizontal_angles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert range and angle data to 3D point cloud.

        LiDAR coordinate system: X forward, Y left, Z up

        Args:
            ranges: Range measurements (num_channels, horizontal_resolution)
            vertical_angles: Vertical angles in degrees (num_channels,)
            horizontal_angles: Horizontal angles in degrees (horizontal_resolution,)

        Returns:
            Tuple of (points, intensities)
            - points: (N, 3) array of [x, y, z] coordinates
            - intensities: (N,) array of intensity values
        """
        # Convert angles to radians
        v_rad = np.deg2rad(vertical_angles)
        h_rad = np.deg2rad(horizontal_angles)

        # Meshgrid for vectorized computation
        V, H = np.meshgrid(v_rad, h_rad, indexing='ij')
        R = ranges

        # Spherical to Cartesian conversion
        # x = r * cos(v) * cos(h)  (forward)
        # y = r * cos(v) * sin(h)  (left)
        # z = r * sin(v)           (up)
        x = R * np.cos(V) * np.cos(H)
        y = R * np.cos(V) * np.sin(H)
        z = R * np.sin(V)

        # Flatten to point cloud
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

        # Filter invalid points (max range or no return)
        valid_mask = R.flatten() < self.config.max_range
        points = points[valid_mask]

        # Generate intensities (simplified: inverse distance)
        intensities = 1.0 - (R.flatten()[valid_mask] / self.config.max_range)
        intensities = np.clip(intensities, 0.0, 1.0).astype(np.float32)

        return points.astype(np.float32), intensities

    def _generate_synthetic_ranges(self) -> np.ndarray:
        """
        Generate synthetic range data for testing.

        Creates a simple scene with ground plane and random obstacles.

        Returns:
            Range array (num_channels, horizontal_resolution) in meters
        """
        ranges = np.full(
            (self.config.num_channels, self.config.horizontal_resolution),
            self.config.max_range,
            dtype=np.float32
        )

        # Add ground plane (distance depends on vertical angle)
        ground_height = 122.0  # 400 feet in meters (typical altitude)
        for i, v_angle in enumerate(self.vertical_angles):
            if v_angle < 0:  # Looking down
                v_rad = np.deg2rad(v_angle)
                ground_range = abs(ground_height / np.sin(v_rad))
                ground_range = min(ground_range, self.config.max_range)
                ranges[i, :] = ground_range

        # Add random obstacles (buildings, other vehicles)
        num_obstacles = self.rng.integers(5, 15)
        for _ in range(num_obstacles):
            # Random position and size
            h_center = self.rng.integers(0, self.config.horizontal_resolution)
            h_width = self.rng.integers(5, 30)
            v_center = self.rng.integers(0, self.config.num_channels)
            v_height = self.rng.integers(2, 8)

            # Random distance
            obstacle_range = self.rng.uniform(self.config.min_range + 5, self.config.max_range / 2)

            # Set obstacle ranges
            h_start = max(0, h_center - h_width // 2)
            h_end = min(self.config.horizontal_resolution, h_center + h_width // 2)
            v_start = max(0, v_center - v_height // 2)
            v_end = min(self.config.num_channels, v_center + v_height // 2)

            ranges[v_start:v_end, h_start:h_end] = np.minimum(
                ranges[v_start:v_end, h_start:h_end],
                obstacle_range
            )

        return ranges

    def get_scan_pattern_info(self) -> dict:
        """
        Get information about the LiDAR scan pattern.

        Returns:
            Dictionary with scan pattern details
        """
        return {
            'num_channels': self.config.num_channels,
            'horizontal_resolution': self.config.horizontal_resolution,
            'total_points_per_scan': self.config.num_channels * self.config.horizontal_resolution,
            'vertical_fov': self.config.vertical_fov,
            'horizontal_fov': self.config.horizontal_fov,
            'scan_rate_hz': self.config.rotation_frequency,
            'range': (self.config.min_range, self.config.max_range)
        }

    def __repr__(self) -> str:
        """String representation of LiDAR sensor."""
        return (
            f"LiDARSensor(channels={self.config.num_channels}, "
            f"h_res={self.config.horizontal_resolution}, "
            f"range={self.config.max_range}m, "
            f"freq={self.config.rotation_frequency}Hz)"
        )
