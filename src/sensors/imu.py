"""
IMU Sensor Emulation

Simulates Inertial Measurement Unit (IMU) with accelerometer and gyroscope
sensors including realistic noise characteristics for eVTOL vehicles.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class IMUConfig:
    """
    IMU sensor configuration.

    Attributes:
        sample_rate: Sampling rate in Hz
        accel_noise_std: Accelerometer noise standard deviation in m/s²
        accel_bias_std: Accelerometer bias standard deviation in m/s²
        gyro_noise_std: Gyroscope noise standard deviation in rad/s
        gyro_bias_std: Gyroscope bias standard deviation in rad/s
        accel_range: Maximum acceleration range in m/s² (±range)
        gyro_range: Maximum angular velocity range in rad/s (±range)
    """
    sample_rate: int = 100  # 100 Hz typical for IMU
    accel_noise_std: float = 0.05  # m/s² noise
    accel_bias_std: float = 0.01  # m/s² bias
    gyro_noise_std: float = 0.01  # rad/s noise
    gyro_bias_std: float = 0.005  # rad/s bias
    accel_range: float = 156.8  # ±16g (156.8 m/s²)
    gyro_range: float = 34.9  # ±2000 deg/s (34.9 rad/s)


class IMUSensor:
    """
    IMU sensor with accelerometer and gyroscope measurements.

    Simulates 6-DOF inertial measurements with realistic noise and bias.
    """

    GRAVITY = 9.81  # m/s²

    def __init__(
        self,
        config: Optional[IMUConfig] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize IMU sensor.

        Args:
            config: IMU configuration (default: IMUConfig())
            random_seed: Random seed for reproducibility
        """
        self.config = config or IMUConfig()
        self.rng = np.random.default_rng(random_seed)

        # Sample timing
        self.sample_time = 1.0 / self.config.sample_rate
        self.time_since_last_sample = 0.0

        # Initialize sensor biases (constant for each sensor instance)
        self.accel_bias = self.rng.normal(
            0, self.config.accel_bias_std, size=3
        ).astype(np.float32)
        self.gyro_bias = self.rng.normal(
            0, self.config.gyro_bias_std, size=3
        ).astype(np.float32)

    def update(self, dt: float) -> bool:
        """
        Update sensor timing.

        Args:
            dt: Time step in seconds

        Returns:
            True if a new sample is ready, False otherwise
        """
        self.time_since_last_sample += dt
        if self.time_since_last_sample >= self.sample_time:
            self.time_since_last_sample = 0.0
            return True
        return False

    def measure_acceleration(
        self,
        true_acceleration: np.ndarray,
        orientation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Measure acceleration with noise and bias.

        Accelerometer measures specific force (acceleration - gravity in body frame).

        Args:
            true_acceleration: True acceleration vector [ax, ay, az] in world frame (m/s²)
            orientation: Vehicle orientation [roll, pitch, yaw] in radians (optional)

        Returns:
            Measured acceleration [ax, ay, az] in body frame (m/s²)
        """
        # Convert to body frame if orientation provided
        if orientation is not None:
            true_accel_body = self._world_to_body(true_acceleration, orientation)
            gravity_body = self._world_to_body(
                np.array([0, 0, -self.GRAVITY], dtype=np.float32),
                orientation
            )
        else:
            # Assume body frame = world frame
            true_accel_body = true_acceleration
            gravity_body = np.array([0, 0, -self.GRAVITY], dtype=np.float32)

        # Accelerometer measures specific force (a - g)
        specific_force = true_accel_body - gravity_body

        # Add bias
        biased_measurement = specific_force + self.accel_bias

        # Add noise
        noise = self.rng.normal(0, self.config.accel_noise_std, size=3)
        noisy_measurement = biased_measurement + noise

        # Clip to sensor range
        noisy_measurement = np.clip(
            noisy_measurement,
            -self.config.accel_range,
            self.config.accel_range
        )

        return noisy_measurement.astype(np.float32)

    def measure_angular_velocity(
        self,
        true_angular_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Measure angular velocity with noise and bias.

        Args:
            true_angular_velocity: True angular velocity [wx, wy, wz] in body frame (rad/s)

        Returns:
            Measured angular velocity [wx, wy, wz] in body frame (rad/s)
        """
        # Add bias
        biased_measurement = true_angular_velocity + self.gyro_bias

        # Add noise
        noise = self.rng.normal(0, self.config.gyro_noise_std, size=3)
        noisy_measurement = biased_measurement + noise

        # Clip to sensor range
        noisy_measurement = np.clip(
            noisy_measurement,
            -self.config.gyro_range,
            self.config.gyro_range
        )

        return noisy_measurement.astype(np.float32)

    def measure_full_state(
        self,
        acceleration: np.ndarray,
        angular_velocity: np.ndarray,
        orientation: Optional[np.ndarray] = None
    ) -> dict:
        """
        Measure full IMU state (accelerometer + gyroscope).

        Args:
            acceleration: True acceleration [ax, ay, az] in world frame (m/s²)
            angular_velocity: True angular velocity [wx, wy, wz] in body frame (rad/s)
            orientation: Vehicle orientation [roll, pitch, yaw] in radians (optional)

        Returns:
            Dictionary with IMU measurements
        """
        accel = self.measure_acceleration(acceleration, orientation)
        gyro = self.measure_angular_velocity(angular_velocity)

        return {
            'accelerometer': accel,
            'gyroscope': gyro,
            'timestamp': 0.0  # Would be actual time in real system
        }

    def _world_to_body(
        self,
        vector_world: np.ndarray,
        orientation: np.ndarray
    ) -> np.ndarray:
        """
        Transform vector from world frame to body frame.

        Uses ZYX Euler angles (roll, pitch, yaw).

        Args:
            vector_world: Vector in world frame [x, y, z]
            orientation: Orientation [roll, pitch, yaw] in radians

        Returns:
            Vector in body frame [x, y, z]
        """
        roll, pitch, yaw = orientation

        # Rotation matrices
        # Roll (rotation about X-axis)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ], dtype=np.float32)

        # Pitch (rotation about Y-axis)
        R_y = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ], dtype=np.float32)

        # Yaw (rotation about Z-axis)
        R_z = np.array([
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Combined rotation: R = R_z * R_y * R_x
        R = R_z @ R_y @ R_x

        # Transform to body frame (inverse rotation)
        vector_body = R.T @ vector_world

        return vector_body

    def calibrate(self, num_samples: int = 100) -> dict:
        """
        Simulate IMU calibration to estimate biases.

        In real systems, this would be done while the IMU is stationary.

        Args:
            num_samples: Number of samples to average for calibration

        Returns:
            Dictionary with estimated biases
        """
        # Simulate stationary measurements (zero motion + gravity)
        accel_samples = []
        gyro_samples = []

        for _ in range(num_samples):
            # Stationary: a = 0, w = 0, gravity = -g in z
            accel = self.measure_acceleration(
                np.array([0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0], dtype=np.float32)
            )
            gyro = self.measure_angular_velocity(
                np.array([0, 0, 0], dtype=np.float32)
            )

            accel_samples.append(accel)
            gyro_samples.append(gyro)

        # Estimate biases (mean of measurements)
        accel_samples = np.array(accel_samples)
        gyro_samples = np.array(gyro_samples)

        # For accelerometer, we expect [0, 0, g] when stationary
        expected_accel = np.array([0, 0, self.GRAVITY], dtype=np.float32)
        estimated_accel_bias = np.mean(accel_samples, axis=0) - expected_accel

        # For gyroscope, we expect [0, 0, 0] when stationary
        estimated_gyro_bias = np.mean(gyro_samples, axis=0)

        return {
            'accelerometer_bias': estimated_accel_bias,
            'gyroscope_bias': estimated_gyro_bias,
            'num_samples': num_samples
        }

    def get_config_info(self) -> dict:
        """
        Get IMU configuration information.

        Returns:
            Dictionary with IMU specs
        """
        return {
            'sample_rate_hz': self.config.sample_rate,
            'accelerometer': {
                'range_m_s2': self.config.accel_range,
                'noise_std_m_s2': self.config.accel_noise_std,
                'bias_std_m_s2': self.config.accel_bias_std
            },
            'gyroscope': {
                'range_rad_s': self.config.gyro_range,
                'range_deg_s': np.rad2deg(self.config.gyro_range),
                'noise_std_rad_s': self.config.gyro_noise_std,
                'bias_std_rad_s': self.config.gyro_bias_std
            }
        }

    def reset_biases(self) -> None:
        """Reset sensor biases to new random values."""
        self.accel_bias = self.rng.normal(
            0, self.config.accel_bias_std, size=3
        ).astype(np.float32)
        self.gyro_bias = self.rng.normal(
            0, self.config.gyro_bias_std, size=3
        ).astype(np.float32)

    def __repr__(self) -> str:
        """String representation of IMU sensor."""
        return (
            f"IMUSensor(rate={self.config.sample_rate}Hz, "
            f"accel_range=±{self.config.accel_range:.1f}m/s², "
            f"gyro_range=±{np.rad2deg(self.config.gyro_range):.0f}deg/s)"
        )
