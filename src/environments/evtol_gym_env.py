"""
FlyingCar RL Gymnasium Environment
Wraps the eVTOL simulation as a Gym-compatible RL environment.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class EVTOLEnv(gym.Env):
    """
    Single-agent Gymnasium environment for eVTOL training.

    Observation Space:
        - Position (x, y, z): 3D coordinates in simulation space
        - Velocity (vx, vy, vz): 3D velocity vector
        - Orientation (roll, pitch, yaw): Euler angles
        - Angular velocity (wx, wy, wz): Angular velocity components
        - Sensors: LiDAR distances (8 rays), GPS noise, IMU readings
        - Battery level: 0-1 normalized
        - Time remaining: normalized episode time
        Total: 30 dimensions

    Action Space:
        - Thrust: 0-1 (normalized throttle)
        - Roll: -1 to 1 (bank angle control)
        - Pitch: -1 to 1 (nose up/down)
        - Yaw: -1 to 1 (rotation control)
        Total: 4 continuous actions

    Reward Function:
        - Altitude compliance: +1.0 for staying in 400-500ft range
        - Collision penalty: -10.0 for hitting obstacles
        - Energy efficiency: +0.1 for efficient flight
        - Goal reaching: +5.0 for reaching waypoints
        - Out of bounds: -5.0 for leaving arena
        - Time penalty: -0.01 per step (encourage fast completion)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        vehicle_type: str = "medium",
        max_steps: int = 1000,
        altitude_min: float = 400.0,
        altitude_max: float = 500.0,
        arena_bounds: Tuple[float, float, float] = (1000.0, 1000.0, 100.0),
        enable_wind: bool = True,
        enable_sensor_noise: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the eVTOL Gym environment.

        Args:
            vehicle_type: Type of vehicle ("small", "medium", "large")
            max_steps: Maximum steps per episode
            altitude_min: Minimum allowed altitude (feet)
            altitude_max: Maximum allowed altitude (feet)
            arena_bounds: (x, y, z) bounds of the arena in meters
            enable_wind: Whether to simulate wind turbulence
            enable_sensor_noise: Whether to add noise to sensor readings
            render_mode: Rendering mode (None, "human", "rgb_array")
        """
        super().__init__()

        self.vehicle_type = vehicle_type
        self.max_steps = max_steps
        self.altitude_min = altitude_min
        self.altitude_max = altitude_max
        self.arena_bounds = np.array(arena_bounds)
        self.enable_wind = enable_wind
        self.enable_sensor_noise = enable_sensor_noise
        self.render_mode = render_mode

        # Vehicle parameters based on type
        self.vehicle_params = self._get_vehicle_params(vehicle_type)

        # Define observation space (30 dimensions)
        # Use float32 explicitly to avoid gymnasium dtype warnings
        self.observation_space = spaces.Box(
            low=np.array([
                -arena_bounds[0], -arena_bounds[1], 0.0,  # position (x, y, z)
                -50.0, -50.0, -50.0,  # velocity (vx, vy, vz) m/s
                -np.pi, -np.pi, -np.pi,  # orientation (roll, pitch, yaw)
                -10.0, -10.0, -10.0,  # angular velocity (wx, wy, wz)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # LiDAR (8 rays)
                -10.0, -10.0, -10.0,  # GPS noise (x, y, z)
                -50.0, -50.0, -50.0,  # IMU accelerometer (ax, ay, az)
                0.0,  # battery level (0-1)
                0.0,  # time remaining (0-1)
            ], dtype=np.float32),
            high=np.array([
                arena_bounds[0], arena_bounds[1], arena_bounds[2],  # position
                50.0, 50.0, 50.0,  # velocity
                np.pi, np.pi, np.pi,  # orientation
                10.0, 10.0, 10.0,  # angular velocity
                100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,  # LiDAR
                10.0, 10.0, 10.0,  # GPS noise
                50.0, 50.0, 50.0,  # IMU
                1.0,  # battery
                1.0,  # time
            ], dtype=np.float32),
            dtype=np.float32,
        )

        # Define action space (4 continuous actions)
        # Use float32 explicitly to avoid gymnasium dtype warnings
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),  # thrust, roll, pitch, yaw
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.goal_reached = False

        # Vehicle state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.battery_level = 1.0

        # Wind state
        self.wind_velocity = np.zeros(3)

        # Goal/waypoint
        self.goal_position = np.zeros(3)

        logger.info(f"Initialized EVTOLEnv with vehicle type: {vehicle_type}")

    def _get_vehicle_params(self, vehicle_type: str) -> Dict[str, float]:
        """Get vehicle parameters based on type."""
        params = {
            "small": {
                "mass": 100.0,  # kg
                "max_thrust": 1200.0,  # N
                "drag_coefficient": 0.3,
                "battery_capacity": 10.0,  # kWh
                "max_speed": 30.0,  # m/s
            },
            "medium": {
                "mass": 500.0,
                "max_thrust": 6000.0,
                "drag_coefficient": 0.4,
                "battery_capacity": 50.0,
                "max_speed": 40.0,
            },
            "large": {
                "mass": 1000.0,
                "max_thrust": 12000.0,
                "drag_coefficient": 0.5,
                "battery_capacity": 100.0,
                "max_speed": 35.0,
            },
        }
        return params.get(vehicle_type, params["medium"])

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.goal_reached = False

        # Random initial position at altitude 400-450 ft
        self.position = np.array([
            self.np_random.uniform(-self.arena_bounds[0] / 2, self.arena_bounds[0] / 2),
            self.np_random.uniform(-self.arena_bounds[1] / 2, self.arena_bounds[1] / 2),
            self.np_random.uniform(self.altitude_min, self.altitude_min + 50.0),
        ])

        # Start with small random velocity
        self.velocity = self.np_random.uniform(-1.0, 1.0, size=3)

        # Random initial orientation (small angles)
        self.orientation = self.np_random.uniform(-0.1, 0.1, size=3)

        # Zero angular velocity
        self.angular_velocity = np.zeros(3)

        # Full battery
        self.battery_level = 1.0

        # Random goal position
        self.goal_position = np.array([
            self.np_random.uniform(-self.arena_bounds[0] / 2, self.arena_bounds[0] / 2),
            self.np_random.uniform(-self.arena_bounds[1] / 2, self.arena_bounds[1] / 2),
            self.np_random.uniform(self.altitude_min, self.altitude_max),
        ])

        # Reset wind
        if self.enable_wind:
            self.wind_velocity = self.np_random.uniform(-2.0, 2.0, size=3)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Array of [thrust, roll, pitch, yaw]

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Parse action
        thrust_cmd = float(action[0])
        roll_cmd = float(action[1])
        pitch_cmd = float(action[2])
        yaw_cmd = float(action[3])

        # Apply physics update
        self._apply_dynamics(thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd)

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _apply_dynamics(self, thrust: float, roll: float, pitch: float, yaw: float):
        """Apply physics dynamics to update vehicle state."""
        dt = 0.02  # 50 Hz simulation

        # Get vehicle parameters
        mass = self.vehicle_params["mass"]
        max_thrust = self.vehicle_params["max_thrust"]
        drag_coeff = self.vehicle_params["drag_coefficient"]

        # Update orientation based on control inputs
        self.angular_velocity[0] = roll * 2.0  # Roll rate
        self.angular_velocity[1] = pitch * 2.0  # Pitch rate
        self.angular_velocity[2] = yaw * 2.0  # Yaw rate

        self.orientation += self.angular_velocity * dt

        # Normalize angles to [-pi, pi]
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))

        # Calculate thrust force in body frame
        thrust_force = thrust * max_thrust

        # Convert to world frame (simplified - assumes small angles)
        thrust_world = np.array([
            thrust_force * np.sin(self.orientation[1]),  # pitch contributes to forward
            -thrust_force * np.sin(self.orientation[0]),  # roll contributes to side
            thrust_force * np.cos(self.orientation[0]) * np.cos(self.orientation[1]),
        ])

        # Gravity
        gravity = np.array([0.0, 0.0, -9.81 * mass])

        # Drag force
        speed = np.linalg.norm(self.velocity)
        if speed > 0:
            drag = -drag_coeff * speed * self.velocity
        else:
            drag = np.zeros(3)

        # Wind effect
        if self.enable_wind:
            wind_force = self.wind_velocity * 10.0  # Simple wind force
        else:
            wind_force = np.zeros(3)

        # Total force
        total_force = thrust_world + gravity + drag + wind_force

        # Update velocity (F = ma -> a = F/m)
        acceleration = total_force / mass
        self.velocity += acceleration * dt

        # Update position
        self.position += self.velocity * dt

        # Update battery (simplified linear drain based on thrust)
        energy_consumption = thrust * 0.001 * dt  # Normalized consumption
        self.battery_level = max(0.0, self.battery_level - energy_consumption)

    def _calculate_reward(self) -> float:
        """Calculate reward for current state."""
        reward = 0.0

        # Altitude compliance
        altitude = self.position[2]
        if self.altitude_min <= altitude <= self.altitude_max:
            reward += 1.0
        else:
            # Penalty proportional to deviation
            if altitude < self.altitude_min:
                reward -= 0.5 * (self.altitude_min - altitude) / 100.0
            else:
                reward -= 0.5 * (altitude - self.altitude_max) / 100.0

        # Check for collision (simplified - just boundary check)
        if self._check_collision():
            reward -= 10.0
            self.collision_count += 1

        # Energy efficiency bonus
        if self.battery_level > 0.2:
            reward += 0.1

        # Goal proximity reward (distance-based)
        distance_to_goal = np.linalg.norm(self.position - self.goal_position)
        if distance_to_goal < 10.0:  # Within 10 meters
            reward += 5.0
            self.goal_reached = True
        else:
            # Small reward for getting closer
            reward += max(0, 1.0 - distance_to_goal / 1000.0)

        # Out of bounds penalty
        if np.any(np.abs(self.position[:2]) > self.arena_bounds[:2] / 2):
            reward -= 5.0

        # Time penalty (encourage efficiency)
        reward -= 0.01

        # Battery depletion penalty
        if self.battery_level <= 0.0:
            reward -= 3.0

        return reward

    def _check_collision(self) -> bool:
        """Check if vehicle has collided with obstacles."""
        # Simplified collision check - ground collision
        if self.position[2] < 10.0:  # Hit ground
            return True

        # Arena boundary collision
        if np.any(np.abs(self.position[:2]) > self.arena_bounds[:2] / 2):
            return True

        # TODO: Add building/obstacle collision detection

        return False

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Collision
        if self._check_collision():
            return True

        # Battery depleted
        if self.battery_level <= 0.0:
            return True

        # Goal reached
        if self.goal_reached:
            return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # LiDAR simulation (8 rays in horizontal plane)
        lidar = np.ones(8) * 50.0  # Default distance
        for i in range(8):
            angle = i * np.pi / 4
            # Simplified - just random noise for now
            if self.enable_sensor_noise:
                lidar[i] += self.np_random.uniform(-2.0, 2.0)

        # GPS noise
        if self.enable_sensor_noise:
            gps_noise = self.np_random.uniform(-1.0, 1.0, size=3)
        else:
            gps_noise = np.zeros(3)

        # IMU (accelerometer) - simplified
        imu = self.velocity * 2.0  # Rough acceleration estimate
        if self.enable_sensor_noise:
            imu += self.np_random.uniform(-0.5, 0.5, size=3)

        # Time remaining
        time_remaining = 1.0 - (self.current_step / self.max_steps)

        # Construct observation vector
        observation = np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
            self.angular_velocity,
            lidar,
            gps_noise,
            imu,
            np.array([self.battery_level]),
            np.array([time_remaining]),
        ]).astype(np.float32)

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information."""
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "orientation": self.orientation.copy(),
            "battery_level": self.battery_level,
            "distance_to_goal": np.linalg.norm(self.position - self.goal_position),
            "collision_count": self.collision_count,
            "episode_reward": self.episode_reward,
            "current_step": self.current_step,
            "goal_reached": self.goal_reached,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # TODO: Integrate with Omniverse viewer
            print(f"Step {self.current_step}: Pos={self.position}, Vel={self.velocity}")
        elif self.render_mode == "rgb_array":
            # TODO: Return RGB image from simulation
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        """Clean up resources."""
        pass
