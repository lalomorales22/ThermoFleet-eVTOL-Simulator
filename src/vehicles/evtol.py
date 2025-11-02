"""
eVTOL Vehicle Class

Main vehicle class that integrates configuration, physics, and state management
for autonomous flying vehicles.
"""

import numpy as np
from typing import Optional, Dict, Any
import uuid

from .evtol_config import VehicleBlueprint, VehicleType, get_vehicle_blueprint
from .physics import (
    AerodynamicsModel,
    WindTurbulenceModel,
    BatteryModel,
    WindConditions,
    GRAVITY
)


class eVTOL:
    """
    eVTOL (electric Vertical Take-Off and Landing) vehicle.

    Integrates vehicle configuration, physics simulation, and state management.
    """

    def __init__(
        self,
        vehicle_type: VehicleType,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        wind_conditions: Optional[WindConditions] = None,
        vehicle_id: Optional[str] = None
    ):
        """
        Initialize eVTOL vehicle.

        Args:
            vehicle_type: Type of vehicle (SMALL, MEDIUM, or LARGE)
            position: Initial position [x, y, z] in meters
            velocity: Initial velocity [vx, vy, vz] in m/s (default: zeros)
            wind_conditions: Wind conditions for simulation (default: calm)
            vehicle_id: Unique identifier (default: auto-generated UUID)
        """
        # Get vehicle blueprint
        self.blueprint = get_vehicle_blueprint(vehicle_type)
        self.vehicle_type = vehicle_type

        # Unique identifier
        self.id = vehicle_id or str(uuid.uuid4())

        # State variables
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity if velocity is not None else [0, 0, 0], dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.orientation = np.array([0, 0, 0], dtype=np.float32)  # [roll, pitch, yaw] in radians
        self.angular_velocity = np.zeros(3, dtype=np.float32)

        # Control inputs (0-1 normalized)
        self.thrust_input = 0.0  # Vertical thrust
        self.pitch_input = 0.0   # Pitch control
        self.roll_input = 0.0    # Roll control
        self.yaw_input = 0.0     # Yaw control

        # Physics models
        self.aerodynamics = AerodynamicsModel()
        self.wind_model = WindTurbulenceModel(
            wind_conditions or WindConditions(wind_velocity=np.zeros(3))
        )
        self.battery = BatteryModel(
            capacity_kwh=self.blueprint.battery_capacity,
            initial_soc=1.0
        )

        # Status flags
        self.active = True
        self.crashed = False
        self.battery_depleted = False

        # Simulation tracking
        self.sim_time = 0.0

    def set_controls(
        self,
        thrust: float,
        pitch: float = 0.0,
        roll: float = 0.0,
        yaw: float = 0.0
    ) -> None:
        """
        Set control inputs for the vehicle.

        Args:
            thrust: Thrust input (0-1), where 1 is maximum thrust
            pitch: Pitch control (-1 to 1), negative is nose down
            roll: Roll control (-1 to 1), negative is left
            yaw: Yaw control (-1 to 1), negative is counter-clockwise
        """
        self.thrust_input = np.clip(thrust, 0.0, 1.0)
        self.pitch_input = np.clip(pitch, -1.0, 1.0)
        self.roll_input = np.clip(roll, -1.0, 1.0)
        self.yaw_input = np.clip(yaw, -1.0, 1.0)

    def update_physics(self, dt: float) -> None:
        """
        Update vehicle physics for one time step.

        Integrates forces to compute acceleration, velocity, and position.

        Args:
            dt: Time step in seconds
        """
        if not self.active:
            return

        # Calculate forces
        forces = self._calculate_forces()

        # Update acceleration (F = ma, so a = F/m)
        self.acceleration = forces / self.blueprint.mass

        # Update velocity (v = v0 + a*dt)
        self.velocity += self.acceleration * dt

        # Clamp velocity to max speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.blueprint.max_speed:
            self.velocity = self.velocity / speed * self.blueprint.max_speed

        # Update position (x = x0 + v*dt)
        self.position += self.velocity * dt

        # Update orientation (simplified - based on control inputs)
        self._update_orientation(dt)

        # Update battery
        power_consumption = self.battery.calculate_power_consumption(
            base_power=self.blueprint.power_consumption_base,
            thrust_fraction=self.thrust_input,
            velocity=self.velocity
        )
        battery_ok = self.battery.consume_power(power_consumption, dt)

        if not battery_ok:
            self.battery_depleted = True
            self.active = False

        # Update wind model
        self.wind_model.update(dt)

        # Update simulation time
        self.sim_time += dt

        # Check for crash conditions
        self._check_crash()

    def _calculate_forces(self) -> np.ndarray:
        """
        Calculate total forces acting on the vehicle.

        Returns:
            Total force vector [fx, fy, fz] in N
        """
        forces = np.zeros(3, dtype=np.float32)

        # 1. Gravity (always downward)
        gravity_force = np.array([0, 0, -self.blueprint.weight], dtype=np.float32)
        forces += gravity_force

        # 2. Thrust (simplified - mostly vertical, affected by orientation)
        thrust_magnitude = self.thrust_input * self.blueprint.max_thrust
        # Apply thrust in body frame (considering pitch and roll for simplification)
        thrust_force = np.array([
            thrust_magnitude * np.sin(self.orientation[1]),  # pitch component
            thrust_magnitude * np.sin(self.orientation[0]),  # roll component
            thrust_magnitude * np.cos(self.orientation[1]) * np.cos(self.orientation[0])  # vertical
        ], dtype=np.float32)
        forces += thrust_force

        # 3. Aerodynamic drag
        drag_force = self.aerodynamics.calculate_drag_force(
            velocity=self.velocity,
            drag_coefficient=self.blueprint.drag_coefficient,
            frontal_area=self.blueprint.frontal_area
        )
        forces += drag_force

        # 4. Aerodynamic lift (vertical component only)
        lift_magnitude = self.aerodynamics.calculate_lift_force(
            velocity=self.velocity,
            lift_coefficient=self.blueprint.lift_coefficient,
            wing_area=self.blueprint.wing_area
        )
        forces += np.array([0, 0, lift_magnitude], dtype=np.float32)

        # 5. Wind and turbulence
        wind_force = self.wind_model.get_wind_force(self.blueprint.frontal_area)
        forces += wind_force

        return forces

    def _update_orientation(self, dt: float) -> None:
        """
        Update vehicle orientation based on control inputs.

        Simplified model: control inputs directly affect angular velocity.

        Args:
            dt: Time step in seconds
        """
        # Update angular velocity based on control inputs
        max_angular_vel = self.blueprint.max_angular_velocity

        # Target angular velocities from controls
        target_angular_vel = np.array([
            self.roll_input * max_angular_vel,
            self.pitch_input * max_angular_vel,
            self.yaw_input * max_angular_vel
        ], dtype=np.float32)

        # Smooth transition (simple first-order system)
        damping = 0.5
        self.angular_velocity += (target_angular_vel - self.angular_velocity) * damping * dt

        # Update orientation
        self.orientation += self.angular_velocity * dt

        # Normalize angles to [-π, π]
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))

    def _check_crash(self) -> None:
        """Check for crash conditions."""
        # Ground collision (z < 0)
        if self.position[2] < 0:
            self.crashed = True
            self.active = False
            self.position[2] = 0  # Snap to ground
            self.velocity = np.zeros(3, dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        """
        Get current vehicle state.

        Returns:
            Dictionary containing vehicle state information
        """
        return {
            'id': self.id,
            'vehicle_type': self.vehicle_type.value,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'acceleration': self.acceleration.tolist(),
            'orientation': self.orientation.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'thrust_input': self.thrust_input,
            'active': self.active,
            'crashed': self.crashed,
            'battery': self.battery.get_state(),
            'sim_time': self.sim_time
        }

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None
    ) -> None:
        """
        Reset vehicle to initial state.

        Args:
            position: New initial position (default: keep current)
            velocity: New initial velocity (default: zeros)
        """
        if position is not None:
            self.position = np.array(position, dtype=np.float32)

        self.velocity = np.array(velocity if velocity is not None else [0, 0, 0], dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.orientation = np.array([0, 0, 0], dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)

        self.thrust_input = 0.0
        self.pitch_input = 0.0
        self.roll_input = 0.0
        self.yaw_input = 0.0

        self.active = True
        self.crashed = False
        self.battery_depleted = False

        self.battery.reset(initial_soc=1.0)
        self.sim_time = 0.0

    def __repr__(self) -> str:
        """String representation of vehicle."""
        return (
            f"eVTOL(id={self.id[:8]}..., type={self.vehicle_type.value}, "
            f"pos={self.position}, active={self.active})"
        )
