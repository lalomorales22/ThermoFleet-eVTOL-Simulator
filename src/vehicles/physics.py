"""
Physics Simulation Module for eVTOL Vehicles

Implements aerodynamics, wind/turbulence models, and battery simulation
for realistic flight dynamics.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


# Constants
AIR_DENSITY = 1.225  # kg/m³ at sea level
GRAVITY = 9.81  # m/s²


@dataclass
class WindConditions:
    """
    Wind and turbulence conditions for the simulation.

    Attributes:
        wind_velocity: Mean wind velocity vector [vx, vy, vz] in m/s
        turbulence_intensity: Turbulence intensity factor (0-1)
        gust_frequency: Frequency of wind gusts in Hz
    """
    wind_velocity: np.ndarray  # [vx, vy, vz]
    turbulence_intensity: float = 0.1
    gust_frequency: float = 0.5

    def __post_init__(self):
        """Ensure wind_velocity is a numpy array"""
        if not isinstance(self.wind_velocity, np.ndarray):
            self.wind_velocity = np.array(self.wind_velocity, dtype=np.float32)


class AerodynamicsModel:
    """
    Aerodynamics model for eVTOL vehicles.

    Implements drag and lift forces based on vehicle velocity and configuration.
    """

    @staticmethod
    def calculate_drag_force(
        velocity: np.ndarray,
        drag_coefficient: float,
        frontal_area: float,
        air_density: float = AIR_DENSITY
    ) -> np.ndarray:
        """
        Calculate aerodynamic drag force.

        Drag equation: F_d = 0.5 * ρ * v² * C_d * A

        Args:
            velocity: Velocity vector [vx, vy, vz] in m/s
            drag_coefficient: Drag coefficient (Cd)
            frontal_area: Frontal area in m²
            air_density: Air density in kg/m³

        Returns:
            Drag force vector [fx, fy, fz] in N (opposite to velocity direction)
        """
        speed = np.linalg.norm(velocity)
        if speed < 0.01:  # Avoid division by zero
            return np.zeros(3, dtype=np.float32)

        # Drag magnitude
        drag_magnitude = 0.5 * air_density * speed**2 * drag_coefficient * frontal_area

        # Drag direction (opposite to velocity)
        drag_direction = -velocity / speed

        return drag_magnitude * drag_direction

    @staticmethod
    def calculate_lift_force(
        velocity: np.ndarray,
        lift_coefficient: float,
        wing_area: float,
        air_density: float = AIR_DENSITY
    ) -> float:
        """
        Calculate aerodynamic lift force (vertical component).

        Lift equation: F_l = 0.5 * ρ * v² * C_l * A

        Args:
            velocity: Velocity vector [vx, vy, vz] in m/s
            lift_coefficient: Lift coefficient (Cl)
            wing_area: Wing/rotor area in m²
            air_density: Air density in kg/m³

        Returns:
            Lift force magnitude in N (positive upward)
        """
        # Use horizontal speed for lift calculation (vx, vy)
        horizontal_speed = np.linalg.norm(velocity[:2])

        if horizontal_speed < 0.01:  # Minimal lift at low speeds
            return 0.0

        lift_magnitude = 0.5 * air_density * horizontal_speed**2 * lift_coefficient * wing_area
        return float(lift_magnitude)


class WindTurbulenceModel:
    """
    Wind and turbulence simulation using Gaussian noise models.

    Implements realistic wind gusts and turbulence forces.
    """

    def __init__(self, conditions: WindConditions, random_seed: int = None):
        """
        Initialize wind turbulence model.

        Args:
            conditions: WindConditions configuration
            random_seed: Random seed for reproducibility
        """
        self.conditions = conditions
        self.rng = np.random.default_rng(random_seed)
        self.time = 0.0

    def update(self, dt: float) -> None:
        """
        Update internal time for time-varying wind.

        Args:
            dt: Time step in seconds
        """
        self.time += dt

    def get_wind_force(self, frontal_area: float) -> np.ndarray:
        """
        Calculate wind force including turbulence.

        Wind force combines:
        1. Mean wind force (steady component)
        2. Turbulent fluctuations (Gaussian noise)
        3. Periodic gusts (sinusoidal component)

        Args:
            frontal_area: Vehicle frontal area in m²

        Returns:
            Wind force vector [fx, fy, fz] in N
        """
        # Mean wind force (simplified drag-like calculation)
        wind_speed = np.linalg.norm(self.conditions.wind_velocity)
        mean_wind_force = 0.5 * AIR_DENSITY * wind_speed**2 * frontal_area
        mean_direction = self.conditions.wind_velocity / (wind_speed + 1e-6)

        # Turbulent fluctuations (Gaussian noise)
        turbulence_std = self.conditions.turbulence_intensity * mean_wind_force
        turbulence = self.rng.normal(0, turbulence_std, size=3).astype(np.float32)

        # Periodic gusts (sinusoidal component)
        gust_amplitude = 0.3 * mean_wind_force  # 30% of mean
        gust_phase = 2 * np.pi * self.conditions.gust_frequency * self.time
        gust = gust_amplitude * np.sin(gust_phase) * mean_direction

        # Total wind force
        total_force = mean_wind_force * mean_direction + turbulence + gust

        return total_force.astype(np.float32)


class BatteryModel:
    """
    Battery simulation with linear energy drain.

    Models battery capacity, power consumption, and state of charge.
    """

    def __init__(self, capacity_kwh: float, initial_soc: float = 1.0):
        """
        Initialize battery model.

        Args:
            capacity_kwh: Battery capacity in kWh
            initial_soc: Initial state of charge (0-1)
        """
        self.capacity_kwh = capacity_kwh
        self.capacity_joules = capacity_kwh * 3.6e6  # Convert kWh to Joules
        self.energy_remaining = self.capacity_joules * initial_soc
        self.soc = initial_soc

    def consume_power(self, power_kw: float, dt: float) -> bool:
        """
        Consume power from battery over time step.

        Args:
            power_kw: Power consumption in kW
            dt: Time step in seconds

        Returns:
            True if battery has sufficient charge, False if depleted
        """
        # Calculate energy consumed (E = P * t)
        energy_consumed = power_kw * 1000 * dt  # Convert kW to W, then to Joules

        # Update remaining energy
        self.energy_remaining = max(0, self.energy_remaining - energy_consumed)

        # Update state of charge
        self.soc = self.energy_remaining / self.capacity_joules

        return self.soc > 0.0

    def calculate_power_consumption(
        self,
        base_power: float,
        thrust_fraction: float,
        velocity: np.ndarray
    ) -> float:
        """
        Calculate instantaneous power consumption based on flight conditions.

        Power consumption model:
        P_total = P_base + P_thrust + P_drag

        Args:
            base_power: Base power consumption in kW (systems, avionics)
            thrust_fraction: Fraction of max thrust being used (0-1)
            velocity: Velocity vector in m/s

        Returns:
            Total power consumption in kW
        """
        # Base power (always consumed)
        power = base_power

        # Thrust power (linear with thrust fraction, simplified)
        # In reality: P = F * v, but we approximate for hover/acceleration
        thrust_power = base_power * 2.0 * thrust_fraction

        # Velocity-dependent power (increases with speed)
        speed = np.linalg.norm(velocity)
        velocity_power = 0.01 * speed**2  # Quadratic drag-like term

        total_power = power + thrust_power + velocity_power

        return float(total_power)

    def get_state(self) -> dict:
        """
        Get battery state information.

        Returns:
            Dictionary with battery state metrics
        """
        return {
            'soc': self.soc,
            'energy_remaining_kwh': self.energy_remaining / 3.6e6,
            'capacity_kwh': self.capacity_kwh,
            'depleted': self.soc <= 0.0
        }

    def reset(self, initial_soc: float = 1.0) -> None:
        """
        Reset battery to initial state.

        Args:
            initial_soc: Initial state of charge (0-1)
        """
        self.energy_remaining = self.capacity_joules * initial_soc
        self.soc = initial_soc
