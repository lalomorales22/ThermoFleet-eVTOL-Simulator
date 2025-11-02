"""
Vehicle Spawning System

API for spawning multiple eVTOL vehicles in the simulation arena
with configurable parameters and entry logic.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from .evtol import eVTOL
from .evtol_config import VehicleType
from .physics import WindConditions


class SpawnPattern(Enum):
    """Spawn pattern types for vehicle placement"""
    RANDOM = "random"
    GRID = "grid"
    CIRCLE = "circle"
    LINE = "line"


class VehicleSpawner:
    """
    Vehicle spawning system for managing multiple eVTOL agents.

    Handles spawning, tracking, and management of vehicle fleets.
    """

    # Default altitude constraints (in meters)
    DEFAULT_MIN_ALTITUDE = 400 * 0.3048  # 400 feet in meters (~122m)
    DEFAULT_MAX_ALTITUDE = 500 * 0.3048  # 500 feet in meters (~152m)
    DEFAULT_SPAWN_ALTITUDE = 400 * 0.3048  # Spawn at 400 feet

    def __init__(
        self,
        arena_bounds: Tuple[float, float, float, float],
        min_altitude: float = DEFAULT_MIN_ALTITUDE,
        max_altitude: float = DEFAULT_MAX_ALTITUDE,
        wind_conditions: Optional[WindConditions] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize vehicle spawner.

        Args:
            arena_bounds: (x_min, x_max, y_min, y_max) bounds in meters
            min_altitude: Minimum operational altitude in meters
            max_altitude: Maximum operational altitude in meters
            wind_conditions: Wind conditions for all vehicles
            random_seed: Random seed for reproducibility
        """
        self.arena_bounds = arena_bounds
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.wind_conditions = wind_conditions or WindConditions(
            wind_velocity=np.zeros(3)
        )

        self.rng = np.random.default_rng(random_seed)
        self.vehicles: List[eVTOL] = []

    def spawn_vehicles(
        self,
        count: int,
        vehicle_types: Optional[List[VehicleType]] = None,
        pattern: SpawnPattern = SpawnPattern.RANDOM,
        spawn_altitude: Optional[float] = None
    ) -> List[eVTOL]:
        """
        Spawn multiple vehicles in the arena.

        Args:
            count: Number of vehicles to spawn
            vehicle_types: List of vehicle types (if None, random distribution)
            pattern: Spawn pattern for vehicle placement
            spawn_altitude: Specific spawn altitude (default: DEFAULT_SPAWN_ALTITUDE)

        Returns:
            List of spawned eVTOL vehicles
        """
        spawn_alt = spawn_altitude or self.DEFAULT_SPAWN_ALTITUDE

        # Determine vehicle types
        if vehicle_types is None:
            # Random distribution: 50% small, 30% medium, 20% large
            vehicle_types = self._generate_random_vehicle_types(count)
        elif len(vehicle_types) != count:
            raise ValueError(
                f"vehicle_types length ({len(vehicle_types)}) must match count ({count})"
            )

        # Generate spawn positions
        positions = self._generate_spawn_positions(count, pattern, spawn_alt)

        # Create vehicles
        spawned_vehicles = []
        for i in range(count):
            vehicle = eVTOL(
                vehicle_type=vehicle_types[i],
                position=positions[i],
                wind_conditions=self.wind_conditions
            )
            spawned_vehicles.append(vehicle)
            self.vehicles.append(vehicle)

        return spawned_vehicles

    def spawn_single_vehicle(
        self,
        vehicle_type: VehicleType,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None
    ) -> eVTOL:
        """
        Spawn a single vehicle.

        Args:
            vehicle_type: Type of vehicle to spawn
            position: Spawn position (if None, random within arena)
            velocity: Initial velocity (default: zeros)

        Returns:
            Spawned eVTOL vehicle
        """
        if position is None:
            position = self._generate_spawn_positions(1, SpawnPattern.RANDOM)[0]

        vehicle = eVTOL(
            vehicle_type=vehicle_type,
            position=position,
            velocity=velocity,
            wind_conditions=self.wind_conditions
        )

        self.vehicles.append(vehicle)
        return vehicle

    def _generate_random_vehicle_types(self, count: int) -> List[VehicleType]:
        """
        Generate random vehicle type distribution.

        Distribution: 50% small, 30% medium, 20% large

        Args:
            count: Number of vehicle types to generate

        Returns:
            List of VehicleType enums
        """
        probabilities = [0.5, 0.3, 0.2]  # small, medium, large
        types = [VehicleType.SMALL, VehicleType.MEDIUM, VehicleType.LARGE]

        vehicle_types = self.rng.choice(types, size=count, p=probabilities)
        return vehicle_types.tolist()

    def _generate_spawn_positions(
        self,
        count: int,
        pattern: SpawnPattern,
        altitude: float = None
    ) -> List[np.ndarray]:
        """
        Generate spawn positions based on pattern.

        Args:
            count: Number of positions to generate
            pattern: Spawn pattern type
            altitude: Spawn altitude (default: DEFAULT_SPAWN_ALTITUDE)

        Returns:
            List of position arrays [x, y, z]
        """
        spawn_alt = altitude or self.DEFAULT_SPAWN_ALTITUDE
        x_min, x_max, y_min, y_max = self.arena_bounds

        if pattern == SpawnPattern.RANDOM:
            return self._generate_random_positions(count, spawn_alt)

        elif pattern == SpawnPattern.GRID:
            return self._generate_grid_positions(count, spawn_alt)

        elif pattern == SpawnPattern.CIRCLE:
            return self._generate_circle_positions(count, spawn_alt)

        elif pattern == SpawnPattern.LINE:
            return self._generate_line_positions(count, spawn_alt)

        else:
            raise ValueError(f"Unknown spawn pattern: {pattern}")

    def _generate_random_positions(
        self,
        count: int,
        altitude: float
    ) -> List[np.ndarray]:
        """Generate random positions within arena bounds."""
        x_min, x_max, y_min, y_max = self.arena_bounds

        positions = []
        for _ in range(count):
            x = self.rng.uniform(x_min, x_max)
            y = self.rng.uniform(y_min, y_max)
            z = altitude + self.rng.uniform(-5, 5)  # Small altitude variation

            positions.append(np.array([x, y, z], dtype=np.float32))

        return positions

    def _generate_grid_positions(
        self,
        count: int,
        altitude: float
    ) -> List[np.ndarray]:
        """Generate grid-based positions."""
        x_min, x_max, y_min, y_max = self.arena_bounds

        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(count)))
        x_spacing = (x_max - x_min) / (grid_size + 1)
        y_spacing = (y_max - y_min) / (grid_size + 1)

        positions = []
        for i in range(count):
            row = i // grid_size
            col = i % grid_size

            x = x_min + (col + 1) * x_spacing
            y = y_min + (row + 1) * y_spacing
            z = altitude

            positions.append(np.array([x, y, z], dtype=np.float32))

        return positions

    def _generate_circle_positions(
        self,
        count: int,
        altitude: float
    ) -> List[np.ndarray]:
        """Generate circular formation positions."""
        x_min, x_max, y_min, y_max = self.arena_bounds

        # Circle center and radius
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        radius = min(x_max - center_x, y_max - center_y) * 0.8

        positions = []
        for i in range(count):
            angle = 2 * np.pi * i / count
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            z = altitude

            positions.append(np.array([x, y, z], dtype=np.float32))

        return positions

    def _generate_line_positions(
        self,
        count: int,
        altitude: float
    ) -> List[np.ndarray]:
        """Generate line formation positions."""
        x_min, x_max, y_min, y_max = self.arena_bounds

        # Line along x-axis at y center
        y_center = (y_min + y_max) / 2
        x_spacing = (x_max - x_min) / (count + 1)

        positions = []
        for i in range(count):
            x = x_min + (i + 1) * x_spacing
            y = y_center
            z = altitude

            positions.append(np.array([x, y, z], dtype=np.float32))

        return positions

    def update_all_vehicles(self, dt: float) -> None:
        """
        Update physics for all active vehicles.

        Args:
            dt: Time step in seconds
        """
        for vehicle in self.vehicles:
            if vehicle.active:
                vehicle.update_physics(dt)

    def get_active_vehicles(self) -> List[eVTOL]:
        """Get list of all active vehicles."""
        return [v for v in self.vehicles if v.active]

    def get_vehicle_states(self) -> List[Dict[str, Any]]:
        """Get state information for all vehicles."""
        return [v.get_state() for v in self.vehicles]

    def remove_inactive_vehicles(self) -> int:
        """
        Remove inactive vehicles from the fleet.

        Returns:
            Number of vehicles removed
        """
        initial_count = len(self.vehicles)
        self.vehicles = [v for v in self.vehicles if v.active]
        removed_count = initial_count - len(self.vehicles)
        return removed_count

    def reset_all_vehicles(self) -> None:
        """Reset all vehicles to their initial spawn positions."""
        for vehicle in self.vehicles:
            vehicle.reset()

    def clear_all_vehicles(self) -> None:
        """Remove all vehicles from the spawner."""
        self.vehicles.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get fleet statistics.

        Returns:
            Dictionary with fleet statistics
        """
        total = len(self.vehicles)
        active = len([v for v in self.vehicles if v.active])
        crashed = len([v for v in self.vehicles if v.crashed])
        battery_depleted = len([v for v in self.vehicles if v.battery_depleted])

        vehicle_type_counts = {
            'small': len([v for v in self.vehicles if v.vehicle_type == VehicleType.SMALL]),
            'medium': len([v for v in self.vehicles if v.vehicle_type == VehicleType.MEDIUM]),
            'large': len([v for v in self.vehicles if v.vehicle_type == VehicleType.LARGE]),
        }

        return {
            'total_vehicles': total,
            'active_vehicles': active,
            'crashed_vehicles': crashed,
            'battery_depleted_vehicles': battery_depleted,
            'vehicle_type_counts': vehicle_type_counts,
            'arena_bounds': self.arena_bounds,
            'altitude_range': (self.min_altitude, self.max_altitude)
        }

    def __repr__(self) -> str:
        """String representation of spawner."""
        stats = self.get_statistics()
        return (
            f"VehicleSpawner(total={stats['total_vehicles']}, "
            f"active={stats['active_vehicles']}, "
            f"bounds={self.arena_bounds})"
        )
