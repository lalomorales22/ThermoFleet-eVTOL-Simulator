"""
User Control Components for FlyingCarRL

Provides interfaces for controlling simulation parameters and spawning agents.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json


class SimulationState(Enum):
    """Simulation state"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    RECORDING = "recording"


@dataclass
class VehicleConfig:
    """Configuration for a vehicle type"""
    type: str  # 'small', 'medium', 'large'
    mass: float  # kg
    max_thrust: float  # N
    drag_coefficient: float
    battery_capacity: float  # Wh
    color: Tuple[float, float, float] = (0.2, 0.5, 0.8)


@dataclass
class ArenaConfig:
    """Configuration for an arena"""
    name: str
    location: str  # e.g., "New York City"
    bounds: Tuple[float, float, float, float]  # min_x, max_x, min_y, max_y
    altitude_range: Tuple[float, float] = (400.0, 500.0)  # feet
    terrain_file: Optional[str] = None
    wind_enabled: bool = True
    wind_speed: float = 5.0  # m/s
    wind_direction: float = 0.0  # degrees


# Predefined vehicle types
VEHICLE_TYPES = {
    'small': VehicleConfig(
        type='small',
        mass=100.0,
        max_thrust=1500.0,
        drag_coefficient=0.25,
        battery_capacity=10000.0,
        color=(0.2, 0.8, 0.2)
    ),
    'medium': VehicleConfig(
        type='medium',
        mass=500.0,
        max_thrust=6000.0,
        drag_coefficient=0.35,
        battery_capacity=50000.0,
        color=(0.2, 0.5, 0.8)
    ),
    'large': VehicleConfig(
        type='large',
        mass=1000.0,
        max_thrust=12000.0,
        drag_coefficient=0.50,
        battery_capacity=100000.0,
        color=(0.8, 0.2, 0.2)
    )
}


# Predefined arenas
ARENAS = {
    'NYC_Manhattan': ArenaConfig(
        name='NYC_Manhattan',
        location='New York City, Manhattan',
        bounds=(-5000, 5000, -5000, 5000),
        terrain_file='assets/arenas/nyc_manhattan.usd'
    ),
    'SF_Downtown': ArenaConfig(
        name='SF_Downtown',
        location='San Francisco, Downtown',
        bounds=(-3000, 3000, -3000, 3000),
        terrain_file='assets/arenas/sf_downtown.usd'
    ),
    'LA_Airport': ArenaConfig(
        name='LA_Airport',
        location='Los Angeles, LAX Area',
        bounds=(-8000, 8000, -8000, 8000),
        terrain_file='assets/arenas/la_airport.usd'
    ),
    'Test_Grid': ArenaConfig(
        name='Test_Grid',
        location='Test Environment',
        bounds=(-1000, 1000, -1000, 1000),
        wind_enabled=False
    )
}


class SimulationController:
    """
    Controls simulation execution and state.
    """

    def __init__(self):
        """Initialize simulation controller"""
        self.state = SimulationState.STOPPED
        self.current_episode = 0
        self.current_timestep = 0
        self.total_reward = 0.0

        # Callbacks
        self.on_start_callbacks: List[Callable] = []
        self.on_stop_callbacks: List[Callable] = []
        self.on_pause_callbacks: List[Callable] = []
        self.on_reset_callbacks: List[Callable] = []

    def start(self) -> bool:
        """Start simulation"""
        if self.state == SimulationState.RUNNING:
            return False

        self.state = SimulationState.RUNNING
        print("▶ Simulation started")

        # Execute callbacks
        for callback in self.on_start_callbacks:
            callback()

        return True

    def stop(self) -> bool:
        """Stop simulation"""
        if self.state == SimulationState.STOPPED:
            return False

        self.state = SimulationState.STOPPED
        print("■ Simulation stopped")

        # Execute callbacks
        for callback in self.on_stop_callbacks:
            callback()

        return True

    def pause(self) -> bool:
        """Pause simulation"""
        if self.state != SimulationState.RUNNING:
            return False

        self.state = SimulationState.PAUSED
        print("⏸ Simulation paused")

        # Execute callbacks
        for callback in self.on_pause_callbacks:
            callback()

        return True

    def resume(self) -> bool:
        """Resume simulation"""
        if self.state != SimulationState.PAUSED:
            return False

        self.state = SimulationState.RUNNING
        print("▶ Simulation resumed")

        return True

    def reset(self) -> bool:
        """Reset simulation"""
        was_running = self.state == SimulationState.RUNNING

        self.state = SimulationState.STOPPED
        self.current_episode = 0
        self.current_timestep = 0
        self.total_reward = 0.0

        print("⟲ Simulation reset")

        # Execute callbacks
        for callback in self.on_reset_callbacks:
            callback()

        if was_running:
            self.start()

        return True

    def step(self) -> bool:
        """Execute single simulation step (for manual stepping)"""
        if self.state != SimulationState.PAUSED:
            return False

        self.current_timestep += 1
        return True

    def is_running(self) -> bool:
        """Check if simulation is running"""
        return self.state == SimulationState.RUNNING

    def is_paused(self) -> bool:
        """Check if simulation is paused"""
        return self.state == SimulationState.PAUSED

    def is_stopped(self) -> bool:
        """Check if simulation is stopped"""
        return self.state == SimulationState.STOPPED

    def get_state(self) -> SimulationState:
        """Get current state"""
        return self.state

    def register_callback(self, event: str, callback: Callable):
        """
        Register callback for simulation events.

        Args:
            event: Event name ('start', 'stop', 'pause', 'reset')
            callback: Callback function
        """
        if event == 'start':
            self.on_start_callbacks.append(callback)
        elif event == 'stop':
            self.on_stop_callbacks.append(callback)
        elif event == 'pause':
            self.on_pause_callbacks.append(callback)
        elif event == 'reset':
            self.on_reset_callbacks.append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")


class VehicleSpawner:
    """
    Manages spawning of eVTOL vehicles.
    """

    def __init__(self, arena_config: ArenaConfig):
        """
        Initialize vehicle spawner.

        Args:
            arena_config: Arena configuration
        """
        self.arena = arena_config
        self.spawned_vehicles = {}
        self.next_id = 0

    def spawn_single(
        self,
        vehicle_type: str = 'medium',
        position: Optional[np.ndarray] = None,
        custom_config: Optional[VehicleConfig] = None
    ) -> int:
        """
        Spawn a single vehicle.

        Args:
            vehicle_type: Type of vehicle
            position: Initial position (random if None)
            custom_config: Custom vehicle configuration

        Returns:
            Vehicle ID
        """
        # Get vehicle config
        if custom_config:
            config = custom_config
        else:
            config = VEHICLE_TYPES.get(vehicle_type, VEHICLE_TYPES['medium'])

        # Generate position if not provided
        if position is None:
            position = self._random_position()

        # Create vehicle entry
        vehicle_id = self.next_id
        self.next_id += 1

        self.spawned_vehicles[vehicle_id] = {
            'id': vehicle_id,
            'type': vehicle_type,
            'config': config,
            'position': position,
            'active': True
        }

        print(f"✓ Spawned {vehicle_type} vehicle (ID: {vehicle_id}) at {position}")

        return vehicle_id

    def spawn_fleet(
        self,
        count: int,
        vehicle_type: str = 'medium',
        formation: str = 'grid'
    ) -> List[int]:
        """
        Spawn a fleet of vehicles.

        Args:
            count: Number of vehicles
            vehicle_type: Type of vehicles
            formation: Formation type ('grid', 'random', 'line')

        Returns:
            List of vehicle IDs
        """
        positions = self._generate_formation(count, formation)

        vehicle_ids = []
        for pos in positions:
            vid = self.spawn_single(vehicle_type, position=pos)
            vehicle_ids.append(vid)

        print(f"✓ Spawned fleet of {count} {vehicle_type} vehicles in {formation} formation")

        return vehicle_ids

    def spawn_mixed_fleet(
        self,
        counts: Dict[str, int],
        formation: str = 'random'
    ) -> List[int]:
        """
        Spawn a mixed fleet of different vehicle types.

        Args:
            counts: Dictionary mapping vehicle type to count
            formation: Formation type

        Returns:
            List of vehicle IDs
        """
        vehicle_ids = []

        for vehicle_type, count in counts.items():
            ids = self.spawn_fleet(count, vehicle_type, formation)
            vehicle_ids.extend(ids)

        total = sum(counts.values())
        print(f"✓ Spawned mixed fleet of {total} vehicles")

        return vehicle_ids

    def despawn(self, vehicle_id: int) -> bool:
        """Despawn a vehicle"""
        if vehicle_id in self.spawned_vehicles:
            self.spawned_vehicles[vehicle_id]['active'] = False
            print(f"✓ Despawned vehicle {vehicle_id}")
            return True
        return False

    def despawn_all(self):
        """Despawn all vehicles"""
        for vid in list(self.spawned_vehicles.keys()):
            self.despawn(vid)
        self.spawned_vehicles.clear()
        self.next_id = 0
        print("✓ Despawned all vehicles")

    def get_active_vehicles(self) -> List[int]:
        """Get list of active vehicle IDs"""
        return [
            vid for vid, info in self.spawned_vehicles.items()
            if info.get('active', False)
        ]

    def get_vehicle_info(self, vehicle_id: int) -> Optional[Dict]:
        """Get vehicle information"""
        return self.spawned_vehicles.get(vehicle_id)

    def _random_position(self) -> np.ndarray:
        """Generate random position within arena bounds"""
        min_x, max_x, min_y, max_y = self.arena.bounds
        min_alt, max_alt = self.arena.altitude_range

        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = np.random.uniform(min_alt, max_alt)

        return np.array([x, y, z])

    def _generate_formation(
        self,
        count: int,
        formation: str
    ) -> List[np.ndarray]:
        """
        Generate positions for a formation.

        Args:
            count: Number of positions
            formation: Formation type

        Returns:
            List of positions
        """
        positions = []

        if formation == 'grid':
            # Grid formation
            grid_size = int(np.ceil(np.sqrt(count)))
            spacing = 50.0  # meters

            for i in range(count):
                row = i // grid_size
                col = i % grid_size
                x = (col - grid_size / 2) * spacing
                y = (row - grid_size / 2) * spacing
                z = 450.0  # middle of altitude range

                positions.append(np.array([x, y, z]))

        elif formation == 'line':
            # Line formation
            spacing = 30.0
            for i in range(count):
                x = (i - count / 2) * spacing
                y = 0.0
                z = 450.0

                positions.append(np.array([x, y, z]))

        elif formation == 'random':
            # Random positions
            for _ in range(count):
                positions.append(self._random_position())

        else:
            raise ValueError(f"Unknown formation: {formation}")

        return positions


class ArenaSelector:
    """
    Manages arena selection and configuration.
    """

    def __init__(self):
        """Initialize arena selector"""
        self.current_arena = None

    def list_arenas(self) -> List[str]:
        """Get list of available arenas"""
        return list(ARENAS.keys())

    def get_arena_info(self, arena_name: str) -> Optional[ArenaConfig]:
        """Get arena configuration"""
        return ARENAS.get(arena_name)

    def select_arena(self, arena_name: str) -> Optional[ArenaConfig]:
        """
        Select an arena.

        Args:
            arena_name: Name of arena

        Returns:
            Arena configuration
        """
        if arena_name not in ARENAS:
            print(f"❌ Unknown arena: {arena_name}")
            available = ', '.join(self.list_arenas())
            print(f"Available arenas: {available}")
            return None

        self.current_arena = ARENAS[arena_name]
        print(f"✓ Selected arena: {arena_name}")
        print(f"  Location: {self.current_arena.location}")
        print(f"  Bounds: {self.current_arena.bounds}")
        print(f"  Altitude: {self.current_arena.altitude_range[0]}-{self.current_arena.altitude_range[1]} ft")

        return self.current_arena

    def get_current_arena(self) -> Optional[ArenaConfig]:
        """Get currently selected arena"""
        return self.current_arena

    def create_custom_arena(
        self,
        name: str,
        location: str,
        bounds: Tuple[float, float, float, float],
        **kwargs
    ) -> ArenaConfig:
        """
        Create a custom arena.

        Args:
            name: Arena name
            location: Location description
            bounds: Arena bounds
            **kwargs: Additional arena parameters

        Returns:
            Arena configuration
        """
        arena = ArenaConfig(
            name=name,
            location=location,
            bounds=bounds,
            **kwargs
        )

        # Add to available arenas
        ARENAS[name] = arena
        print(f"✓ Created custom arena: {name}")

        return arena

    def save_arena(self, arena_name: str, filepath: str):
        """Save arena configuration to file"""
        if arena_name not in ARENAS:
            print(f"❌ Arena not found: {arena_name}")
            return

        arena = ARENAS[arena_name]
        arena_dict = {
            'name': arena.name,
            'location': arena.location,
            'bounds': arena.bounds,
            'altitude_range': arena.altitude_range,
            'terrain_file': arena.terrain_file,
            'wind_enabled': arena.wind_enabled,
            'wind_speed': arena.wind_speed,
            'wind_direction': arena.wind_direction,
        }

        with open(filepath, 'w') as f:
            json.dump(arena_dict, f, indent=2)

        print(f"✓ Saved arena config to: {filepath}")

    @staticmethod
    def load_arena(filepath: str) -> ArenaConfig:
        """Load arena configuration from file"""
        with open(filepath, 'r') as f:
            arena_dict = json.load(f)

        arena = ArenaConfig(**arena_dict)
        ARENAS[arena.name] = arena

        print(f"✓ Loaded arena: {arena.name}")
        return arena
