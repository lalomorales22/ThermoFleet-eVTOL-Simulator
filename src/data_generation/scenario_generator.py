"""
Synthetic Scenario Generator for ThermoFleet-eVTOL-Simulator

Generates diverse training scenarios including:
- Weather conditions (wind, rain, fog, snow, temperature)
- Traffic patterns (rush hour, emergencies, delivery routes)
- Failure modes (rotor, battery, sensors, communication)
- Edge cases (bird strikes, drone swarms, obstacles)

Author: ThermoFleet Team
Date: November 13, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class WeatherType(Enum):
    """Types of weather conditions"""
    CLEAR = "clear"
    WINDY = "windy"
    RAINY = "rainy"
    FOGGY = "foggy"
    SNOWY = "snowy"
    STORMY = "stormy"
    MIXED = "mixed"


class TrafficDensity(Enum):
    """Traffic density levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    RUSH_HOUR = "rush_hour"
    EMERGENCY = "emergency"


class FailureType(Enum):
    """Types of system failures"""
    ROTOR_FAILURE = "rotor_failure"
    BATTERY_DEGRADATION = "battery_degradation"
    GPS_DROPOUT = "gps_dropout"
    IMU_DRIFT = "imu_drift"
    COMMUNICATION_LOSS = "communication_loss"
    PAYLOAD_SHIFT = "payload_shift"
    SENSOR_MALFUNCTION = "sensor_malfunction"


class EdgeCaseType(Enum):
    """Types of edge cases"""
    BIRD_STRIKE = "bird_strike"
    DRONE_SWARM = "drone_swarm"
    BALLOON = "balloon"
    KITE = "kite"
    WIND_SHEAR = "wind_shear"
    NEAR_MISS = "near_miss"
    UNEXPECTED_OBSTACLE = "unexpected_obstacle"


@dataclass
class WeatherCondition:
    """Weather condition parameters"""
    weather_type: WeatherType
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    wind_gusts: bool
    turbulence_level: float  # 0-1
    precipitation: float  # 0-1 (rain/snow intensity)
    visibility: float  # meters
    temperature: float  # celsius
    temperature_gradient: float  # C/100m altitude
    dynamic_changes: bool
    thermal_activity: float  # 0-1 (thermals/updrafts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        result = asdict(self)
        result['weather_type'] = self.weather_type.value
        return result


@dataclass
class TrafficPattern:
    """Traffic pattern parameters"""
    density: TrafficDensity
    time_of_day: str  # morning, afternoon, evening, night
    num_vehicles: int
    hotspot_locations: List[Tuple[float, float, float]]  # (x, y, z) positions
    emergency_vehicles: int
    delivery_routes: List[List[Tuple[float, float, float]]]
    congestion_zones: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        result = asdict(self)
        result['density'] = self.density.value
        return result


@dataclass
class FailureMode:
    """Failure mode parameters"""
    failure_type: FailureType
    severity: float  # 0-1
    start_timestep: int
    duration: int  # timesteps
    recovery_possible: bool
    degradation_rate: float  # for gradual failures
    affected_components: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        result = asdict(self)
        result['failure_type'] = self.failure_type.value
        return result


@dataclass
class EdgeCase:
    """Edge case parameters"""
    case_type: EdgeCaseType
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    size: float
    timestep: int
    duration: int
    danger_level: float  # 0-1
    avoidable: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        result = asdict(self)
        result['case_type'] = self.case_type.value
        return result


@dataclass
class Scenario:
    """Complete scenario definition"""
    scenario_id: str
    difficulty: float  # 0-1
    weather: WeatherCondition
    traffic: TrafficPattern
    failures: List[FailureMode] = field(default_factory=list)
    edge_cases: List[EdgeCase] = field(default_factory=list)
    arena_bounds: Tuple[float, float, float, float, float, float] = field(
        default=(-1000, 1000, -1000, 1000, 120, 150)
    )
    max_timesteps: int = 1000
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'scenario_id': self.scenario_id,
            'difficulty': self.difficulty,
            'weather': self.weather.to_dict(),
            'traffic': self.traffic.to_dict(),
            'failures': [f.to_dict() for f in self.failures],
            'edge_cases': [e.to_dict() for e in self.edge_cases],
            'arena_bounds': self.arena_bounds,
            'max_timesteps': self.max_timesteps,
            'seed': self.seed
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class WeatherConditionGenerator:
    """
    Generates diverse weather conditions for training.
    
    Features:
    - Wind profiles (constant, gusts, turbulence, thermals)
    - Precipitation (rain, snow, fog)
    - Temperature gradients
    - Dynamic weather transitions
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize weather generator."""
        self.rng = np.random.default_rng(seed)
        logger.info("WeatherConditionGenerator initialized")
    
    def generate(
        self,
        weather_type: Optional[WeatherType] = None,
        severity: float = 0.5
    ) -> WeatherCondition:
        """
        Generate weather condition.
        
        Args:
            weather_type: Specific weather type or random if None
            severity: Weather severity (0=mild, 1=extreme)
            
        Returns:
            WeatherCondition object
        """
        if weather_type is None:
            weather_type = self.rng.choice(list(WeatherType))
        
        # Base parameters
        wind_speed = self._generate_wind_speed(weather_type, severity)
        wind_direction = self.rng.uniform(0, 360)
        
        # Weather-specific parameters
        if weather_type == WeatherType.CLEAR:
            precipitation = 0.0
            visibility = 10000.0
            turbulence = 0.1 * severity
            wind_gusts = False
            thermal_activity = 0.3 + 0.5 * severity
            
        elif weather_type == WeatherType.WINDY:
            precipitation = 0.0
            visibility = 8000.0
            turbulence = 0.3 + 0.5 * severity
            wind_gusts = severity > 0.5
            thermal_activity = 0.5 + 0.3 * severity
            
        elif weather_type == WeatherType.RAINY:
            precipitation = 0.3 + 0.7 * severity
            visibility = max(500, 5000 * (1 - severity))
            turbulence = 0.2 + 0.4 * severity
            wind_gusts = severity > 0.6
            thermal_activity = 0.1
            
        elif weather_type == WeatherType.FOGGY:
            precipitation = 0.1
            visibility = max(100, 1000 * (1 - severity))
            turbulence = 0.1
            wind_gusts = False
            thermal_activity = 0.0
            
        elif weather_type == WeatherType.SNOWY:
            precipitation = 0.4 + 0.6 * severity
            visibility = max(300, 3000 * (1 - severity))
            turbulence = 0.3 + 0.4 * severity
            wind_gusts = severity > 0.5
            thermal_activity = 0.0
            
        elif weather_type == WeatherType.STORMY:
            precipitation = 0.7 + 0.3 * severity
            visibility = max(200, 2000 * (1 - severity))
            turbulence = 0.6 + 0.4 * severity
            wind_gusts = True
            thermal_activity = 0.0
            wind_speed *= 1.5  # Higher wind in storms
            
        else:  # MIXED
            precipitation = self.rng.uniform(0, 0.5)
            visibility = self.rng.uniform(1000, 8000)
            turbulence = self.rng.uniform(0.2, 0.6)
            wind_gusts = self.rng.random() > 0.5
            thermal_activity = self.rng.uniform(0, 0.4)
        
        # Temperature (varies with weather)
        if weather_type == WeatherType.SNOWY:
            temperature = self.rng.uniform(-10, 5)
        elif weather_type == WeatherType.STORMY:
            temperature = self.rng.uniform(10, 25)
        else:
            temperature = self.rng.uniform(5, 30)
        
        # Temperature gradient (typically -6.5 C/km, but varies)
        temperature_gradient = -0.65 + self.rng.uniform(-0.2, 0.2)
        
        # Dynamic changes more likely in certain weather
        dynamic_changes = (
            weather_type in [WeatherType.STORMY, WeatherType.MIXED, WeatherType.WINDY]
            or self.rng.random() < 0.3
        )
        
        return WeatherCondition(
            weather_type=weather_type,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            wind_gusts=wind_gusts,
            turbulence_level=turbulence,
            precipitation=precipitation,
            visibility=visibility,
            temperature=temperature,
            temperature_gradient=temperature_gradient,
            dynamic_changes=dynamic_changes,
            thermal_activity=thermal_activity
        )
    
    def _generate_wind_speed(self, weather_type: WeatherType, severity: float) -> float:
        """Generate wind speed based on weather type and severity."""
        base_speeds = {
            WeatherType.CLEAR: (0, 5),
            WeatherType.WINDY: (10, 20),
            WeatherType.RAINY: (5, 15),
            WeatherType.FOGGY: (0, 3),
            WeatherType.SNOWY: (5, 15),
            WeatherType.STORMY: (15, 30),
            WeatherType.MIXED: (3, 18)
        }
        
        min_speed, max_speed = base_speeds[weather_type]
        return min_speed + (max_speed - min_speed) * severity
    
    def generate_batch(self, n_scenarios: int, difficulty_range: Tuple[float, float] = (0.0, 1.0)) -> List[WeatherCondition]:
        """Generate batch of weather conditions with varying difficulty."""
        scenarios = []
        for _ in range(n_scenarios):
            severity = self.rng.uniform(*difficulty_range)
            weather = self.generate(severity=severity)
            scenarios.append(weather)
        return scenarios


class TrafficPatternGenerator:
    """
    Generates realistic traffic patterns.
    
    Features:
    - Time-of-day variations (rush hour, off-peak)
    - Hotspot locations (airports, downtown, events)
    - Emergency vehicle priority
    - Delivery route congestion
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize traffic generator."""
        self.rng = np.random.default_rng(seed)
        logger.info("TrafficPatternGenerator initialized")
    
    def generate(
        self,
        density: Optional[TrafficDensity] = None,
        arena_bounds: Tuple[float, float, float, float, float, float] = (-1000, 1000, -1000, 1000, 120, 150),
        time_of_day: Optional[str] = None
    ) -> TrafficPattern:
        """
        Generate traffic pattern.
        
        Args:
            density: Traffic density or random if None
            arena_bounds: Arena boundaries (x_min, x_max, y_min, y_max, z_min, z_max)
            time_of_day: Time of day (morning, afternoon, evening, night)
            
        Returns:
            TrafficPattern object
        """
        if density is None:
            density = self.rng.choice(list(TrafficDensity))
        
        if time_of_day is None:
            time_of_day = self.rng.choice(['morning', 'afternoon', 'evening', 'night'])
        
        # Number of vehicles based on density
        vehicle_counts = {
            TrafficDensity.LOW: (5, 15),
            TrafficDensity.MEDIUM: (15, 40),
            TrafficDensity.HIGH: (40, 80),
            TrafficDensity.RUSH_HOUR: (80, 150),
            TrafficDensity.EMERGENCY: (10, 30)
        }
        
        min_v, max_v = vehicle_counts[density]
        num_vehicles = int(self.rng.uniform(min_v, max_v))
        
        # Adjust for time of day
        if time_of_day in ['morning', 'evening']:
            num_vehicles = int(num_vehicles * 1.3)  # Rush hour multiplier
        elif time_of_day == 'night':
            num_vehicles = int(num_vehicles * 0.4)  # Reduced night traffic
        
        # Generate hotspot locations
        num_hotspots = int(self.rng.uniform(2, 6))
        hotspot_locations = self._generate_hotspots(num_hotspots, arena_bounds)
        
        # Emergency vehicles
        if density == TrafficDensity.EMERGENCY:
            emergency_vehicles = int(self.rng.uniform(3, 8))
        else:
            emergency_vehicles = int(self.rng.uniform(0, 3))
        
        # Delivery routes (paths through arena)
        num_routes = int(self.rng.uniform(3, 10))
        delivery_routes = self._generate_delivery_routes(num_routes, arena_bounds)
        
        # Congestion zones
        num_congestion = int(self.rng.uniform(1, 4))
        congestion_zones = self._generate_congestion_zones(num_congestion, arena_bounds)
        
        return TrafficPattern(
            density=density,
            time_of_day=time_of_day,
            num_vehicles=num_vehicles,
            hotspot_locations=hotspot_locations,
            emergency_vehicles=emergency_vehicles,
            delivery_routes=delivery_routes,
            congestion_zones=congestion_zones
        )
    
    def _generate_hotspots(
        self,
        n: int,
        bounds: Tuple[float, float, float, float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """Generate hotspot locations."""
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        hotspots = []
        for _ in range(n):
            x = self.rng.uniform(x_min * 0.8, x_max * 0.8)
            y = self.rng.uniform(y_min * 0.8, y_max * 0.8)
            z = self.rng.uniform(z_min, z_max)
            hotspots.append((float(x), float(y), float(z)))
        return hotspots
    
    def _generate_delivery_routes(
        self,
        n: int,
        bounds: Tuple[float, float, float, float, float, float]
    ) -> List[List[Tuple[float, float, float]]]:
        """Generate delivery route paths."""
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        routes = []
        
        for _ in range(n):
            route_length = int(self.rng.uniform(3, 8))
            route = []
            for _ in range(route_length):
                x = self.rng.uniform(x_min, x_max)
                y = self.rng.uniform(y_min, y_max)
                z = self.rng.uniform(z_min, z_max)
                route.append((float(x), float(y), float(z)))
            routes.append(route)
        
        return routes
    
    def _generate_congestion_zones(
        self,
        n: int,
        bounds: Tuple[float, float, float, float, float, float]
    ) -> List[Dict[str, Any]]:
        """Generate congestion zones."""
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        zones = []
        
        for _ in range(n):
            center_x = self.rng.uniform(x_min * 0.7, x_max * 0.7)
            center_y = self.rng.uniform(y_min * 0.7, y_max * 0.7)
            center_z = self.rng.uniform(z_min, z_max)
            radius = self.rng.uniform(50, 200)
            severity = self.rng.uniform(0.3, 0.9)
            
            zones.append({
                'center': (float(center_x), float(center_y), float(center_z)),
                'radius': float(radius),
                'severity': float(severity)
            })
        
        return zones


class FailureModeInjector:
    """
    Injects realistic failure modes during episodes.
    
    Features:
    - Rotor failures (single/multiple)
    - Battery degradation
    - Sensor malfunctions (GPS, IMU, LiDAR)
    - Communication loss
    - Payload shifting
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize failure mode injector."""
        self.rng = np.random.default_rng(seed)
        logger.info("FailureModeInjector initialized")
    
    def generate(
        self,
        failure_type: Optional[FailureType] = None,
        severity: float = 0.5,
        max_timesteps: int = 1000
    ) -> FailureMode:
        """
        Generate a failure mode.
        
        Args:
            failure_type: Specific failure type or random if None
            severity: Failure severity (0=minor, 1=critical)
            max_timesteps: Maximum episode timesteps
            
        Returns:
            FailureMode object
        """
        if failure_type is None:
            failure_type = self.rng.choice(list(FailureType))
        
        # Start time (usually mid-episode, but can vary)
        start_timestep = int(self.rng.uniform(
            max_timesteps * 0.2,
            max_timesteps * 0.8
        ))
        
        # Duration depends on failure type and severity
        if failure_type == FailureType.COMMUNICATION_LOSS:
            # Brief to extended communication losses
            duration = int(self.rng.uniform(10, 100) * severity)
            recovery_possible = severity < 0.8
            affected = ['communication', 'telemetry']
            
        elif failure_type == FailureType.GPS_DROPOUT:
            # GPS can drop out temporarily
            duration = int(self.rng.uniform(20, 150) * severity)
            recovery_possible = True
            affected = ['gps', 'navigation']
            
        elif failure_type == FailureType.IMU_DRIFT:
            # IMU drift is gradual and persistent
            duration = max_timesteps - start_timestep
            recovery_possible = False
            affected = ['imu', 'attitude_estimation']
            
        elif failure_type == FailureType.ROTOR_FAILURE:
            # Rotor failure is sudden and serious
            duration = max_timesteps - start_timestep
            recovery_possible = severity < 0.5  # Can compensate if not too severe
            affected = ['rotor_1', 'thrust', 'stability'] if severity > 0.7 else ['rotor_1']
            
        elif failure_type == FailureType.BATTERY_DEGRADATION:
            # Battery degradation is gradual
            duration = max_timesteps - start_timestep
            recovery_possible = False
            affected = ['battery', 'power_system']
            
        elif failure_type == FailureType.PAYLOAD_SHIFT:
            # Sudden or gradual payload shift
            duration = int(self.rng.uniform(5, 50))
            recovery_possible = severity < 0.6
            affected = ['center_of_mass', 'stability']
            
        else:  # SENSOR_MALFUNCTION
            duration = int(self.rng.uniform(30, 200))
            recovery_possible = severity < 0.7
            sensor_types = ['lidar', 'camera', 'barometer', 'magnetometer']
            affected = [self.rng.choice(sensor_types)]
        
        # Degradation rate (for gradual failures)
        if failure_type in [FailureType.BATTERY_DEGRADATION, FailureType.IMU_DRIFT]:
            degradation_rate = self.rng.uniform(0.001, 0.01) * severity
        else:
            degradation_rate = 0.0
        
        return FailureMode(
            failure_type=failure_type,
            severity=severity,
            start_timestep=start_timestep,
            duration=duration,
            recovery_possible=recovery_possible,
            degradation_rate=degradation_rate,
            affected_components=affected
        )
    
    def generate_batch(
        self,
        n_failures: int,
        max_timesteps: int = 1000,
        severity_range: Tuple[float, float] = (0.2, 0.8)
    ) -> List[FailureMode]:
        """Generate multiple failure modes for a single episode."""
        failures = []
        for _ in range(n_failures):
            severity = self.rng.uniform(*severity_range)
            failure = self.generate(severity=severity, max_timesteps=max_timesteps)
            failures.append(failure)
        return failures


class EdgeCaseLibrary:
    """
    Library of edge cases and unexpected events.
    
    Features:
    - Bird strikes
    - Drone swarm interactions
    - Unexpected obstacles (balloons, kites)
    - Wind shear events
    - Near-miss scenarios with other aircraft
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize edge case library."""
        self.rng = np.random.default_rng(seed)
        logger.info("EdgeCaseLibrary initialized")
    
    def generate(
        self,
        case_type: Optional[EdgeCaseType] = None,
        arena_bounds: Tuple[float, float, float, float, float, float] = (-1000, 1000, -1000, 1000, 120, 150),
        max_timesteps: int = 1000
    ) -> EdgeCase:
        """
        Generate an edge case.
        
        Args:
            case_type: Specific edge case type or random if None
            arena_bounds: Arena boundaries
            max_timesteps: Maximum episode timesteps
            
        Returns:
            EdgeCase object
        """
        if case_type is None:
            case_type = self.rng.choice(list(EdgeCaseType))
        
        x_min, x_max, y_min, y_max, z_min, z_max = arena_bounds
        
        # Generate position within arena
        position = (
            float(self.rng.uniform(x_min, x_max)),
            float(self.rng.uniform(y_min, y_max)),
            float(self.rng.uniform(z_min, z_max))
        )
        
        # Timestep when edge case appears
        timestep = int(self.rng.uniform(50, max_timesteps * 0.9))
        
        # Case-specific parameters
        if case_type == EdgeCaseType.BIRD_STRIKE:
            # Birds move across path
            velocity = (
                float(self.rng.uniform(-15, 15)),
                float(self.rng.uniform(-15, 15)),
                float(self.rng.uniform(-5, 5))
            )
            size = self.rng.uniform(0.3, 1.0)  # meters
            duration = int(self.rng.uniform(10, 30))
            danger_level = 0.7
            avoidable = True
            
        elif case_type == EdgeCaseType.DRONE_SWARM:
            # Multiple small drones
            velocity = (
                float(self.rng.uniform(-10, 10)),
                float(self.rng.uniform(-10, 10)),
                float(self.rng.uniform(-3, 3))
            )
            size = self.rng.uniform(2.0, 5.0)  # swarm size
            duration = int(self.rng.uniform(20, 100))
            danger_level = 0.5
            avoidable = True
            
        elif case_type == EdgeCaseType.BALLOON:
            # Floating balloon
            velocity = (
                float(self.rng.uniform(-2, 2)),
                float(self.rng.uniform(-2, 2)),
                float(self.rng.uniform(0.5, 2))  # Rising
            )
            size = self.rng.uniform(1.0, 3.0)
            duration = int(self.rng.uniform(50, 200))
            danger_level = 0.3
            avoidable = True
            
        elif case_type == EdgeCaseType.KITE:
            # Kite on string
            velocity = (
                float(self.rng.uniform(-5, 5)),
                float(self.rng.uniform(-5, 5)),
                float(self.rng.uniform(-1, 1))
            )
            size = self.rng.uniform(1.0, 2.0)
            duration = int(self.rng.uniform(30, 150))
            danger_level = 0.4
            avoidable = True
            
        elif case_type == EdgeCaseType.WIND_SHEAR:
            # Sudden wind change
            velocity = (
                float(self.rng.uniform(-25, 25)),
                float(self.rng.uniform(-25, 25)),
                float(self.rng.uniform(-10, 10))
            )
            size = self.rng.uniform(50, 150)  # Affected area
            duration = int(self.rng.uniform(5, 20))  # Brief but intense
            danger_level = 0.8
            avoidable = False
            
        elif case_type == EdgeCaseType.NEAR_MISS:
            # Another aircraft passing close
            velocity = (
                float(self.rng.uniform(-40, 40)),
                float(self.rng.uniform(-40, 40)),
                float(self.rng.uniform(-10, 10))
            )
            size = self.rng.uniform(3.0, 6.0)
            duration = int(self.rng.uniform(10, 40))
            danger_level = 0.9
            avoidable = True
            
        else:  # UNEXPECTED_OBSTACLE
            # Generic unexpected obstacle
            velocity = (
                float(self.rng.uniform(-8, 8)),
                float(self.rng.uniform(-8, 8)),
                float(self.rng.uniform(-3, 3))
            )
            size = self.rng.uniform(0.5, 4.0)
            duration = int(self.rng.uniform(20, 100))
            danger_level = 0.6
            avoidable = True
        
        return EdgeCase(
            case_type=case_type,
            position=position,
            velocity=velocity,
            size=size,
            timestep=timestep,
            duration=duration,
            danger_level=danger_level,
            avoidable=avoidable
        )
    
    def generate_batch(
        self,
        n_cases: int,
        arena_bounds: Tuple[float, float, float, float, float, float] = (-1000, 1000, -1000, 1000, 120, 150),
        max_timesteps: int = 1000
    ) -> List[EdgeCase]:
        """Generate multiple edge cases for a single episode."""
        cases = []
        for _ in range(n_cases):
            case = self.generate(arena_bounds=arena_bounds, max_timesteps=max_timesteps)
            cases.append(case)
        return cases


class ScenarioGenerator:
    """
    Main scenario generator that combines all components.
    
    Generates complete scenarios with:
    - Weather conditions
    - Traffic patterns
    - Failure modes
    - Edge cases
    
    Supports curriculum learning with difficulty scaling.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize scenario generator."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Initialize sub-generators
        self.weather_gen = WeatherConditionGenerator(seed)
        self.traffic_gen = TrafficPatternGenerator(seed)
        self.failure_gen = FailureModeInjector(seed)
        self.edge_case_gen = EdgeCaseLibrary(seed)
        
        self.scenario_count = 0
        logger.info("ScenarioGenerator initialized")
    
    def generate(
        self,
        difficulty: float = 0.5,
        arena_bounds: Tuple[float, float, float, float, float, float] = (-1000, 1000, -1000, 1000, 120, 150),
        max_timesteps: int = 1000,
        include_failures: bool = True,
        include_edge_cases: bool = True
    ) -> Scenario:
        """
        Generate a complete scenario.
        
        Args:
            difficulty: Scenario difficulty (0=easy, 1=extreme)
            arena_bounds: Arena boundaries (x_min, x_max, y_min, y_max, z_min, z_max)
            max_timesteps: Maximum episode timesteps
            include_failures: Whether to include failure modes
            include_edge_cases: Whether to include edge cases
            
        Returns:
            Scenario object
        """
        scenario_id = f"scenario_{self.scenario_count:06d}"
        self.scenario_count += 1
        
        # Generate weather (severity scales with difficulty)
        weather_severity = difficulty * self.rng.uniform(0.7, 1.0)
        weather = self.weather_gen.generate(severity=weather_severity)
        
        # Generate traffic (density scales with difficulty)
        if difficulty < 0.3:
            density = TrafficDensity.LOW
        elif difficulty < 0.6:
            density = TrafficDensity.MEDIUM
        elif difficulty < 0.8:
            density = TrafficDensity.HIGH
        else:
            density = TrafficDensity.RUSH_HOUR
        
        traffic = self.traffic_gen.generate(
            density=density,
            arena_bounds=arena_bounds
        )
        
        # Generate failures (more failures at higher difficulty)
        failures = []
        if include_failures and self.rng.random() < difficulty:
            # Number of failures increases with difficulty
            n_failures = int(self.rng.poisson(difficulty * 2) + 1)
            n_failures = min(n_failures, 3)  # Cap at 3 simultaneous failures
            
            failures = self.failure_gen.generate_batch(
                n_failures=n_failures,
                max_timesteps=max_timesteps,
                severity_range=(difficulty * 0.3, difficulty)
            )
        
        # Generate edge cases (probability and number scale with difficulty)
        edge_cases = []
        if include_edge_cases and self.rng.random() < difficulty * 0.7:
            n_cases = int(self.rng.poisson(difficulty) + 1)
            n_cases = min(n_cases, 3)  # Cap at 3 edge cases
            
            edge_cases = self.edge_case_gen.generate_batch(
                n_cases=n_cases,
                arena_bounds=arena_bounds,
                max_timesteps=max_timesteps
            )
        
        scenario = Scenario(
            scenario_id=scenario_id,
            difficulty=difficulty,
            weather=weather,
            traffic=traffic,
            failures=failures,
            edge_cases=edge_cases,
            arena_bounds=arena_bounds,
            max_timesteps=max_timesteps,
            seed=self.seed
        )
        
        logger.info(
            f"Generated {scenario_id}: difficulty={difficulty:.2f}, "
            f"weather={weather.weather_type.value}, "
            f"traffic={traffic.density.value}, "
            f"failures={len(failures)}, "
            f"edge_cases={len(edge_cases)}"
        )
        
        return scenario
    
    def generate_batch(
        self,
        n_scenarios: int,
        difficulty_range: Tuple[float, float] = (0.0, 1.0),
        curriculum_learning: bool = False,
        **kwargs
    ) -> List[Scenario]:
        """
        Generate batch of scenarios.
        
        Args:
            n_scenarios: Number of scenarios to generate
            difficulty_range: Range of difficulties (min, max)
            curriculum_learning: If True, gradually increase difficulty
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            List of Scenario objects
        """
        scenarios = []
        
        for i in range(n_scenarios):
            if curriculum_learning:
                # Gradually increase difficulty
                difficulty = difficulty_range[0] + (difficulty_range[1] - difficulty_range[0]) * (i / max(n_scenarios - 1, 1))
            else:
                # Random difficulty in range
                difficulty = self.rng.uniform(*difficulty_range)
            
            scenario = self.generate(difficulty=difficulty, **kwargs)
            scenarios.append(scenario)
        
        logger.info(f"Generated batch of {n_scenarios} scenarios")
        return scenarios
    
    def save_scenarios(self, scenarios: List[Scenario], output_path: str):
        """Save scenarios to JSON file."""
        from pathlib import Path
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'scenarios': [s.to_dict() for s in scenarios],
            'count': len(scenarios),
            'generator_seed': self.seed
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(scenarios)} scenarios to {output_path}")
    
    def load_scenarios(self, input_path: str) -> List[Scenario]:
        """Load scenarios from JSON file."""
        from pathlib import Path
        import json
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # TODO: Implement full deserialization from dict to Scenario objects
        logger.info(f"Loaded {data['count']} scenarios from {input_path}")
        return []  # Placeholder

