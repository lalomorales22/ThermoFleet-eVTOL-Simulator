"""
Enhanced eVTOL Gymnasium Environment with Scenario Generation

This environment extends the base EVTOLEnv with:
- Synthetic scenario generation (weather, traffic, failures, edge cases)
- Dynamic scenario application during episodes
- Scenario tracking and logging

Author: ThermoFleet Team
Date: November 13, 2025
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, Any, List
import logging

from .evtol_gym_env import EVTOLEnv
from ..data_generation.scenario_generator import (
    ScenarioGenerator,
    Scenario,
    WeatherType,
    TrafficDensity,
    FailureType,
    EdgeCaseType
)

logger = logging.getLogger(__name__)


class EVTOLScenarioEnv(EVTOLEnv):
    """
    Enhanced eVTOL environment with scenario generation support.
    
    Features:
    - Automatic scenario generation with configurable difficulty
    - Weather simulation (wind, turbulence, visibility)
    - Traffic patterns (other vehicles, congestion)
    - Failure mode injection (sensors, rotors, battery)
    - Edge case handling (bird strikes, wind shear, near-misses)
    - Curriculum learning support
    """
    
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
        # Scenario generation parameters
        enable_scenarios: bool = True,
        scenario_difficulty: float = 0.5,
        auto_generate_scenarios: bool = True,
        scenario_seed: Optional[int] = None,
        curriculum_learning: bool = False,
        # Scenario components toggles
        enable_weather_scenarios: bool = True,
        enable_traffic_scenarios: bool = True,
        enable_failure_scenarios: bool = True,
        enable_edge_cases: bool = True,
        # Specific scenario constraints (optional)
        weather_type: Optional[str] = None,  # Force specific weather type
        traffic_density: Optional[str] = None,  # Force specific traffic density
    ):
        """
        Initialize enhanced eVTOL environment with scenarios.
        
        Args:
            vehicle_type: Type of vehicle ("small", "medium", "large")
            max_steps: Maximum steps per episode
            altitude_min: Minimum allowed altitude (feet)
            altitude_max: Maximum allowed altitude (feet)
            arena_bounds: (x, y, z) bounds of the arena in meters
            enable_wind: Whether to simulate wind turbulence
            enable_sensor_noise: Whether to add noise to sensor readings
            render_mode: Rendering mode (None, "human", "rgb_array")
            enable_scenarios: Whether to use scenario generation
            scenario_difficulty: Base scenario difficulty (0-1)
            auto_generate_scenarios: Auto-generate new scenario each reset
            scenario_seed: Random seed for scenario generation
            curriculum_learning: Gradually increase difficulty over episodes
            enable_weather_scenarios: Include weather conditions
            enable_traffic_scenarios: Include traffic patterns
            enable_failure_scenarios: Include failure modes
            enable_edge_cases: Include edge cases
        """
        # Initialize base environment
        super().__init__(
            vehicle_type=vehicle_type,
            max_steps=max_steps,
            altitude_min=altitude_min,
            altitude_max=altitude_max,
            arena_bounds=arena_bounds,
            enable_wind=enable_wind,
            enable_sensor_noise=enable_sensor_noise,
            render_mode=render_mode
        )
        
        # Scenario generation setup
        self.enable_scenarios = enable_scenarios
        self.scenario_difficulty = scenario_difficulty
        self.auto_generate_scenarios = auto_generate_scenarios
        self.curriculum_learning = curriculum_learning
        self.episode_count = 0
        
        # Scenario components
        self.enable_weather_scenarios = enable_weather_scenarios
        self.enable_traffic_scenarios = enable_traffic_scenarios
        self.enable_failure_scenarios = enable_failure_scenarios
        self.enable_edge_cases = enable_edge_cases
        
        # Specific scenario constraints
        self.forced_weather_type = weather_type
        self.forced_traffic_density = traffic_density
        
        # Initialize scenario generator
        if self.enable_scenarios:
            self.scenario_generator = ScenarioGenerator(seed=scenario_seed)
            self.current_scenario: Optional[Scenario] = None
            logger.info(f"EVTOLScenarioEnv initialized with scenario generation" + 
                       (f" (forced weather: {weather_type})" if weather_type else "") +
                       (f" (forced traffic: {traffic_density})" if traffic_density else ""))
        else:
            self.scenario_generator = None
            self.current_scenario = None
            logger.info("EVTOLScenarioEnv initialized without scenarios (using defaults)")
        
        # Scenario state tracking
        self.active_failures: List[Any] = []
        self.active_edge_cases: List[Any] = []
        self.weather_effects: Dict[str, float] = {}
        self.traffic_vehicles: List[Dict[str, Any]] = []
    
    def set_scenario(self, scenario: Scenario):
        """
        Set a specific scenario to use.
        
        Args:
            scenario: Scenario object to apply
        """
        self.current_scenario = scenario
        logger.info(f"Set scenario: {scenario.scenario_id}, difficulty={scenario.difficulty:.2f}")
    
    def get_current_scenario(self) -> Optional[Scenario]:
        """Get the current scenario being used."""
        return self.current_scenario
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and optionally generate new scenario."""
        # Standard reset
        observation, info = super().reset(seed=seed, options=options)
        
        # Generate or use provided scenario
        if self.enable_scenarios and self.auto_generate_scenarios:
            # Calculate difficulty for curriculum learning
            if self.curriculum_learning:
                # Gradually increase difficulty over episodes
                max_difficulty = 1.0
                min_difficulty = 0.0
                curriculum_episodes = 1000  # Reach max difficulty after 1000 episodes
                
                progress = min(self.episode_count / curriculum_episodes, 1.0)
                current_difficulty = min_difficulty + (max_difficulty - min_difficulty) * progress
            else:
                current_difficulty = self.scenario_difficulty
            
            # Generate scenario
            arena_bounds_full = (
                -self.arena_bounds[0] / 2, self.arena_bounds[0] / 2,
                -self.arena_bounds[1] / 2, self.arena_bounds[1] / 2,
                self.altitude_min, self.altitude_max
            )
            
            self.current_scenario = self.scenario_generator.generate(
                difficulty=current_difficulty,
                arena_bounds=arena_bounds_full,
                max_timesteps=self.max_steps,
                include_failures=self.enable_failure_scenarios,
                include_edge_cases=self.enable_edge_cases
            )
            
            # Override with forced weather/traffic types if specified
            if self.forced_weather_type:
                try:
                    forced_weather = WeatherType(self.forced_weather_type)
                    # Generate a new weather condition with the forced type
                    self.current_scenario.weather = self.scenario_generator.weather_gen.generate(
                        weather_type=forced_weather,
                        severity=current_difficulty
                    )
                except ValueError:
                    logger.warning(f"Invalid weather type: {self.forced_weather_type}, using generated weather")
            
            if self.forced_traffic_density:
                try:
                    forced_density = TrafficDensity(self.forced_traffic_density)
                    # Generate a new traffic pattern with the forced density
                    self.current_scenario.traffic = self.scenario_generator.traffic_gen.generate(
                        density=forced_density,
                        arena_bounds=arena_bounds_full
                    )
                except ValueError:
                    logger.warning(f"Invalid traffic density: {self.forced_traffic_density}, using generated traffic")
            
            logger.debug(f"Generated scenario: {self.current_scenario.scenario_id}")
        
        # Apply scenario if available
        if self.current_scenario and self.enable_scenarios:
            self._apply_scenario()
            
            # Add scenario info to info dict
            info['scenario'] = {
                'scenario_id': self.current_scenario.scenario_id,
                'difficulty': self.current_scenario.difficulty,
                'weather_type': self.current_scenario.weather.weather_type.value,
                'traffic_density': self.current_scenario.traffic.density.value,
                'num_failures': len(self.current_scenario.failures),
                'num_edge_cases': len(self.current_scenario.edge_cases)
            }
        
        self.episode_count += 1
        
        return observation, info
    
    def _apply_scenario(self):
        """Apply the current scenario to the environment."""
        if not self.current_scenario:
            return
        
        # Apply weather conditions
        if self.enable_weather_scenarios:
            self._apply_weather(self.current_scenario.weather)
        
        # Setup traffic pattern
        if self.enable_traffic_scenarios:
            self._setup_traffic(self.current_scenario.traffic)
        
        # Prepare failures (will be injected during episode)
        self.active_failures = self.current_scenario.failures.copy()
        
        # Prepare edge cases (will be triggered during episode)
        self.active_edge_cases = self.current_scenario.edge_cases.copy()
    
    def _apply_weather(self, weather):
        """
        Apply weather conditions to the environment.
        
        Args:
            weather: WeatherCondition object
        """
        # Wind
        wind_magnitude = weather.wind_speed
        wind_angle = np.radians(weather.wind_direction)
        
        self.wind_velocity = np.array([
            wind_magnitude * np.cos(wind_angle),
            wind_magnitude * np.sin(wind_angle),
            0.0  # Horizontal wind primarily
        ])
        
        # Store weather effects for dynamics calculation
        self.weather_effects = {
            'wind_speed': weather.wind_speed,
            'wind_gusts': weather.wind_gusts,
            'turbulence': weather.turbulence_level,
            'precipitation': weather.precipitation,
            'visibility': weather.visibility,
            'temperature': weather.temperature,
            'thermal_activity': weather.thermal_activity
        }
        
        logger.debug(
            f"Applied weather: {weather.weather_type.value}, "
            f"wind={weather.wind_speed:.1f}m/s, "
            f"turbulence={weather.turbulence_level:.2f}"
        )
    
    def _setup_traffic(self, traffic):
        """
        Setup traffic pattern.
        
        Args:
            traffic: TrafficPattern object
        """
        # Create traffic vehicles (simplified - just store metadata for now)
        self.traffic_vehicles = []
        
        for i in range(traffic.num_vehicles):
            # Simple traffic vehicle with random position/velocity
            vehicle = {
                'id': i,
                'position': self.np_random.uniform(
                    [-self.arena_bounds[0]/2, -self.arena_bounds[1]/2, self.altitude_min],
                    [self.arena_bounds[0]/2, self.arena_bounds[1]/2, self.altitude_max]
                ),
                'velocity': self.np_random.uniform(-20, 20, size=3),
                'is_emergency': i < traffic.emergency_vehicles
            }
            self.traffic_vehicles.append(vehicle)
        
        logger.debug(
            f"Setup traffic: {traffic.density.value}, "
            f"{traffic.num_vehicles} vehicles, "
            f"{traffic.emergency_vehicles} emergency"
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with scenario effects.
        
        Args:
            action: Array of [thrust, roll, pitch, yaw]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Check for failure injection
        if self.enable_scenarios and self.enable_failure_scenarios:
            self._check_inject_failures()
        
        # Check for edge case triggers
        if self.enable_scenarios and self.enable_edge_cases:
            self._check_trigger_edge_cases()
        
        # Update traffic (simple simulation)
        if self.enable_scenarios and self.enable_traffic_scenarios:
            self._update_traffic()
        
        # Execute standard step
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Add scenario-specific info
        if self.current_scenario:
            info['scenario_effects'] = {
                'active_failures': len([f for f in self.active_failures if f.start_timestep <= self.current_step]),
                'weather_wind': self.weather_effects.get('wind_speed', 0.0),
                'weather_turbulence': self.weather_effects.get('turbulence', 0.0),
                'nearby_traffic': len(self.traffic_vehicles)
            }
        
        return observation, reward, terminated, truncated, info
    
    def _check_inject_failures(self):
        """Check if any failures should be injected at current timestep."""
        for failure in self.active_failures:
            if failure.start_timestep == self.current_step:
                self._inject_failure(failure)
    
    def _inject_failure(self, failure):
        """
        Inject a specific failure mode.
        
        Args:
            failure: FailureMode object
        """
        logger.debug(
            f"Injecting failure: {failure.failure_type.value} "
            f"at timestep {self.current_step}, severity={failure.severity:.2f}"
        )
        
        # Apply failure effects based on type
        if failure.failure_type == FailureType.GPS_DROPOUT:
            # GPS failure increases sensor noise significantly
            self.enable_sensor_noise = True
            # In _get_observation, this will increase GPS noise
            
        elif failure.failure_type == FailureType.IMU_DRIFT:
            # IMU drift affects orientation estimation
            # Gradually accumulate error
            drift = failure.degradation_rate * (self.current_step - failure.start_timestep)
            self.orientation += self.np_random.uniform(-drift, drift, size=3)
            
        elif failure.failure_type == FailureType.ROTOR_FAILURE:
            # Rotor failure reduces available thrust
            self.vehicle_params["max_thrust"] *= (1.0 - failure.severity * 0.5)
            
        elif failure.failure_type == FailureType.BATTERY_DEGRADATION:
            # Battery degradation increases consumption
            additional_drain = failure.degradation_rate * failure.severity
            self.battery_level = max(0.0, self.battery_level - additional_drain)
            
        elif failure.failure_type == FailureType.COMMUNICATION_LOSS:
            # Communication loss might affect command latency (simplified)
            pass
            
        elif failure.failure_type == FailureType.PAYLOAD_SHIFT:
            # Payload shift affects center of mass (simplified as angular disturbance)
            self.angular_velocity += self.np_random.uniform(-2, 2, size=3) * failure.severity
    
    def _check_trigger_edge_cases(self):
        """Check if any edge cases should be triggered at current timestep."""
        for edge_case in self.active_edge_cases:
            if edge_case.timestep == self.current_step:
                self._trigger_edge_case(edge_case)
    
    def _trigger_edge_case(self, edge_case):
        """
        Trigger a specific edge case.
        
        Args:
            edge_case: EdgeCase object
        """
        logger.debug(
            f"Triggering edge case: {edge_case.case_type.value} "
            f"at timestep {self.current_step}, danger={edge_case.danger_level:.2f}"
        )
        
        # Check if edge case is near vehicle
        distance = np.linalg.norm(self.position - np.array(edge_case.position))
        
        if distance < edge_case.size * 10:  # Within interaction range
            
            if edge_case.case_type == EdgeCaseType.BIRD_STRIKE:
                # Small collision-like event
                if distance < edge_case.size:
                    self.collision_count += 1
                    logger.warning(f"Bird strike at {self.position}!")
                
            elif edge_case.case_type == EdgeCaseType.WIND_SHEAR:
                # Sudden wind change
                shear_force = np.array(edge_case.velocity) * edge_case.danger_level
                self.velocity += shear_force * 0.1  # Apply impulse
                
            elif edge_case.case_type == EdgeCaseType.NEAR_MISS:
                # Close encounter with another aircraft
                # Add to collision count if too close
                if distance < edge_case.size:
                    self.collision_count += 1
                    logger.warning(f"Near-miss event at {self.position}!")
    
    def _update_traffic(self):
        """Update traffic vehicle positions (simple simulation)."""
        dt = 0.02  # 50 Hz
        
        for vehicle in self.traffic_vehicles:
            # Simple linear motion
            vehicle['position'] += vehicle['velocity'] * dt
            
            # Wrap around arena bounds
            for i in range(2):  # x, y
                if vehicle['position'][i] < -self.arena_bounds[i] / 2:
                    vehicle['position'][i] = self.arena_bounds[i] / 2
                elif vehicle['position'][i] > self.arena_bounds[i] / 2:
                    vehicle['position'][i] = -self.arena_bounds[i] / 2
    
    def _apply_dynamics(self, thrust: float, roll: float, pitch: float, yaw: float):
        """Override to include scenario weather effects."""
        # Apply base dynamics
        super()._apply_dynamics(thrust, roll, pitch, yaw)
        
        # Apply additional scenario effects
        if self.enable_scenarios and self.weather_effects:
            dt = 0.02
            
            # Turbulence (random perturbations)
            if self.weather_effects.get('turbulence', 0) > 0:
                turbulence = self.weather_effects['turbulence']
                turb_force = self.np_random.uniform(-turbulence, turbulence, size=3) * 50
                self.velocity += turb_force * dt / self.vehicle_params['mass']
            
            # Wind gusts (sudden wind spikes)
            if self.weather_effects.get('wind_gusts', False):
                if self.np_random.random() < 0.01:  # 1% chance per step
                    gust = self.np_random.uniform(0.5, 1.5) * self.wind_velocity
                    self.velocity += gust * dt
            
            # Thermals (updrafts/downdrafts)
            if self.weather_effects.get('thermal_activity', 0) > 0:
                thermal = self.weather_effects['thermal_activity']
                if self.np_random.random() < 0.05:  # 5% chance
                    vertical_force = self.np_random.uniform(-thermal, thermal) * 100
                    self.velocity[2] += vertical_force * dt / self.vehicle_params['mass']
            
            # Temperature effects on battery (simplified)
            temp = self.weather_effects.get('temperature', 20)
            if temp < 0 or temp > 35:  # Extreme temperatures
                temp_factor = 1.0 + abs(temp - 20) * 0.01
                # Battery drains faster in extreme temps (handled in base dynamics)
    
    def _get_observation(self) -> np.ndarray:
        """Override to include scenario sensor effects."""
        observation = super()._get_observation()
        
        # Apply scenario-based sensor degradation
        if self.enable_scenarios and self.active_failures:
            for failure in self.active_failures:
                if failure.start_timestep <= self.current_step < failure.start_timestep + failure.duration:
                    
                    if failure.failure_type == FailureType.GPS_DROPOUT:
                        # Severely degrade GPS (indices 18-20 in observation)
                        observation[18:21] += self.np_random.uniform(-10, 10, size=3) * failure.severity
                    
                    elif failure.failure_type == FailureType.IMU_DRIFT:
                        # Degrade IMU readings (indices 21-23)
                        drift_error = failure.degradation_rate * (self.current_step - failure.start_timestep)
                        observation[21:24] += self.np_random.uniform(-drift_error, drift_error, size=3)
                    
                    elif failure.failure_type == FailureType.SENSOR_MALFUNCTION:
                        # Degrade LiDAR (indices 12-19)
                        if 'lidar' in failure.affected_components:
                            observation[12:20] *= (1.0 - failure.severity * 0.5)
                            observation[12:20] += self.np_random.uniform(-5, 5, size=8) * failure.severity
        
        # Visibility effects from weather
        if self.enable_scenarios and self.weather_effects:
            visibility = self.weather_effects.get('visibility', 10000)
            if visibility < 1000:  # Low visibility
                # Degrade LiDAR range
                visibility_factor = visibility / 1000.0
                observation[12:20] *= visibility_factor
        
        return observation
    
    def _calculate_reward(self) -> float:
        """Override to include scenario-specific rewards/penalties."""
        reward = super()._calculate_reward()
        
        # Additional scenario-based penalties/bonuses
        if self.enable_scenarios:
            
            # Penalty for operating in difficult weather
            if self.weather_effects.get('turbulence', 0) > 0.5:
                reward -= 0.05  # Small penalty for difficult conditions
            
            # Bonus for successful operation under failures
            active_failures = sum(
                1 for f in self.active_failures 
                if f.start_timestep <= self.current_step < f.start_timestep + f.duration
            )
            if active_failures > 0:
                # Small bonus for surviving with failures
                reward += 0.1 * active_failures
            
            # Penalty for near-misses with traffic
            if self.traffic_vehicles:
                min_distance = float('inf')
                for vehicle in self.traffic_vehicles:
                    dist = np.linalg.norm(self.position - vehicle['position'])
                    min_distance = min(min_distance, dist)
                
                if min_distance < 20.0:  # Very close
                    reward -= 1.0
                elif min_distance < 50.0:  # Close
                    reward -= 0.2
        
        return reward
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """
        Get summary of current scenario and its effects.
        
        Returns:
            Dictionary with scenario information
        """
        if not self.current_scenario:
            return {}
        
        return {
            'scenario_id': self.current_scenario.scenario_id,
            'difficulty': self.current_scenario.difficulty,
            'weather': {
                'type': self.current_scenario.weather.weather_type.value,
                'wind_speed': self.current_scenario.weather.wind_speed,
                'turbulence': self.current_scenario.weather.turbulence_level,
                'visibility': self.current_scenario.weather.visibility,
            },
            'traffic': {
                'density': self.current_scenario.traffic.density.value,
                'num_vehicles': self.current_scenario.traffic.num_vehicles,
                'emergency_vehicles': self.current_scenario.traffic.emergency_vehicles,
            },
            'failures': [
                {
                    'type': f.failure_type.value,
                    'severity': f.severity,
                    'start': f.start_timestep,
                    'duration': f.duration
                }
                for f in self.current_scenario.failures
            ],
            'edge_cases': [
                {
                    'type': e.case_type.value,
                    'danger_level': e.danger_level,
                    'timestep': e.timestep
                }
                for e in self.current_scenario.edge_cases
            ],
            'episode_count': self.episode_count,
            'current_step': self.current_step
        }

