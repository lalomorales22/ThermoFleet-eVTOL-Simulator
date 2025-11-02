"""
Edge Case Scenarios for eVTOL Training
Simulates various failure modes and challenging conditions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class EdgeCaseType(Enum):
    """Types of edge case scenarios."""
    WEATHER_CHANGE = "weather_change"
    GPS_FAILURE = "gps_failure"
    WIND_TURBULENCE = "wind_turbulence"
    SENSOR_FAILURE = "sensor_failure"
    BATTERY_DEGRADATION = "battery_degradation"
    MOTOR_FAILURE = "motor_failure"
    COMMUNICATION_LOSS = "communication_loss"
    BIRD_STRIKE = "bird_strike"
    SUDDEN_OBSTACLE = "sudden_obstacle"


class WeatherCondition(Enum):
    """Weather conditions."""
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"
    STORM = "storm"
    SNOW = "snow"


class EdgeCaseManager:
    """
    Manages edge case scenarios during training.

    Features:
    - Random or scheduled edge case injection
    - Multiple simultaneous failures
    - Difficulty progression
    - Scenario recording for analysis
    """

    def __init__(
        self,
        enable_weather: bool = True,
        enable_gps_failure: bool = True,
        enable_sensor_failure: bool = True,
        enable_turbulence: bool = True,
        edge_case_probability: float = 0.1,
        max_simultaneous_failures: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Initialize edge case manager.

        Args:
            enable_weather: Enable weather changes
            enable_gps_failure: Enable GPS failures
            enable_sensor_failure: Enable sensor failures
            enable_turbulence: Enable wind turbulence
            edge_case_probability: Probability of edge case per step
            max_simultaneous_failures: Max number of concurrent failures
            seed: Random seed
        """
        self.enable_weather = enable_weather
        self.enable_gps_failure = enable_gps_failure
        self.enable_sensor_failure = enable_sensor_failure
        self.enable_turbulence = enable_turbulence
        self.edge_case_probability = edge_case_probability
        self.max_simultaneous_failures = max_simultaneous_failures

        self.rng = np.random.default_rng(seed)

        # Active scenarios
        self.active_scenarios: List[Dict[str, Any]] = []
        self.weather_condition = WeatherCondition.CLEAR

        # Statistics
        self.scenario_count = {etype: 0 for etype in EdgeCaseType}
        self.total_scenarios = 0

        logger.info("Initialized EdgeCaseManager")

    def reset(self):
        """Reset all active scenarios."""
        self.active_scenarios.clear()
        self.weather_condition = WeatherCondition.CLEAR

    def step(self, current_step: int) -> Dict[str, Any]:
        """
        Update edge cases for current step.

        Args:
            current_step: Current simulation step

        Returns:
            Dictionary of active effects
        """
        # Update existing scenarios
        self._update_scenarios(current_step)

        # Possibly trigger new scenario
        if self.rng.random() < self.edge_case_probability:
            if len(self.active_scenarios) < self.max_simultaneous_failures:
                self._trigger_random_scenario(current_step)

        # Compile effects
        effects = self._compile_effects()

        return effects

    def _update_scenarios(self, current_step: int):
        """Update and remove expired scenarios."""
        self.active_scenarios = [
            scenario for scenario in self.active_scenarios
            if scenario["end_step"] > current_step
        ]

    def _trigger_random_scenario(self, current_step: int):
        """Trigger a random edge case scenario."""
        # Select available scenario types
        available = []

        if self.enable_weather:
            available.append(EdgeCaseType.WEATHER_CHANGE)

        if self.enable_gps_failure:
            available.append(EdgeCaseType.GPS_FAILURE)

        if self.enable_sensor_failure:
            available.append(EdgeCaseType.SENSOR_FAILURE)

        if self.enable_turbulence:
            available.append(EdgeCaseType.WIND_TURBULENCE)

        # Additional scenarios
        available.extend([
            EdgeCaseType.BATTERY_DEGRADATION,
            EdgeCaseType.MOTOR_FAILURE,
            EdgeCaseType.BIRD_STRIKE,
        ])

        if not available:
            return

        # Select random scenario
        scenario_type = self.rng.choice(available)

        # Create scenario
        scenario = self._create_scenario(scenario_type, current_step)

        if scenario:
            self.active_scenarios.append(scenario)
            self.scenario_count[scenario_type] += 1
            self.total_scenarios += 1
            logger.info(f"Triggered {scenario_type.value} at step {current_step}")

    def _create_scenario(self, scenario_type: EdgeCaseType, start_step: int) -> Optional[Dict[str, Any]]:
        """Create a specific scenario."""
        duration = self.rng.integers(50, 200)  # 1-4 seconds at 50Hz

        scenario = {
            "type": scenario_type,
            "start_step": start_step,
            "end_step": start_step + duration,
        }

        if scenario_type == EdgeCaseType.WEATHER_CHANGE:
            scenario["weather"] = self.rng.choice(list(WeatherCondition))
            scenario["visibility"] = self.rng.uniform(0.3, 1.0)
            scenario["wind_increase"] = self.rng.uniform(1.5, 3.0)

        elif scenario_type == EdgeCaseType.GPS_FAILURE:
            scenario["failure_type"] = self.rng.choice(["complete", "degraded", "drift"])
            scenario["noise_level"] = self.rng.uniform(5.0, 20.0)  # meters

        elif scenario_type == EdgeCaseType.WIND_TURBULENCE:
            scenario["intensity"] = self.rng.uniform(1.5, 4.0)
            scenario["direction_variance"] = self.rng.uniform(30, 90)  # degrees

        elif scenario_type == EdgeCaseType.SENSOR_FAILURE:
            scenario["sensor_type"] = self.rng.choice(["lidar", "imu", "camera"])
            scenario["failure_mode"] = self.rng.choice(["complete", "noisy", "stuck"])

        elif scenario_type == EdgeCaseType.BATTERY_DEGRADATION:
            scenario["drain_multiplier"] = self.rng.uniform(1.5, 3.0)

        elif scenario_type == EdgeCaseType.MOTOR_FAILURE:
            scenario["motor_id"] = self.rng.integers(0, 4)
            scenario["power_reduction"] = self.rng.uniform(0.3, 0.7)

        elif scenario_type == EdgeCaseType.BIRD_STRIKE:
            scenario["impact_force"] = self.rng.uniform(100, 500)  # Newtons
            scenario["impact_direction"] = self.rng.uniform(0, 2 * np.pi)
            scenario["duration"] = 5  # Very short

        return scenario

    def _compile_effects(self) -> Dict[str, Any]:
        """Compile all active scenario effects."""
        effects = {
            "gps_noise": np.zeros(3),
            "wind_force": np.zeros(3),
            "sensor_failures": {},
            "battery_drain_multiplier": 1.0,
            "motor_power": np.ones(4),
            "visibility": 1.0,
            "weather": WeatherCondition.CLEAR,
        }

        for scenario in self.active_scenarios:
            stype = scenario["type"]

            if stype == EdgeCaseType.WEATHER_CHANGE:
                effects["weather"] = scenario["weather"]
                effects["visibility"] = min(effects["visibility"], scenario["visibility"])
                effects["wind_force"] += np.array([
                    self.rng.uniform(-1, 1),
                    self.rng.uniform(-1, 1),
                    self.rng.uniform(-0.5, 0.5),
                ]) * scenario["wind_increase"]

            elif stype == EdgeCaseType.GPS_FAILURE:
                if scenario["failure_type"] == "complete":
                    effects["gps_noise"] = np.array([999.0, 999.0, 999.0])
                elif scenario["failure_type"] == "degraded":
                    effects["gps_noise"] += self.rng.uniform(
                        -scenario["noise_level"],
                        scenario["noise_level"],
                        size=3
                    )
                elif scenario["failure_type"] == "drift":
                    # Cumulative drift
                    drift_rate = 0.1
                    steps_elapsed = max(1, scenario["end_step"] - scenario["start_step"])
                    effects["gps_noise"] += np.array([
                        drift_rate * steps_elapsed,
                        drift_rate * steps_elapsed,
                        0.0
                    ])

            elif stype == EdgeCaseType.WIND_TURBULENCE:
                # Random turbulence
                angle = self.rng.uniform(0, 2 * np.pi)
                intensity = scenario["intensity"]
                effects["wind_force"] += intensity * np.array([
                    np.cos(angle),
                    np.sin(angle),
                    self.rng.uniform(-0.3, 0.3),
                ])

            elif stype == EdgeCaseType.SENSOR_FAILURE:
                sensor = scenario["sensor_type"]
                mode = scenario["failure_mode"]
                effects["sensor_failures"][sensor] = mode

            elif stype == EdgeCaseType.BATTERY_DEGRADATION:
                effects["battery_drain_multiplier"] *= scenario["drain_multiplier"]

            elif stype == EdgeCaseType.MOTOR_FAILURE:
                motor_id = scenario["motor_id"]
                effects["motor_power"][motor_id] *= (1.0 - scenario["power_reduction"])

            elif stype == EdgeCaseType.BIRD_STRIKE:
                # Sudden impulse force
                angle = scenario["impact_direction"]
                force = scenario["impact_force"]
                effects["wind_force"] += force * np.array([
                    np.cos(angle),
                    np.sin(angle),
                    -0.5,  # Downward component
                ])

        return effects

    def get_statistics(self) -> Dict[str, Any]:
        """Get edge case statistics."""
        return {
            "total_scenarios": self.total_scenarios,
            "active_scenarios": len(self.active_scenarios),
            "scenario_counts": {
                etype.value: count
                for etype, count in self.scenario_count.items()
            },
            "current_weather": self.weather_condition.value,
        }


class EdgeCaseEvaluator:
    """Evaluate agent performance under edge cases."""

    def __init__(self, edge_case_manager: EdgeCaseManager):
        """
        Initialize evaluator.

        Args:
            edge_case_manager: Edge case manager instance
        """
        self.manager = edge_case_manager
        self.results = []

    def evaluate_scenario(
        self,
        env,
        agent,
        scenario_type: EdgeCaseType,
        n_episodes: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate agent on specific scenario type.

        Args:
            env: Environment instance
            agent: Trained agent
            scenario_type: Type of scenario to test
            n_episodes: Number of test episodes

        Returns:
            Performance metrics
        """
        success_count = 0
        total_rewards = []
        scenario_durations = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0

            # Inject specific scenario
            scenario_step = np.random.randint(50, 200)

            while not done and steps < 1000:
                # Apply edge case at specific step
                if steps == scenario_step:
                    scenario = self.manager._create_scenario(scenario_type, steps)
                    if scenario:
                        self.manager.active_scenarios.append(scenario)

                # Get action from agent
                action, _ = agent.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                steps += 1

            # Check success (reached goal without crashing)
            if info.get("goal_reached", False):
                success_count += 1

            total_rewards.append(episode_reward)
            scenario_durations.append(steps)

        # Compile results
        results = {
            "scenario_type": scenario_type.value,
            "success_rate": success_count / n_episodes,
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_duration": np.mean(scenario_durations),
        }

        self.results.append(results)
        logger.info(f"Scenario {scenario_type.value}: Success={results['success_rate']:.2%}")

        return results

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results."""
        return self.results


if __name__ == "__main__":
    # Test edge case manager
    logging.basicConfig(level=logging.INFO)

    manager = EdgeCaseManager(edge_case_probability=0.2)

    # Simulate 1000 steps
    for step in range(1000):
        effects = manager.step(step)

        if step % 100 == 0:
            stats = manager.get_statistics()
            logger.info(f"Step {step}: {stats}")

    print("\nFinal Statistics:")
    print(manager.get_statistics())
