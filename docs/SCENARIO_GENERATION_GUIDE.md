# Scenario Generation System - Complete Guide

**Priority 1.1 Implementation: Synthetic Scenario Generator**

This guide documents the complete implementation of the synthetic scenario generation system, including weather conditions, traffic patterns, failure modes, and edge cases.

---

## 🎯 Overview

The Scenario Generation System automatically creates diverse training scenarios to make your eVTOL agents more robust. It includes:

- **Weather Conditions**: Wind, rain, fog, snow, storms, turbulence
- **Traffic Patterns**: Rush hour, emergency vehicles, delivery routes, congestion
- **Failure Modes**: Rotor failures, battery degradation, sensor malfunctions, communication loss
- **Edge Cases**: Bird strikes, drone swarms, wind shear, near-misses

All scenarios are logged to the database for analysis and tracking.

---

## 📁 New Files Created

### Core Implementation
- `src/data_generation/__init__.py` - Module exports
- `src/data_generation/scenario_generator.py` - **Main implementation** (1,100+ lines)
- `src/environments/evtol_scenario_env.py` - Enhanced environment with scenario support
- `src/database/scenario_logger.py` - Database logging for scenarios

### Database
- `scripts/migrate_db_scenarios.py` - Database schema migration
- New tables: `scenario_templates`, `episode_scenarios`, `scenario_metrics`

### Testing
- `scripts/test_scenario_generation.py` - Comprehensive test suite

---

## 🚀 Quick Start

### 1. Run Database Migration

First, add the scenario tracking tables to your database:

```bash
python scripts/migrate_db_scenarios.py
```

### 2. Basic Usage

```python
from src.environments import EVTOLScenarioEnv

# Create environment with scenario generation
env = EVTOLScenarioEnv(
    vehicle_type="medium",
    enable_scenarios=True,
    scenario_difficulty=0.5,  # 0=easy, 1=extreme
    enable_weather_scenarios=True,
    enable_traffic_scenarios=True,
    enable_failure_scenarios=True,
    enable_edge_cases=True
)

# Reset generates a new scenario automatically
obs, info = env.reset()

print(f"Scenario: {info['scenario']['scenario_id']}")
print(f"Weather: {info['scenario']['weather_type']}")
print(f"Traffic: {info['scenario']['traffic_density']}")
print(f"Failures: {info['scenario']['num_failures']}")
print(f"Edge cases: {info['scenario']['num_edge_cases']}")
```

### 3. Training with Scenarios

```python
from stable_baselines3 import PPO
from src.environments import EVTOLScenarioEnv

# Create environment
env = EVTOLScenarioEnv(
    scenario_difficulty=0.5,
    curriculum_learning=True  # Gradually increase difficulty
)

# Train
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

---

## 🌦️ Weather Conditions

### Available Weather Types

```python
from src.data_generation import WeatherType

weather_types = [
    WeatherType.CLEAR,    # Clear skies, minimal wind
    WeatherType.WINDY,    # Strong winds, gusts
    WeatherType.RAINY,    # Rain, reduced visibility
    WeatherType.FOGGY,    # Heavy fog, very low visibility
    WeatherType.SNOWY,    # Snow, cold temperatures
    WeatherType.STORMY,   # Storms, high turbulence
    WeatherType.MIXED,    # Mixed conditions
]
```

### Weather Parameters

Each weather condition includes:
- **Wind speed** (m/s): 0-30 m/s
- **Wind direction** (degrees): 0-360°
- **Wind gusts** (boolean): Sudden wind spikes
- **Turbulence level** (0-1): Random perturbations
- **Precipitation** (0-1): Rain/snow intensity
- **Visibility** (meters): 100-10,000m
- **Temperature** (°C): -10 to 35°C
- **Temperature gradient** (°C/100m): Altitude-based
- **Thermal activity** (0-1): Updrafts/downdrafts

### Manual Weather Generation

```python
from src.data_generation import WeatherConditionGenerator, WeatherType

gen = WeatherConditionGenerator(seed=42)

# Generate specific weather
stormy_weather = gen.generate(
    weather_type=WeatherType.STORMY,
    severity=0.8  # High severity
)

print(f"Wind: {stormy_weather.wind_speed:.1f} m/s")
print(f"Turbulence: {stormy_weather.turbulence_level:.2f}")
print(f"Visibility: {stormy_weather.visibility:.0f}m")
```

---

## 🚗 Traffic Patterns

### Traffic Density Levels

```python
from src.data_generation import TrafficDensity

densities = [
    TrafficDensity.LOW,        # 5-15 vehicles
    TrafficDensity.MEDIUM,     # 15-40 vehicles
    TrafficDensity.HIGH,       # 40-80 vehicles
    TrafficDensity.RUSH_HOUR,  # 80-150 vehicles
    TrafficDensity.EMERGENCY,  # 10-30 + emergency vehicles
]
```

### Traffic Components

Each traffic pattern includes:
- **Number of vehicles**: Based on density and time of day
- **Hotspot locations**: High-traffic areas (airports, downtown)
- **Emergency vehicles**: Priority traffic (ambulances, police)
- **Delivery routes**: Predefined paths through the arena
- **Congestion zones**: Areas with reduced speeds

### Manual Traffic Generation

```python
from src.data_generation import TrafficPatternGenerator, TrafficDensity

gen = TrafficPatternGenerator(seed=42)

# Generate rush hour traffic
traffic = gen.generate(
    density=TrafficDensity.RUSH_HOUR,
    time_of_day='morning'
)

print(f"Vehicles: {traffic.num_vehicles}")
print(f"Emergency: {traffic.emergency_vehicles}")
print(f"Hotspots: {len(traffic.hotspot_locations)}")
print(f"Delivery routes: {len(traffic.delivery_routes)}")
```

---

## ⚠️ Failure Modes

### Available Failure Types

```python
from src.data_generation import FailureType

failure_types = [
    FailureType.ROTOR_FAILURE,          # N-1 scenarios
    FailureType.BATTERY_DEGRADATION,    # Gradual drain
    FailureType.GPS_DROPOUT,            # Navigation loss
    FailureType.IMU_DRIFT,              # Attitude error
    FailureType.COMMUNICATION_LOSS,     # Telemetry dropout
    FailureType.PAYLOAD_SHIFT,          # CG change
    FailureType.SENSOR_MALFUNCTION,     # LiDAR, camera, etc.
]
```

### Failure Parameters

Each failure mode has:
- **Severity** (0-1): Impact magnitude
- **Start timestep**: When failure occurs
- **Duration**: How long it lasts (timesteps)
- **Recovery possible**: Can the system compensate?
- **Degradation rate**: For gradual failures
- **Affected components**: List of impacted systems

### Manual Failure Injection

```python
from src.data_generation import FailureModeInjector, FailureType

gen = FailureModeInjector(seed=42)

# Generate rotor failure
failure = gen.generate(
    failure_type=FailureType.ROTOR_FAILURE,
    severity=0.7,
    max_timesteps=1000
)

print(f"Type: {failure.failure_type.value}")
print(f"Starts at: timestep {failure.start_timestep}")
print(f"Duration: {failure.duration} timesteps")
print(f"Recoverable: {failure.recovery_possible}")
print(f"Affected: {', '.join(failure.affected_components)}")
```

---

## 💥 Edge Cases

### Available Edge Case Types

```python
from src.data_generation import EdgeCaseType

edge_cases = [
    EdgeCaseType.BIRD_STRIKE,         # Wildlife collision
    EdgeCaseType.DRONE_SWARM,         # Multiple small drones
    EdgeCaseType.BALLOON,             # Floating balloon
    EdgeCaseType.KITE,                # Kite on string
    EdgeCaseType.WIND_SHEAR,          # Sudden wind change
    EdgeCaseType.NEAR_MISS,           # Close aircraft encounter
    EdgeCaseType.UNEXPECTED_OBSTACLE,  # Generic obstacle
]
```

### Edge Case Parameters

- **Position**: (x, y, z) location
- **Velocity**: Movement vector
- **Size**: Physical dimensions (meters)
- **Timestep**: When it appears
- **Duration**: How long it persists
- **Danger level** (0-1): Severity
- **Avoidable**: Can the agent avoid it?

### Manual Edge Case Generation

```python
from src.data_generation import EdgeCaseLibrary, EdgeCaseType

gen = EdgeCaseLibrary(seed=42)

# Generate wind shear event
edge_case = gen.generate(
    case_type=EdgeCaseType.WIND_SHEAR
)

print(f"Type: {edge_case.case_type.value}")
print(f"Position: {edge_case.position}")
print(f"Velocity: {edge_case.velocity}")
print(f"Danger: {edge_case.danger_level:.2f}")
print(f"Avoidable: {edge_case.avoidable}")
```

---

## 🎮 Complete Scenario Generation

### Automatic Scenario Generation

The `ScenarioGenerator` combines all components:

```python
from src.data_generation import ScenarioGenerator

gen = ScenarioGenerator(seed=42)

# Generate scenario at specific difficulty
scenario = gen.generate(
    difficulty=0.7,  # 0=easy, 1=extreme
    max_timesteps=1000,
    include_failures=True,
    include_edge_cases=True
)

# Scenario includes:
print(scenario.weather.weather_type.value)
print(scenario.traffic.density.value)
print(f"Failures: {len(scenario.failures)}")
print(f"Edge cases: {len(scenario.edge_cases)}")

# Save to JSON
json_str = scenario.to_json()
print(json_str)
```

### Batch Generation

```python
# Generate 100 scenarios with varying difficulty
scenarios = gen.generate_batch(
    n_scenarios=100,
    difficulty_range=(0.0, 1.0)
)

# Curriculum learning (gradually increase difficulty)
curriculum = gen.generate_batch(
    n_scenarios=100,
    difficulty_range=(0.0, 1.0),
    curriculum_learning=True  # Starts easy, ends hard
)

# Save scenarios
gen.save_scenarios(scenarios, 'data/scenarios/batch_001.json')
```

---

## 🗄️ Database Integration

### Scenario Logging

Scenarios are automatically logged to the database when using `EVTOLScenarioEnv`:

```python
from src.database.scenario_logger import ScenarioLogger

logger = ScenarioLogger()

# Log scenario template
scenario_id = logger.log_scenario_template(scenario.to_dict())

# Associate with episode
episode_scenario_id = logger.associate_episode_with_scenario(
    episode_id=123,
    scenario_template_id=scenario_id,
    completion_time=45.3,
    success=True,
    reward=85.2,
    challenge_rating=0.7,
    learning_value=0.8
)

# Update aggregated metrics
logger.update_scenario_metrics(
    weather_type='stormy',
    traffic_density='high',
    difficulty_bin='hard',
    has_failures=True,
    has_edge_cases=True,
    success=True,
    reward=85.2,
    duration=45.3,
    collisions=0,
    altitude_violations=0
)
```

### Query Scenario Performance

```python
# Get performance summary
summary = logger.get_scenario_performance_summary(
    weather_type='stormy',
    difficulty_bin='hard'
)

print(f"Episodes: {summary['total_episodes']}")
print(f"Success rate: {summary['overall_success_rate']:.2f}")
print(f"Avg reward: {summary['avg_reward']:.2f}")

# Get best learning scenarios
best_scenarios = logger.get_best_learning_scenarios(limit=10)

for s in best_scenarios:
    print(f"{s['scenario_id']}: learning_value={s['learning_value']:.2f}")

# Get difficulty progression
progression = logger.get_difficulty_progression()

for difficulty, stats in progression.items():
    print(f"{difficulty}: {stats['success_rate']:.2f} success rate")
```

---

## 📊 Database Schema

### New Tables

#### `scenario_templates`
Stores generated scenario configurations:
- `scenario_id` (unique): Scenario identifier
- `difficulty` (0-1): Difficulty level
- `weather_config` (JSON): Weather parameters
- `traffic_config` (JSON): Traffic parameters
- `failures_config` (JSON): Array of failures
- `edge_cases_config` (JSON): Array of edge cases
- `times_used`: How many episodes used this
- `avg_success_rate`: Performance statistics
- `avg_reward`: Average reward achieved

#### `episode_scenarios`
Links episodes to scenarios:
- `episode_id`: Foreign key to episodes table
- `scenario_id`: Foreign key to scenario_templates
- `completion_time`: Episode duration
- `success`: Whether successful
- `reward`: Total reward
- `challenge_rating`: How challenging (0-1)
- `learning_value`: Estimated learning value (0-1)

#### `scenario_metrics`
Aggregated performance by scenario type:
- `weather_type`: Weather category
- `traffic_density`: Traffic level
- `difficulty_bin`: easy, medium, hard, extreme
- `has_failures`: Boolean
- `has_edge_cases`: Boolean
- `total_episodes`: Count
- `success_rate`: Percentage
- `avg_reward`, `avg_duration`, `avg_collisions`, `avg_altitude_violations`

---

## 🧪 Testing

### Run Complete Test Suite

```bash
# Run all tests
python scripts/test_scenario_generation.py

# Save generated scenarios
python scripts/test_scenario_generation.py --save-scenarios

# Verbose output
python scripts/test_scenario_generation.py --verbose
```

### Test Coverage

The test suite validates:
1. ✅ Weather generation (all 7 types)
2. ✅ Traffic generation (all 5 densities)
3. ✅ Failure injection (all 7 types)
4. ✅ Edge cases (all 7 types)
5. ✅ Complete scenario generation
6. ✅ Serialization (JSON)
7. ✅ Environment integration

---

## 📈 Training Examples

### Example 1: Basic Training with Scenarios

```python
from stable_baselines3 import PPO
from src.environments import EVTOLScenarioEnv

env = EVTOLScenarioEnv(
    vehicle_type="medium",
    scenario_difficulty=0.5
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("models/ppo_scenarios")
```

### Example 2: Curriculum Learning

```python
env = EVTOLScenarioEnv(
    vehicle_type="medium",
    scenario_difficulty=0.5,
    curriculum_learning=True  # Gradually increase difficulty
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)  # Long training
```

### Example 3: Specific Scenario Components

```python
# Train only with weather and traffic (no failures/edge cases)
env = EVTOLScenarioEnv(
    vehicle_type="medium",
    enable_weather_scenarios=True,
    enable_traffic_scenarios=True,
    enable_failure_scenarios=False,  # Disable failures
    enable_edge_cases=False  # Disable edge cases
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Example 4: Custom Scenario

```python
from src.data_generation import ScenarioGenerator
from src.environments import EVTOLScenarioEnv

# Generate specific scenario
gen = ScenarioGenerator(seed=42)
scenario = gen.generate(difficulty=0.9, max_timesteps=1000)

# Create environment and set scenario
env = EVTOLScenarioEnv(vehicle_type="medium", auto_generate_scenarios=False)
env.set_scenario(scenario)

# Train on this specific scenario
obs, info = env.reset()
# ... training loop ...
```

---

## 🔧 Advanced Usage

### Custom Scenario Parameters

```python
from src.data_generation import (
    Scenario,
    WeatherCondition,
    TrafficPattern,
    FailureMode,
    EdgeCase,
    WeatherType,
    TrafficDensity,
    FailureType,
    EdgeCaseType
)

# Create custom weather
weather = WeatherCondition(
    weather_type=WeatherType.STORMY,
    wind_speed=25.0,
    wind_direction=270.0,
    wind_gusts=True,
    turbulence_level=0.8,
    precipitation=0.7,
    visibility=500.0,
    temperature=15.0,
    temperature_gradient=-0.65,
    dynamic_changes=True,
    thermal_activity=0.0
)

# Create custom traffic
traffic = TrafficPattern(
    density=TrafficDensity.RUSH_HOUR,
    time_of_day='morning',
    num_vehicles=120,
    hotspot_locations=[(100, 100, 450), (200, 200, 450)],
    emergency_vehicles=5,
    delivery_routes=[],
    congestion_zones=[]
)

# Create custom failure
failure = FailureMode(
    failure_type=FailureType.ROTOR_FAILURE,
    severity=0.7,
    start_timestep=500,
    duration=500,
    recovery_possible=False,
    degradation_rate=0.0,
    affected_components=['rotor_1', 'thrust']
)

# Create custom edge case
edge_case = EdgeCase(
    case_type=EdgeCaseType.WIND_SHEAR,
    position=(250.0, 250.0, 450.0),
    velocity=(-20.0, 0.0, -5.0),
    size=100.0,
    timestep=300,
    duration=20,
    danger_level=0.9,
    avoidable=False
)

# Combine into scenario
scenario = Scenario(
    scenario_id='custom_001',
    difficulty=0.85,
    weather=weather,
    traffic=traffic,
    failures=[failure],
    edge_cases=[edge_case],
    max_timesteps=1000
)

# Use in environment
env = EVTOLScenarioEnv(auto_generate_scenarios=False)
env.set_scenario(scenario)
```

---

## 📊 Performance Analysis

### Analyze Scenario Performance

```python
from src.database.scenario_logger import ScenarioLogger

logger = ScenarioLogger()

# Compare weather types
for weather in ['clear', 'windy', 'rainy', 'stormy']:
    summary = logger.get_scenario_performance_summary(weather_type=weather)
    print(f"{weather}: {summary['overall_success_rate']:.2%} success rate")

# Compare difficulty levels
progression = logger.get_difficulty_progression()
for difficulty, stats in progression.items():
    print(f"{difficulty}: {stats['success_rate']:.2%} success, {stats['avg_reward']:.1f} avg reward")

# Find best learning scenarios
best = logger.get_best_learning_scenarios(limit=10)
print(f"Top 10 learning scenarios:")
for s in best:
    print(f"  {s['scenario_id']}: learning_value={s['learning_value']:.2f}")
```

---

## 🎓 Best Practices

### 1. Start with Low Difficulty

Begin training with easy scenarios and gradually increase:

```python
env = EVTOLScenarioEnv(
    scenario_difficulty=0.2,  # Start easy
    curriculum_learning=True  # Auto-increase
)
```

### 2. Enable Scenarios Progressively

Don't enable everything at once:

```python
# Phase 1: Weather only
env = EVTOLScenarioEnv(
    enable_weather_scenarios=True,
    enable_traffic_scenarios=False,
    enable_failure_scenarios=False,
    enable_edge_cases=False
)

# Phase 2: Add traffic
env = EVTOLScenarioEnv(
    enable_weather_scenarios=True,
    enable_traffic_scenarios=True,
    enable_failure_scenarios=False,
    enable_edge_cases=False
)

# Phase 3: Add failures and edge cases
env = EVTOLScenarioEnv(
    enable_weather_scenarios=True,
    enable_traffic_scenarios=True,
    enable_failure_scenarios=True,
    enable_edge_cases=True
)
```

### 3. Save Interesting Scenarios

```python
gen = ScenarioGenerator(seed=42)
scenarios = []

for i in range(1000):
    scenario = gen.generate(difficulty=0.8)
    # Save scenarios that meet certain criteria
    if len(scenario.failures) >= 2 and len(scenario.edge_cases) >= 1:
        scenarios.append(scenario)

gen.save_scenarios(scenarios, 'data/scenarios/challenging_scenarios.json')
```

### 4. Monitor Database Statistics

Regularly check scenario performance:

```bash
# Query database
sqlite3 data/database/thermofleet_evtol.db "SELECT weather_type, AVG(success_rate) FROM scenario_metrics GROUP BY weather_type;"
```

---

## 🚀 Next Steps

Now that you have scenario generation:

1. **Train with scenarios**: Use `EVTOLScenarioEnv` instead of `EVTOLEnv`
2. **Analyze performance**: Query scenario metrics to find weaknesses
3. **Create custom scenarios**: Build specific test cases
4. **Implement curriculum learning**: Gradually increase difficulty
5. **Expand scenario library**: Add more weather types, failure modes, etc.

---

## 📚 References

- `TASKS.md`: Priority 1.1 requirements
- `src/data_generation/scenario_generator.py`: Implementation source code
- `scripts/test_scenario_generation.py`: Test examples
- `docs/GETTING_STARTED.md`: General setup guide

---

**Implementation Complete!** ✅

All Priority 1.1 tasks have been implemented and tested. The scenario generation system is production-ready and fully integrated with your database and training pipeline.

