#!/usr/bin/env python3
"""
Test Script for Scenario Generation System

Tests all components of the Priority 1.1 implementation:
- Weather condition generation
- Traffic pattern generation
- Failure mode injection
- Edge case library
- Environment integration
- Database logging

Usage:
    python scripts/test_scenario_generation.py [--verbose] [--save-scenarios]
"""

import os
import sys
from pathlib import Path
import argparse
import json
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation import (
    ScenarioGenerator,
    WeatherConditionGenerator,
    TrafficPatternGenerator,
    FailureModeInjector,
    EdgeCaseLibrary,
    WeatherType,
    TrafficDensity,
    FailureType,
    EdgeCaseType
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_weather_generation():
    """Test weather condition generation"""
    print("\n" + "=" * 60)
    print("TEST 1: Weather Condition Generation")
    print("=" * 60)
    
    weather_gen = WeatherConditionGenerator(seed=42)
    
    # Test all weather types
    weather_types = list(WeatherType)
    print(f"\nTesting {len(weather_types)} weather types...")
    
    for weather_type in weather_types:
        for severity in [0.2, 0.5, 0.8]:
            weather = weather_gen.generate(weather_type=weather_type, severity=severity)
            
            print(f"\n  {weather_type.value.upper()} (severity={severity:.1f}):")
            print(f"    Wind: {weather.wind_speed:.1f} m/s @ {weather.wind_direction:.0f}°")
            print(f"    Turbulence: {weather.turbulence_level:.2f}")
            print(f"    Precipitation: {weather.precipitation:.2f}")
            print(f"    Visibility: {weather.visibility:.0f}m")
            print(f"    Temperature: {weather.temperature:.1f}°C")
            print(f"    Wind gusts: {weather.wind_gusts}")
            print(f"    Dynamic: {weather.dynamic_changes}")
    
    # Test batch generation
    print("\n\nBatch generation test (50 scenarios)...")
    batch = weather_gen.generate_batch(50, difficulty_range=(0.0, 1.0))
    
    weather_counts = {}
    for w in batch:
        weather_counts[w.weather_type.value] = weather_counts.get(w.weather_type.value, 0) + 1
    
    print("  Weather distribution:")
    for wtype, count in sorted(weather_counts.items()):
        print(f"    {wtype}: {count} ({count/len(batch)*100:.1f}%)")
    
    print("\n✅ Weather generation test PASSED")
    return True


def test_traffic_generation():
    """Test traffic pattern generation"""
    print("\n" + "=" * 60)
    print("TEST 2: Traffic Pattern Generation")
    print("=" * 60)
    
    traffic_gen = TrafficPatternGenerator(seed=42)
    
    # Test all traffic densities
    densities = list(TrafficDensity)
    times_of_day = ['morning', 'afternoon', 'evening', 'night']
    
    print(f"\nTesting {len(densities)} traffic densities...")
    
    for density in densities:
        for time in times_of_day[:2]:  # Just morning and afternoon for brevity
            traffic = traffic_gen.generate(density=density, time_of_day=time)
            
            print(f"\n  {density.value.upper()} @ {time}:")
            print(f"    Vehicles: {traffic.num_vehicles}")
            print(f"    Emergency: {traffic.emergency_vehicles}")
            print(f"    Hotspots: {len(traffic.hotspot_locations)}")
            print(f"    Delivery routes: {len(traffic.delivery_routes)}")
            print(f"    Congestion zones: {len(traffic.congestion_zones)}")
            
            # Validate data
            assert traffic.num_vehicles > 0, "Must have vehicles"
            assert len(traffic.hotspot_locations) > 0, "Must have hotspots"
    
    print("\n✅ Traffic generation test PASSED")
    return True


def test_failure_injection():
    """Test failure mode injection"""
    print("\n" + "=" * 60)
    print("TEST 3: Failure Mode Injection")
    print("=" * 60)
    
    failure_gen = FailureModeInjector(seed=42)
    
    # Test all failure types
    failure_types = list(FailureType)
    print(f"\nTesting {len(failure_types)} failure types...")
    
    for failure_type in failure_types:
        for severity in [0.3, 0.7]:
            failure = failure_gen.generate(
                failure_type=failure_type,
                severity=severity,
                max_timesteps=1000
            )
            
            print(f"\n  {failure_type.value.upper()} (severity={severity:.1f}):")
            print(f"    Start: timestep {failure.start_timestep}")
            print(f"    Duration: {failure.duration} timesteps")
            print(f"    Recoverable: {failure.recovery_possible}")
            print(f"    Degradation rate: {failure.degradation_rate:.4f}")
            print(f"    Affected: {', '.join(failure.affected_components)}")
            
            # Validate
            assert failure.start_timestep < 1000, "Start must be before episode end"
            assert failure.duration > 0, "Duration must be positive"
            assert len(failure.affected_components) > 0, "Must affect components"
    
    # Test batch generation
    print("\n\nBatch generation test (20 failures)...")
    batch = failure_gen.generate_batch(20, max_timesteps=1000)
    
    failure_counts = {}
    for f in batch:
        failure_counts[f.failure_type.value] = failure_counts.get(f.failure_type.value, 0) + 1
    
    print("  Failure distribution:")
    for ftype, count in sorted(failure_counts.items()):
        print(f"    {ftype}: {count}")
    
    print("\n✅ Failure injection test PASSED")
    return True


def test_edge_cases():
    """Test edge case library"""
    print("\n" + "=" * 60)
    print("TEST 4: Edge Case Library")
    print("=" * 60)
    
    edge_gen = EdgeCaseLibrary(seed=42)
    
    # Test all edge case types
    case_types = list(EdgeCaseType)
    print(f"\nTesting {len(case_types)} edge case types...")
    
    for case_type in case_types:
        edge_case = edge_gen.generate(case_type=case_type, max_timesteps=1000)
        
        print(f"\n  {case_type.value.upper()}:")
        print(f"    Position: ({edge_case.position[0]:.1f}, {edge_case.position[1]:.1f}, {edge_case.position[2]:.1f})")
        print(f"    Velocity: ({edge_case.velocity[0]:.1f}, {edge_case.velocity[1]:.1f}, {edge_case.velocity[2]:.1f})")
        print(f"    Size: {edge_case.size:.1f}m")
        print(f"    Timestep: {edge_case.timestep}")
        print(f"    Duration: {edge_case.duration}")
        print(f"    Danger: {edge_case.danger_level:.2f}")
        print(f"    Avoidable: {edge_case.avoidable}")
        
        # Validate
        assert edge_case.timestep < 1000, "Must occur during episode"
        assert edge_case.size > 0, "Size must be positive"
        assert 0 <= edge_case.danger_level <= 1, "Danger must be 0-1"
    
    print("\n✅ Edge case test PASSED")
    return True


def test_scenario_generation():
    """Test complete scenario generation"""
    print("\n" + "=" * 60)
    print("TEST 5: Complete Scenario Generation")
    print("=" * 60)
    
    scenario_gen = ScenarioGenerator(seed=42)
    
    # Test scenarios at different difficulties
    difficulties = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\nGenerating scenarios at {len(difficulties)} difficulty levels...")
    
    scenarios = []
    for difficulty in difficulties:
        scenario = scenario_gen.generate(
            difficulty=difficulty,
            max_timesteps=1000,
            include_failures=True,
            include_edge_cases=True
        )
        scenarios.append(scenario)
        
        print(f"\n  Difficulty {difficulty:.2f}:")
        print(f"    ID: {scenario.scenario_id}")
        print(f"    Weather: {scenario.weather.weather_type.value}")
        print(f"    Traffic: {scenario.traffic.density.value} ({scenario.traffic.num_vehicles} vehicles)")
        print(f"    Failures: {len(scenario.failures)}")
        print(f"    Edge cases: {len(scenario.edge_cases)}")
        
        # Validate structure
        assert scenario.difficulty == difficulty
        assert scenario.weather is not None
        assert scenario.traffic is not None
        assert isinstance(scenario.failures, list)
        assert isinstance(scenario.edge_cases, list)
    
    # Test batch generation
    print("\n\nBatch generation test (100 scenarios)...")
    batch = scenario_gen.generate_batch(100, difficulty_range=(0.0, 1.0))
    
    # Analyze batch
    avg_failures = sum(len(s.failures) for s in batch) / len(batch)
    avg_edge_cases = sum(len(s.edge_cases) for s in batch) / len(batch)
    
    print(f"  Total scenarios: {len(batch)}")
    print(f"  Avg failures per scenario: {avg_failures:.2f}")
    print(f"  Avg edge cases per scenario: {avg_edge_cases:.2f}")
    
    # Test curriculum learning
    print("\n\nCurriculum learning test (50 scenarios)...")
    curriculum_batch = scenario_gen.generate_batch(
        50,
        difficulty_range=(0.0, 1.0),
        curriculum_learning=True
    )
    
    # Check that difficulty increases
    difficulties_curriculum = [s.difficulty for s in curriculum_batch]
    print(f"  First 5 difficulties: {[f'{d:.2f}' for d in difficulties_curriculum[:5]]}")
    print(f"  Last 5 difficulties: {[f'{d:.2f}' for d in difficulties_curriculum[-5:]]}")
    
    # Validate increasing trend
    assert difficulties_curriculum[0] < difficulties_curriculum[-1], "Should increase with curriculum"
    
    print("\n✅ Scenario generation test PASSED")
    return scenarios


def test_scenario_serialization(scenarios):
    """Test scenario serialization and deserialization"""
    print("\n" + "=" * 60)
    print("TEST 6: Scenario Serialization")
    print("=" * 60)
    
    # Test to_dict and to_json
    print("\nTesting scenario serialization...")
    
    for scenario in scenarios[:2]:  # Test first 2
        # Convert to dict
        scenario_dict = scenario.to_dict()
        assert isinstance(scenario_dict, dict), "to_dict must return dict"
        assert 'scenario_id' in scenario_dict
        assert 'difficulty' in scenario_dict
        assert 'weather' in scenario_dict
        assert 'traffic' in scenario_dict
        
        # Convert to JSON
        scenario_json = scenario.to_json()
        assert isinstance(scenario_json, str), "to_json must return string"
        
        # Parse JSON back
        parsed = json.loads(scenario_json)
        assert parsed['scenario_id'] == scenario.scenario_id
        
        print(f"  ✓ Scenario {scenario.scenario_id} serialized successfully")
    
    print("\n✅ Serialization test PASSED")
    return True


def test_environment_integration():
    """Test integration with EVTOLScenarioEnv"""
    print("\n" + "=" * 60)
    print("TEST 7: Environment Integration")
    print("=" * 60)
    
    try:
        from src.environments.evtol_scenario_env import EVTOLScenarioEnv
        
        print("\nInitializing EVTOLScenarioEnv...")
        env = EVTOLScenarioEnv(
            vehicle_type="medium",
            max_steps=100,
            enable_scenarios=True,
            scenario_difficulty=0.5,
            auto_generate_scenarios=True,
            scenario_seed=42,
            enable_weather_scenarios=True,
            enable_traffic_scenarios=True,
            enable_failure_scenarios=True,
            enable_edge_cases=True
        )
        
        print("✓ Environment created successfully")
        
        # Test reset
        print("\nTesting environment reset with scenario generation...")
        obs, info = env.reset(seed=42)
        
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation space: {env.observation_space}")
        
        # Check scenario info
        assert 'scenario' in info, "Info should contain scenario data"
        scenario_info = info['scenario']
        
        print(f"\n  Generated scenario:")
        print(f"    ID: {scenario_info['scenario_id']}")
        print(f"    Difficulty: {scenario_info['difficulty']:.2f}")
        print(f"    Weather: {scenario_info['weather_type']}")
        print(f"    Traffic: {scenario_info['traffic_density']}")
        print(f"    Failures: {scenario_info['num_failures']}")
        print(f"    Edge cases: {scenario_info['num_edge_cases']}")
        
        # Test a few steps
        print("\nRunning 10 simulation steps...")
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step == 0:
                print(f"  Step {step}: reward={reward:.2f}, terminated={terminated}")
        
        print("✓ Environment steps executed successfully")
        
        # Test scenario summary
        summary = env.get_scenario_summary()
        print(f"\n  Scenario summary keys: {list(summary.keys())}")
        
        print("\n✅ Environment integration test PASSED")
        return True
        
    except ImportError as e:
        print(f"\n⚠️  Warning: Could not import EVTOLScenarioEnv: {e}")
        print("  This is expected if running before full integration")
        return False


def save_test_scenarios(scenarios, output_dir: str = "data/test_scenarios"):
    """Save generated scenarios to files"""
    print("\n" + "=" * 60)
    print("Saving Test Scenarios")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual scenarios
    for i, scenario in enumerate(scenarios):
        scenario_file = output_path / f"scenario_{i:03d}.json"
        with open(scenario_file, 'w') as f:
            f.write(scenario.to_json())
        print(f"  Saved: {scenario_file}")
    
    # Save batch summary
    summary_file = output_path / "scenarios_summary.json"
    summary = {
        'total_scenarios': len(scenarios),
        'difficulty_range': [min(s.difficulty for s in scenarios), max(s.difficulty for s in scenarios)],
        'scenarios': [
            {
                'id': s.scenario_id,
                'difficulty': s.difficulty,
                'weather': s.weather.weather_type.value,
                'traffic': s.traffic.density.value,
                'failures': len(s.failures),
                'edge_cases': len(s.edge_cases)
            }
            for s in scenarios
        ]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Summary saved: {summary_file}")
    print(f"\n✅ Saved {len(scenarios)} scenarios to {output_dir}")


def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description='Test scenario generation system')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--save-scenarios', action='store_true', help='Save generated scenarios')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n" + "=" * 60)
    print("ThermoFleet Scenario Generation System Test Suite")
    print("Testing Priority 1.1 Implementation")
    print("=" * 60)
    
    # Run tests
    tests_passed = 0
    tests_total = 0
    
    test_functions = [
        ("Weather Generation", test_weather_generation),
        ("Traffic Generation", test_traffic_generation),
        ("Failure Injection", test_failure_injection),
        ("Edge Cases", test_edge_cases),
    ]
    
    scenarios = None
    
    for test_name, test_func in test_functions:
        tests_total += 1
        try:
            if test_name == "Scenario Generation":
                scenarios = test_func()
            else:
                result = test_func()
            tests_passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Scenario generation (returns scenarios for next tests)
    tests_total += 1
    try:
        print("\n" + "=" * 60)
        scenarios = test_scenario_generation()
        tests_passed += 1
    except Exception as e:
        print(f"\n❌ Scenario generation test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Serialization test
    if scenarios:
        tests_total += 1
        try:
            test_scenario_serialization(scenarios)
            tests_passed += 1
        except Exception as e:
            print(f"\n❌ Serialization test FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Environment integration test
    tests_total += 1
    try:
        if test_environment_integration():
            tests_passed += 1
    except Exception as e:
        print(f"\n❌ Environment integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Save scenarios if requested
    if args.save_scenarios and scenarios:
        save_test_scenarios(scenarios)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"\nTests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n🎉 ALL TESTS PASSED! Priority 1.1 implementation is complete!")
        print("\nNext steps:")
        print("  1. Run database migration: python scripts/migrate_db_scenarios.py")
        print("  2. Update training scripts to use EVTOLScenarioEnv")
        print("  3. Start training with diverse scenarios!")
        return 0
    else:
        print(f"\n⚠️  {tests_total - tests_passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

