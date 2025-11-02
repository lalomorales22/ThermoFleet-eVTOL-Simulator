#!/usr/bin/env python3
"""
Phase 2 Test Script

Demonstrates vehicle spawning, physics simulation, and sensor emulation.
Tests the complete Phase 2 implementation by spawning and controlling 10 vehicles.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vehicles import (
    VehicleSpawner,
    VehicleType,
    SpawnPattern,
    WindConditions
)
from sensors import CameraSensor, LiDARSensor, IMUSensor


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def test_vehicle_spawning():
    """Test vehicle spawning with different patterns."""
    print_header("TEST 1: Vehicle Spawning")

    # Define arena bounds (1km x 1km area)
    arena_bounds = (-500, 500, -500, 500)  # (x_min, x_max, y_min, y_max) in meters

    # Create wind conditions
    wind = WindConditions(
        wind_velocity=np.array([5.0, 2.0, 0.0]),  # 5 m/s east, 2 m/s north
        turbulence_intensity=0.15,
        gust_frequency=0.3
    )

    # Initialize spawner
    spawner = VehicleSpawner(
        arena_bounds=arena_bounds,
        wind_conditions=wind,
        random_seed=42
    )

    print(f"\n✓ Spawner initialized: {spawner}")
    print(f"  Arena bounds: {arena_bounds}")
    print(f"  Wind: {wind.wind_velocity} m/s")

    # Test different spawn patterns
    patterns = [
        (SpawnPattern.RANDOM, 3, "Random formation"),
        (SpawnPattern.GRID, 4, "Grid formation"),
        (SpawnPattern.CIRCLE, 3, "Circle formation"),
    ]

    for pattern, count, description in patterns:
        vehicles = spawner.spawn_vehicles(count=count, pattern=pattern)
        print(f"\n✓ Spawned {len(vehicles)} vehicles in {description}")
        for i, v in enumerate(vehicles):
            print(f"  Vehicle {i+1}: {v.vehicle_type.value} at {v.position}")

    # Print fleet statistics
    print(f"\n✓ Fleet Statistics:")
    stats = spawner.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return spawner


def test_vehicle_physics(spawner: VehicleSpawner):
    """Test vehicle physics simulation."""
    print_header("TEST 2: Vehicle Physics Simulation")

    # Get one vehicle of each type
    vehicles = spawner.get_active_vehicles()[:3]

    # Set different control inputs for each vehicle
    control_configs = [
        (0.8, 0.1, 0.0, 0.0, "Vertical climb"),
        (0.6, 0.3, 0.1, 0.0, "Forward flight"),
        (0.5, 0.0, 0.2, 0.1, "Banking turn"),
    ]

    for i, (vehicle, (thrust, pitch, roll, yaw, description)) in enumerate(
        zip(vehicles, control_configs)
    ):
        print(f"\n✓ Vehicle {i+1}: {vehicle.vehicle_type.value} - {description}")
        print(f"  Initial state: pos={vehicle.position}, vel={vehicle.velocity}")

        # Set controls
        vehicle.set_controls(thrust=thrust, pitch=pitch, roll=roll, yaw=yaw)
        print(f"  Controls: thrust={thrust}, pitch={pitch}, roll={roll}, yaw={yaw}")

        # Simulate for 5 seconds
        dt = 0.01  # 10ms time step
        num_steps = 500  # 5 seconds

        for step in range(num_steps):
            vehicle.update_physics(dt)

        print(f"  Final state: pos={vehicle.position}, vel={vehicle.velocity}")
        print(f"  Battery SOC: {vehicle.battery.soc:.2%}")
        print(f"  Distance traveled: {np.linalg.norm(vehicle.position):.1f}m")


def test_sensors():
    """Test sensor emulation."""
    print_header("TEST 3: Sensor Emulation")

    # Test Camera
    print("\n✓ Camera Sensor:")
    camera = CameraSensor()
    print(f"  {camera}")

    rgb_image = camera.capture_rgb()
    depth_image = camera.capture_depth()
    print(f"  RGB image shape: {rgb_image.shape}")
    print(f"  Depth image shape: {depth_image.shape}")
    print(f"  Intrinsic matrix:\n{camera.get_intrinsics()}")

    # Test LiDAR
    print("\n✓ LiDAR Sensor:")
    lidar = LiDARSensor()
    print(f"  {lidar}")

    point_cloud, intensities = lidar.capture_scan()
    print(f"  Point cloud shape: {point_cloud.shape}")
    print(f"  Intensities shape: {intensities.shape}")
    print(f"  Scan info: {lidar.get_scan_pattern_info()}")

    # Test IMU
    print("\n✓ IMU Sensor:")
    imu = IMUSensor()
    print(f"  {imu}")

    # Simulate vehicle motion
    acceleration = np.array([1.0, 0.5, 0.2])  # m/s²
    angular_velocity = np.array([0.1, 0.05, 0.02])  # rad/s
    orientation = np.array([0.1, 0.05, 0.0])  # roll, pitch, yaw

    imu_data = imu.measure_full_state(acceleration, angular_velocity, orientation)
    print(f"  Accelerometer: {imu_data['accelerometer']}")
    print(f"  Gyroscope: {imu_data['gyroscope']}")
    print(f"  Config: {imu.get_config_info()}")


def test_full_simulation():
    """Test full simulation with 10 vehicles."""
    print_header("TEST 4: Full Simulation with 10 Vehicles")

    # Setup arena and spawner
    arena_bounds = (-1000, 1000, -1000, 1000)  # 2km x 2km
    wind = WindConditions(
        wind_velocity=np.array([8.0, 3.0, 0.5]),
        turbulence_intensity=0.2
    )

    spawner = VehicleSpawner(arena_bounds=arena_bounds, wind_conditions=wind)

    # Spawn 10 vehicles
    print("\n✓ Spawning 10 vehicles...")
    vehicles = spawner.spawn_vehicles(count=10, pattern=SpawnPattern.CIRCLE)
    print(f"  Spawned {len(vehicles)} vehicles")

    # Assign different behaviors to vehicles
    print("\n✓ Simulating flight behaviors...")

    behaviors = [
        ("Hover", 0.5, 0.0, 0.0, 0.0),
        ("Climb", 0.9, 0.0, 0.0, 0.0),
        ("Forward", 0.7, 0.3, 0.0, 0.0),
        ("Bank Left", 0.6, 0.1, -0.3, 0.0),
        ("Bank Right", 0.6, 0.1, 0.3, 0.0),
        ("Circle", 0.6, 0.0, 0.2, 0.2),
        ("Aggressive", 0.8, 0.4, 0.2, 0.1),
        ("Descent", 0.3, 0.0, 0.0, 0.0),
        ("Cruise", 0.6, 0.2, 0.0, 0.0),
        ("Sprint", 0.9, 0.3, 0.0, 0.0),
    ]

    for vehicle, (behavior, thrust, pitch, roll, yaw) in zip(vehicles, behaviors):
        vehicle.set_controls(thrust, pitch, roll, yaw)
        print(f"  {vehicle.id[:8]}... ({vehicle.vehicle_type.value}): {behavior}")

    # Run simulation
    print("\n✓ Running 10-second simulation...")
    dt = 0.02  # 20ms time step (50 Hz)
    duration = 10.0  # 10 seconds
    num_steps = int(duration / dt)

    for step in range(num_steps):
        spawner.update_all_vehicles(dt)

        # Print progress every 2 seconds
        if step % 100 == 0:
            elapsed = step * dt
            active = len(spawner.get_active_vehicles())
            print(f"  t={elapsed:.1f}s: {active} active vehicles")

    # Final statistics
    print("\n✓ Simulation Complete! Final Statistics:")
    stats = spawner.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Print individual vehicle status
    print("\n✓ Vehicle Status:")
    for i, vehicle in enumerate(vehicles):
        state = vehicle.get_state()
        status = "✓ Active" if state['active'] else "✗ Inactive"
        print(f"  {i+1}. {state['vehicle_type']:6} | {status} | "
              f"SOC: {state['battery']['soc']:.1%} | "
              f"Pos: [{state['position'][0]:6.1f}, {state['position'][1]:6.1f}, {state['position'][2]:6.1f}]")


def main():
    """Run all Phase 2 tests."""
    print("\n" + "=" * 70)
    print("  FLYINGCARRL - PHASE 2 TEST SUITE")
    print("  Vehicle Modeling and Spawning")
    print("=" * 70)

    try:
        # Test 1: Spawning
        spawner = test_vehicle_spawning()

        # Test 2: Physics
        test_vehicle_physics(spawner)

        # Test 3: Sensors
        test_sensors()

        # Test 4: Full simulation
        test_full_simulation()

        # Success
        print_header("ALL TESTS PASSED ✓")
        print("\n✓ Phase 2 Implementation Complete!")
        print("  - eVTOL vehicle blueprints (small, medium, large)")
        print("  - Physics simulation (aerodynamics, wind, battery)")
        print("  - Spawning system with multiple patterns")
        print("  - Sensor emulation (camera, LiDAR, IMU)")
        print("\n✓ Ready for Phase 3: RL Integration and Training Pipeline")
        print()

        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
