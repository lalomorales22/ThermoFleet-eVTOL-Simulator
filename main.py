#!/usr/bin/env python3
"""
FlyingCarRL - Main entry point

This script launches the simulation in either visual or headless mode.

Usage:
    python main.py --mode=visual --agents=100
    python main.py --mode=headless --agents=1000 --episodes=10000
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv not installed")
    print("Please install: pip install -r requirements.txt")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Import Phase 2 components
try:
    from vehicles import VehicleSpawner, SpawnPattern, WindConditions
    from sensors import CameraSensor, LiDARSensor, IMUSensor
    PHASE2_AVAILABLE = True
except ImportError as e:
    PHASE2_AVAILABLE = False
    PHASE2_ERROR = str(e)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FlyingCarRL Simulation')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['visual', 'headless'],
        default=os.getenv('DEFAULT_SIM_MODE', 'headless'),
        help='Simulation mode (default: headless)'
    )

    parser.add_argument(
        '--agents',
        type=int,
        default=int(os.getenv('DEFAULT_NUM_AGENTS', '100')),
        help='Number of agents to spawn (default: 100)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=int(os.getenv('DEFAULT_EPISODES', '1000')),
        help='Number of training episodes (default: 1000)'
    )

    parser.add_argument(
        '--arena',
        type=str,
        default='NYC_Manhattan',
        help='Arena to load (default: NYC_Manhattan)'
    )

    parser.add_argument(
        '--db',
        type=str,
        choices=['sqlite', 'mysql'],
        default=os.getenv('DB_TYPE', 'sqlite'),
        help='Database type for logging (default: sqlite)'
    )

    return parser.parse_args()


def run_headless_simulation(args):
    """Run headless simulation with Phase 2 components"""
    print("\n" + "=" * 60)
    print("Starting Headless Simulation")
    print("=" * 60)

    # Define arena bounds (based on arena name, for now use default)
    arena_bounds = (-1000, 1000, -1000, 1000)  # 2km x 2km area

    # Create wind conditions
    wind = WindConditions(
        wind_velocity=np.array([5.0, 2.0, 0.0]),  # Light wind
        turbulence_intensity=0.15
    )

    # Initialize spawner
    spawner = VehicleSpawner(
        arena_bounds=arena_bounds,
        wind_conditions=wind
    )

    print(f"\n✓ Arena: {args.arena}")
    print(f"  Bounds: {arena_bounds}")
    print(f"  Wind: {wind.wind_velocity} m/s")

    # Spawn vehicles
    print(f"\n✓ Spawning {args.agents} vehicles...")
    vehicles = spawner.spawn_vehicles(
        count=args.agents,
        pattern=SpawnPattern.RANDOM
    )
    print(f"  Spawned {len(vehicles)} vehicles")

    stats = spawner.get_statistics()
    print(f"  Vehicle types: {stats['vehicle_type_counts']}")

    # Simple flight behavior (hover with slight forward motion)
    print("\n✓ Setting vehicle controls...")
    for vehicle in vehicles:
        # Basic hover/cruise behavior
        vehicle.set_controls(
            thrust=0.6,
            pitch=0.1 if np.random.random() > 0.5 else 0.0,
            roll=0.0,
            yaw=0.0
        )

    # Run simulation
    print(f"\n✓ Running simulation for {args.episodes} episodes...")
    dt = 0.02  # 20ms time step
    steps_per_episode = 500  # 10 seconds per episode

    for episode in range(min(args.episodes, 10)):  # Limit to 10 episodes for demo
        print(f"\n  Episode {episode + 1}/{args.episodes}")

        for step in range(steps_per_episode):
            spawner.update_all_vehicles(dt)

        # Episode summary
        active = len(spawner.get_active_vehicles())
        print(f"    Active vehicles: {active}/{len(vehicles)}")

        # Reset vehicles for next episode
        spawner.reset_all_vehicles()

    # Final statistics
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    final_stats = spawner.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")


def main():
    """Main entry point"""
    args = parse_args()

    print("=" * 60)
    print("FlyingCarRL - Autonomous eVTOL Training Simulator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Agents: {args.agents}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Arena: {args.arena}")
    print(f"  Database: {args.db}")
    print()

    # Check Phase 2 availability
    if not PHASE2_AVAILABLE:
        print("❌ Phase 2 components not available!")
        print(f"   Error: {PHASE2_ERROR}")
        print()
        print("Next steps:")
        print("  1. Complete Phase 2: Vehicle Modeling and Spawning")
        print("  2. See README.md for full roadmap")
        return 1

    print("✓ Phase 1: Project Setup - Complete")
    print("✓ Phase 2: Vehicle Modeling and Spawning - Complete")
    print()

    # Run simulation based on mode
    if args.mode == 'headless':
        run_headless_simulation(args)
    else:
        print("⚠️  Visual mode not yet implemented (Phase 4)")
        print()
        print("For now, you can:")
        print("  - Run headless mode: python main.py --mode=headless --agents=10")
        print("  - Test Phase 2: python scripts/test_phase2.py")
        print()
        print("Next steps:")
        print("  Phase 3: RL Integration and Training Pipeline")
        print("  Phase 4: Frontend and UI Development")

    return 0


if __name__ == '__main__':
    sys.exit(main())
