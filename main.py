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

    # TODO: Phase 1 - Basic setup complete
    # TODO: Phase 2 - Implement environment and vehicle loading
    # TODO: Phase 3 - Implement RL training loop
    # TODO: Phase 4 - Add visual frontend
    # TODO: Phase 5 - Integrate database logging
    # TODO: Phase 6 - Add testing and optimization

    print("⚠️  Simulation not yet implemented.")
    print("📋 Phase 1 (Project Setup) complete!")
    print()
    print("Next steps:")
    print("  1. Complete Phase 2: Vehicle Modeling and Spawning")
    print("  2. See README.md for full roadmap")
    print()
    print("To verify your setup, run:")
    print("  python scripts/init_db.py")
    print("  python scripts/setup_cesium.py")
    print("  python scripts/setup_arena.py --name NYC_Test --location 'New York City'")

    return 0


if __name__ == '__main__':
    sys.exit(main())
