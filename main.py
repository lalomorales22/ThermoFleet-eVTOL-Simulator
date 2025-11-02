#!/usr/bin/env python3
"""
FlyingCarRL - Main entry point

This script launches the simulation in either visual, headless, or training mode.

Usage:
    python main.py --mode=visual --agents=100
    python main.py --mode=headless --agents=1000 --episodes=10000
    python main.py --mode=training --algo=PPO --vehicle-type=medium
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
        choices=['visual', 'headless', 'training'],
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

    # Training mode arguments
    parser.add_argument(
        '--algo',
        type=str,
        default='PPO',
        choices=['PPO', 'DDPG', 'TD3', 'SAC'],
        help='RL algorithm for training mode (default: PPO)'
    )

    parser.add_argument(
        '--vehicle-type',
        type=str,
        default='medium',
        choices=['small', 'medium', 'large'],
        help='Vehicle type for training (default: medium)'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=1_000_000,
        help='Total training timesteps (default: 1M)'
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

    if args.mode == 'training':
        print(f"  Algorithm: {args.algo}")
        print(f"  Vehicle Type: {args.vehicle_type}")
        print(f"  Timesteps: {args.timesteps:,}")
        print()

        # Import training module
        from src.training.ppo_trainer import PPOTrainer
        from src.training.ddpg_trainer import DDPGTrainer

        print("🚀 Starting RL Training...")
        print()

        # Create trainer based on algorithm
        if args.algo == 'PPO':
            trainer = PPOTrainer(
                vehicle_type=args.vehicle_type,
                n_envs=min(args.agents, 16),  # Limit parallel envs
                total_timesteps=args.timesteps,
                use_wandb=False,
            )
        elif args.algo in ['DDPG', 'TD3', 'SAC']:
            trainer = DDPGTrainer(
                algorithm=args.algo,
                vehicle_type=args.vehicle_type,
                n_envs=min(args.agents, 8),
                total_timesteps=args.timesteps,
                use_wandb=False,
            )
        else:
            print(f"❌ Unknown algorithm: {args.algo}")
            return 1

        # Train the agent
        try:
            trainer.train()
            print("\n✅ Training completed successfully!")

            # Evaluate
            print("\nRunning evaluation...")
            mean_reward, std_reward = trainer.evaluate(n_episodes=10)
            print(f"Final Performance: {mean_reward:.2f} ± {std_reward:.2f}")

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            trainer.cleanup()
            return 0

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    else:
        # Visual or headless mode
        print(f"  Agents: {args.agents}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Arena: {args.arena}")
        print(f"  Database: {args.db}")
        print()

        # Phase status
        print("✅ Phase 1 (Project Setup) - COMPLETE")
        print("✅ Phase 2 (Vehicle Modeling) - COMPLETE")
        print("✅ Phase 3 (RL Integration) - COMPLETE")
        print()
        print("⚠️  Visual/Headless simulation modes will be implemented in Phase 4")
        print()
        print("To train an agent, use:")
        print("  python main.py --mode=training --algo=PPO --vehicle-type=medium")
        print()
        print("Or use the dedicated training script:")
        print("  python train.py --algo=PPO --n-envs=8 --total-timesteps=1000000")
        print()
        print("To verify your setup, run:")
        print("  python scripts/init_db.py")
        print("  python scripts/setup_cesium.py")
        print("  python scripts/setup_arena.py --name NYC_Test --location 'New York City'")

    return 0


if __name__ == '__main__':
    sys.exit(main())
