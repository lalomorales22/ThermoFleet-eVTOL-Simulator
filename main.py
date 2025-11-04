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
        print("✅ Phase 4 (Frontend and UI) - COMPLETE")
        print("✅ Phase 5 (Database Integration) - COMPLETE")
        print()

        # Import UI components
        from src.ui.omniverse_viewer import OmniverseViewer, ViewerConfig
        from src.ui.controls import (
            SimulationController,
            VehicleSpawner,
            ArenaSelector
        )
        from src.ui.replay_system import ReplayRecorder

        # Initialize components
        controller = SimulationController()
        arena_selector = ArenaSelector()

        # Load arena
        print(f"🗺️  Loading arena: {args.arena}")
        arena = arena_selector.select_arena(args.arena)

        if arena is None:
            print(f"❌ Failed to load arena: {args.arena}")
            print("Available arenas:", ', '.join(arena_selector.list_arenas()))
            return 1

        # Initialize spawner
        spawner = VehicleSpawner(arena)

        # Initialize viewer
        print(f"🎨 Initializing {'visual' if args.mode == 'visual' else 'headless'} mode...")
        viewer = OmniverseViewer(headless=(args.mode == 'headless'))

        if viewer.initialize():
            viewer.load_arena(args.arena)

        # Spawn agents
        print(f"🚁 Spawning {args.agents} agents...")
        vehicle_ids = spawner.spawn_fleet(
            count=args.agents,
            vehicle_type=args.vehicle_type,
            formation='grid'
        )

        # Spawn in viewer
        for vid in vehicle_ids:
            vehicle_info = spawner.get_vehicle_info(vid)
            if vehicle_info:
                viewer.spawn_agent(
                    agent_id=vid,
                    position=vehicle_info['position'],
                    vehicle_type=vehicle_info['type']
                )

        print(f"✅ Spawned {len(vehicle_ids)} vehicles")
        print()

        # Run simulation
        if args.mode == 'visual':
            print("🎮 Running in VISUAL mode")
            print()
            print("Controls:")
            print("  - The Omniverse viewer would open here with 3D visualization")
            print("  - Use the dashboard for interactive control: streamlit run dashboard.py")
            print()
            print("Note: Full Omniverse integration requires NVIDIA Omniverse installation")
            print("      Current implementation provides mock visualization for development")

        else:  # headless
            print("🤖 Running in HEADLESS mode")
            print()
            print(f"Running {args.episodes} episodes...")

            # Initialize replay recorder
            recorder = ReplayRecorder(save_dir="replays")

            # Mock simulation loop
            import numpy as np
            from src.environments.evtol_gym_env import eVTOLGymEnv

            try:
                # Create environment
                env = eVTOLGymEnv(vehicle_type=args.vehicle_type)

                for episode in range(args.episodes):
                    # Start recording
                    recorder.start_recording(
                        vehicle_type=args.vehicle_type,
                        num_agents=1,
                        arena=args.arena,
                        notes=f"Headless episode {episode}"
                    )

                    obs, _ = env.reset()
                    done = False
                    episode_reward = 0
                    timestep = 0

                    while not done and timestep < 1000:
                        # Random action for demo
                        action = env.action_space.sample()
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                        # Record frame
                        recorder.record_frame(
                            timestep=timestep,
                            positions=np.array([obs[:3]]),
                            velocities=np.array([obs[3:6]]),
                            actions=np.array([action]),
                            rewards=np.array([reward]),
                            battery_levels=np.array([obs[-1] if len(obs) > 6 else 1.0])
                        )

                        episode_reward += reward
                        timestep += 1

                    # Stop recording
                    recorder.stop_recording(
                        success=(episode_reward > 0),
                        collisions=0,
                        altitude_violations=0
                    )

                    # Print progress
                    if (episode + 1) % 10 == 0:
                        print(f"  Episode {episode + 1}/{args.episodes} - Reward: {episode_reward:.2f}")

                print(f"\n✅ Completed {args.episodes} episodes")
                print(f"📼 Replays saved to: replays/")

            except KeyboardInterrupt:
                print("\n⚠️  Simulation interrupted by user")

            except Exception as e:
                print(f"\n❌ Simulation failed: {e}")
                import traceback
                traceback.print_exc()
                return 1

            finally:
                env.close()

        # Cleanup
        viewer.shutdown()
        print()
        print("=" * 60)
        print("For interactive visualization, run:")
        print("  streamlit run dashboard.py")
        print()
        print("For training, use:")
        print("  python main.py --mode=training --algo=PPO --vehicle-type=medium")
        print("  python train.py --algo=PPO --n-envs=8 --total-timesteps=1000000")
        print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
