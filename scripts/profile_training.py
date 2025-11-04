#!/usr/bin/env python3
"""
Profile training performance for FlyingCarRL.

This script runs a training session with comprehensive profiling enabled,
measuring GPU usage, throughput, and identifying bottlenecks.

Usage:
    python scripts/profile_training.py --algo PPO --timesteps 10000
    python scripts/profile_training.py --algo DDPG --vehicle-type large --profile-memory
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.profiling import PerformanceProfiler, GPUProfiler, ThroughputMonitor
from src.training.ppo_trainer import PPOTrainer
from src.training.ddpg_trainer import DDPGTrainer


def profile_training(
    algorithm: str,
    vehicle_type: str,
    total_timesteps: int,
    profile_memory: bool = True
):
    """
    Run training with profiling enabled.

    Args:
        algorithm: Training algorithm (PPO or DDPG)
        vehicle_type: Vehicle type (small, medium, large)
        total_timesteps: Total training timesteps
        profile_memory: Enable GPU memory profiling
    """
    # Initialize profilers
    perf_profiler = PerformanceProfiler()
    gpu_profiler = GPUProfiler()
    throughput_monitor = ThroughputMonitor()

    print(f"\nStarting profiled training:")
    print(f"  Algorithm: {algorithm}")
    print(f"  Vehicle type: {vehicle_type}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  GPU available: {gpu_profiler.gpu_available}")
    print()

    # Vehicle configuration
    vehicle_configs = {
        'small': {
            'mass': 100,
            'max_thrust': 1200,
            'drag_coefficient': 0.3,
            'battery_capacity': 20000,
            'shape': 'sphere'
        },
        'medium': {
            'mass': 500,
            'max_thrust': 6000,
            'drag_coefficient': 0.4,
            'battery_capacity': 80000,
            'shape': 'winged'
        },
        'large': {
            'mass': 1000,
            'max_thrust': 12000,
            'drag_coefficient': 0.5,
            'battery_capacity': 150000,
            'shape': 'boxy'
        }
    }

    arena_config = {
        'name': 'profile_arena',
        'bounds': {
            'x_min': -1000,
            'x_max': 1000,
            'y_min': -1000,
            'y_max': 1000,
            'z_min': 400,
            'z_max': 500
        },
        'obstacles': [],
        'wind_config': {
            'enabled': True,
            'mean_velocity': [0, 0, 0],
            'std_velocity': [2, 2, 1]
        }
    }

    # Select trainer
    if algorithm.upper() == 'PPO':
        trainer = PPOTrainer(
            vehicle_config=vehicle_configs[vehicle_type],
            arena_config=arena_config
        )
    elif algorithm.upper() == 'DDPG':
        trainer = DDPGTrainer(
            vehicle_config=vehicle_configs[vehicle_type],
            arena_config=arena_config
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Profile initialization
    if profile_memory and gpu_profiler.gpu_available:
        print("Initial GPU state:")
        gpu_profiler.print_gpu_memory()

    # Start throughput monitoring
    throughput_monitor.start()

    # Run training with profiling
    print("Starting training...\n")

    if profile_memory and gpu_profiler.gpu_available:
        with gpu_profiler.profile_memory("full_training"):
            with perf_profiler.profile_block("training"):
                trainer.train(
                    total_timesteps=total_timesteps,
                    save_model=False
                )
    else:
        with perf_profiler.profile_block("training"):
            trainer.train(
                total_timesteps=total_timesteps,
                save_model=False
            )

    # Update throughput (approximate)
    throughput_monitor.update(num_steps=total_timesteps, num_episodes=total_timesteps // 1000)

    # Print results
    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)

    perf_profiler.print_summary()
    throughput_monitor.print_throughput()

    if profile_memory and gpu_profiler.gpu_available:
        gpu_profiler.print_gpu_memory()

    print("\nProfiling complete!")


def main():
    parser = argparse.ArgumentParser(description="Profile FlyingCarRL training")

    parser.add_argument(
        '--algo',
        type=str,
        default='PPO',
        choices=['PPO', 'DDPG'],
        help='Training algorithm'
    )

    parser.add_argument(
        '--vehicle-type',
        type=str,
        default='medium',
        choices=['small', 'medium', 'large'],
        help='Vehicle type to train'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=10000,
        help='Total training timesteps'
    )

    parser.add_argument(
        '--profile-memory',
        action='store_true',
        help='Enable GPU memory profiling'
    )

    args = parser.parse_args()

    profile_training(
        algorithm=args.algo,
        vehicle_type=args.vehicle_type,
        total_timesteps=args.timesteps,
        profile_memory=args.profile_memory
    )


if __name__ == '__main__':
    main()
