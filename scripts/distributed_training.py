#!/usr/bin/env python3
"""
Distributed training setup for FlyingCarRL using Ray.

This script enables training across multiple GPUs or machines using Ray clusters.

Usage:
    # Single machine, multiple GPUs
    python scripts/distributed_training.py --num-workers 4 --algo PPO

    # Connect to Ray cluster
    python scripts/distributed_training.py --ray-address "ray://cluster-head:10001"

    # With specific resources
    python scripts/distributed_training.py --num-gpus 2 --num-cpus 8
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ddpg import DDPGConfig


def setup_ray_cluster(
    num_cpus: int = None,
    num_gpus: int = None,
    ray_address: str = None
):
    """
    Initialize Ray cluster for distributed training.

    Args:
        num_cpus: Number of CPUs to use (None for auto)
        num_gpus: Number of GPUs to use (None for auto)
        ray_address: Address of existing Ray cluster (None for local)
    """
    if ray.is_initialized():
        ray.shutdown()

    if ray_address:
        print(f"Connecting to Ray cluster at {ray_address}")
        ray.init(address=ray_address)
    else:
        print("Initializing local Ray cluster")
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            include_dashboard=True
        )

    print(f"Ray cluster resources: {ray.cluster_resources()}")


def create_distributed_config(algorithm: str, num_workers: int, vehicle_type: str):
    """
    Create RLlib configuration for distributed training.

    Args:
        algorithm: Algorithm name (PPO or DDPG)
        num_workers: Number of parallel workers
        vehicle_type: Vehicle type to train

    Returns:
        RLlib algorithm config
    """
    # Environment config
    env_config = {
        'vehicle_type': vehicle_type,
        'arena': {
            'name': 'distributed_arena',
            'bounds': {
                'x_min': -1000,
                'x_max': 1000,
                'y_min': -1000,
                'y_max': 1000,
                'z_min': 400,
                'z_max': 500
            }
        }
    }

    if algorithm.upper() == 'PPO':
        config = (
            PPOConfig()
            .environment(env='EVTOLEnv', env_config=env_config)
            .framework('torch')
            .rollouts(
                num_rollout_workers=num_workers,
                num_envs_per_worker=1,
                rollout_fragment_length=200
            )
            .training(
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=30,
                lr=3e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                vf_loss_coeff=0.5,
                entropy_coeff=0.01
            )
            .resources(
                num_gpus=1 if ray.cluster_resources().get('GPU', 0) > 0 else 0,
                num_cpus_per_worker=1,
                num_gpus_per_worker=0
            )
        )
    elif algorithm.upper() == 'DDPG':
        config = (
            DDPGConfig()
            .environment(env='EVTOLEnv', env_config=env_config)
            .framework('torch')
            .rollouts(
                num_rollout_workers=num_workers,
                num_envs_per_worker=1
            )
            .training(
                train_batch_size=256,
                lr=1e-3,
                gamma=0.99,
                tau=0.005
            )
            .resources(
                num_gpus=1 if ray.cluster_resources().get('GPU', 0) > 0 else 0,
                num_cpus_per_worker=1
            )
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return config


def run_distributed_training(
    algorithm: str,
    vehicle_type: str,
    num_workers: int,
    total_timesteps: int,
    checkpoint_freq: int = 10
):
    """
    Run distributed training.

    Args:
        algorithm: Algorithm name
        vehicle_type: Vehicle type
        num_workers: Number of parallel workers
        total_timesteps: Total training timesteps
        checkpoint_freq: Checkpoint frequency (iterations)
    """
    print(f"\n{'='*80}")
    print("DISTRIBUTED TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Algorithm:       {algorithm}")
    print(f"Vehicle type:    {vehicle_type}")
    print(f"Workers:         {num_workers}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"{'='*80}\n")

    # Create config
    config = create_distributed_config(algorithm, num_workers, vehicle_type)

    # Create the algorithm
    algo = config.build()

    # Training loop
    try:
        iteration = 0
        timesteps_trained = 0

        while timesteps_trained < total_timesteps:
            result = algo.train()

            iteration += 1
            timesteps_trained = result['timesteps_total']

            # Print progress
            print(f"\nIteration {iteration}:")
            print(f"  Timesteps: {timesteps_trained}/{total_timesteps}")
            print(f"  Episode reward mean: {result.get('episode_reward_mean', 0):.2f}")
            print(f"  Episode length mean: {result.get('episode_len_mean', 0):.2f}")

            # Checkpoint
            if iteration % checkpoint_freq == 0:
                checkpoint_dir = algo.save()
                print(f"  Checkpoint saved: {checkpoint_dir}")

    finally:
        algo.stop()

    print("\nDistributed training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed training for FlyingCarRL"
    )

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
        help='Vehicle type'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel rollout workers'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000000,
        help='Total training timesteps'
    )

    parser.add_argument(
        '--num-cpus',
        type=int,
        default=None,
        help='Number of CPUs (None for auto)'
    )

    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs (None for auto)'
    )

    parser.add_argument(
        '--ray-address',
        type=str,
        default=None,
        help='Ray cluster address (None for local)'
    )

    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='Checkpoint frequency (iterations)'
    )

    args = parser.parse_args()

    # Setup Ray
    setup_ray_cluster(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        ray_address=args.ray_address
    )

    # Run training
    run_distributed_training(
        algorithm=args.algo,
        vehicle_type=args.vehicle_type,
        num_workers=args.num_workers,
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq
    )

    # Cleanup
    ray.shutdown()


if __name__ == '__main__':
    main()
