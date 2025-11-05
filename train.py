#!/usr/bin/env python3
"""
Main training script for ThermoFleet-eVTOL-Simulator.
Supports multiple algorithms and configurations via CLI.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.ppo_trainer import PPOTrainer
from src.training.ddpg_trainer import DDPGTrainer
from src.environments.evtol_gym_env import EVTOLEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train eVTOL autonomous flight agents using RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Algorithm selection
    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        choices=["PPO", "DDPG", "TD3", "SAC"],
        help="RL algorithm to use",
    )

    # Environment settings
    parser.add_argument(
        "--vehicle-type",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Type of eVTOL vehicle to train",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )

    # Training settings
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )

    # Logging and saving
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for TensorBoard logs",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save models",
    )

    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="thermofleet-evtol-simulator",
        help="WandB project name",
    )

    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="WandB run name",
    )

    # Evaluation
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for evaluation",
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI args)",
    )

    # Thermodynamic computing arguments
    parser.add_argument(
        "--thermodynamic",
        action="store_true",
        help="Enable thermodynamic decision making",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Inverse temperature for thermodynamic sampling (higher = more deterministic)",
    )

    parser.add_argument(
        "--path-planner",
        type=str,
        default=None,
        choices=["thermodynamic", "standard"],
        help="Path planning method (thermodynamic uses energy-based planning)",
    )

    parser.add_argument(
        "--n-waypoints",
        type=int,
        default=10,
        help="Number of waypoints for thermodynamic path planning",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def train(args):
    """Run training."""
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {args.algo}")
    logger.info(f"Vehicle Type: {args.vehicle_type}")
    logger.info(f"Total Timesteps: {args.total_timesteps:,}")
    logger.info(f"Parallel Environments: {args.n_envs}")
    logger.info(f"Device: {args.device}")

    # Log thermodynamic settings
    if args.thermodynamic or args.path_planner == "thermodynamic":
        logger.info("\n🔥 THERMODYNAMIC COMPUTING ENABLED")
        if args.thermodynamic:
            logger.info(f"Thermodynamic Decision Making: ON")
            logger.info(f"Beta (inverse temperature): {args.beta}")
        if args.path_planner == "thermodynamic":
            logger.info(f"Energy-Based Path Planner: ON")
            logger.info(f"Number of Waypoints: {args.n_waypoints}")

    logger.info("=" * 60)

    # Initialize thermodynamic modules if enabled
    thermodynamic_decision_maker = None
    thermodynamic_path_planner = None

    if args.thermodynamic:
        from src.thermodynamic import ThermodynamicDecisionMaker
        thermodynamic_decision_maker = ThermodynamicDecisionMaker(beta=args.beta)
        logger.info("✓ Initialized ThermodynamicDecisionMaker")

    if args.path_planner == "thermodynamic":
        from src.thermodynamic import EnergyBasedPathPlanner
        # Arena bounds will be set by the environment
        arena_bounds = (-1000, 1000, -1000, 1000, 120, 150)  # Default bounds
        thermodynamic_path_planner = EnergyBasedPathPlanner(
            arena_bounds=arena_bounds,
            beta=args.beta,
        )
        logger.info(f"✓ Initialized EnergyBasedPathPlanner with {args.n_waypoints} waypoints")

    # Store thermodynamic config for passing to environment
    env_config = {
        'thermodynamic_decision_maker': thermodynamic_decision_maker,
        'thermodynamic_path_planner': thermodynamic_path_planner,
        'n_waypoints': args.n_waypoints if args.path_planner == "thermodynamic" else None,
    }

    # Create trainer based on algorithm
    if args.algo == "PPO":
        trainer = PPOTrainer(
            vehicle_type=args.vehicle_type,
            n_envs=args.n_envs,
            max_steps=args.max_steps,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            device=args.device,
            seed=args.seed,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
        )

    elif args.algo in ["DDPG", "TD3", "SAC"]:
        trainer = DDPGTrainer(
            algorithm=args.algo,
            vehicle_type=args.vehicle_type,
            n_envs=args.n_envs,
            max_steps=args.max_steps,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            device=args.device,
            seed=args.seed,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
        )

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Note: The thermodynamic modules (env_config) should be integrated into the
    # environment creation within the trainer. For now, this demonstrates the
    # argument parsing and initialization. Full integration would require
    # modifying the EVTOLEnv to accept and use these thermodynamic components.

    # Train
    try:
        trainer.train()
        logger.info("Training completed successfully!")

        # Run evaluation
        logger.info("\nRunning post-training evaluation...")
        mean_reward, std_reward = trainer.evaluate(n_episodes=args.n_eval_episodes)
        logger.info(f"Final performance: {mean_reward:.2f} ± {std_reward:.2f}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    finally:
        trainer.cleanup()


def evaluate(args):
    """Run evaluation only."""
    if args.model_path is None:
        raise ValueError("Must provide --model-path for evaluation")

    logger.info("=" * 60)
    logger.info("RUNNING EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Episodes: {args.n_eval_episodes}")
    logger.info("=" * 60)

    # Create environment
    from src.environments.evtol_gym_env import EVTOLEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.utils.evaluation import EVTOLEvaluator

    env = EVTOLEnv(
        vehicle_type=args.vehicle_type,
        max_steps=args.max_steps,
    )

    # Load model
    if args.algo == "PPO":
        from stable_baselines3 import PPO
        model = PPO.load(args.model_path)
    elif args.algo == "DDPG":
        from stable_baselines3 import DDPG
        model = DDPG.load(args.model_path)
    elif args.algo == "TD3":
        from stable_baselines3 import TD3
        model = TD3.load(args.model_path)
    elif args.algo == "SAC":
        from stable_baselines3 import SAC
        model = SAC.load(args.model_path)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    logger.info("Model loaded successfully")

    # Create evaluator
    evaluator = EVTOLEvaluator(
        env=env,
        agent=model,
        n_eval_episodes=args.n_eval_episodes,
        save_trajectories=True,
        output_dir="./eval_results",
    )

    # Run evaluation
    results = evaluator.evaluate(deterministic=True, render=False)
    evaluator.print_summary()

    logger.info("Evaluation complete!")


def main():
    """Main entry point."""
    args = parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Create output directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Run training or evaluation
    try:
        if args.eval_only:
            evaluate(args)
        else:
            train(args)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
