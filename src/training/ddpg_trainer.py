"""
DDPG (Deep Deterministic Policy Gradient) trainer for eVTOL agents.
Well-suited for continuous control tasks like flight control.
"""

import os
import numpy as np
import torch
from typing import Optional, Dict, Any
from pathlib import Path
import logging
import time

from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from ..environments.evtol_gym_env import EVTOLEnv
from ..database.db_logger import DatabaseLogger
from ..database.callbacks import DatabaseLoggingCallback

logger = logging.getLogger(__name__)


class DDPGTrainer:
    """
    DDPG/TD3/SAC trainer for eVTOL continuous control.

    Supports:
    - DDPG: Deep Deterministic Policy Gradient
    - TD3: Twin Delayed DDPG (more stable)
    - SAC: Soft Actor-Critic (entropy-regularized)

    These algorithms are particularly well-suited for continuous control
    tasks like flight control where smooth actions are important.
    """

    def __init__(
        self,
        algorithm: str = "TD3",
        vehicle_type: str = "medium",
        n_envs: int = 4,
        max_steps: int = 1000,
        total_timesteps: int = 1_000_000,
        learning_rate: float = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,  # For TD3
        target_policy_noise: float = 0.2,  # For TD3
        target_noise_clip: float = 0.5,  # For TD3
        action_noise_std: float = 0.1,
        device: str = "auto",
        seed: Optional[int] = None,
        log_dir: str = "./logs",
        save_dir: str = "./models",
        use_wandb: bool = False,
        wandb_project: str = "thermofleet-evtol-simulator",
        wandb_name: Optional[str] = None,
    ):
        """
        Initialize DDPG/TD3/SAC trainer.

        Args:
            algorithm: Algorithm to use ("DDPG", "TD3", "SAC")
            vehicle_type: Type of vehicle to train
            n_envs: Number of parallel environments
            max_steps: Max steps per episode
            total_timesteps: Total training timesteps
            learning_rate: Learning rate for actor and critic
            buffer_size: Replay buffer size
            learning_starts: Steps before learning starts
            batch_size: Minibatch size
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Update the model every train_freq steps
            gradient_steps: Gradient steps per update
            policy_delay: Policy update delay (TD3 only)
            target_policy_noise: Std of target policy smoothing noise (TD3)
            target_noise_clip: Range to clip target policy noise (TD3)
            action_noise_std: Std of action noise for exploration
            device: Device (cpu/cuda/auto)
            seed: Random seed
            log_dir: Logging directory
            save_dir: Model save directory
            use_wandb: Whether to use wandb
            wandb_project: WandB project name
            wandb_name: WandB run name
        """
        assert algorithm in ["DDPG", "TD3", "SAC"], f"Unknown algorithm: {algorithm}"

        self.algorithm = algorithm
        self.vehicle_type = vehicle_type
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.total_timesteps = total_timesteps
        self.seed = seed
        self.action_noise_std = action_noise_std

        # Algorithm hyperparameters
        self.algo_kwargs = {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "batch_size": batch_size,
            "tau": tau,
            "gamma": gamma,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "device": device,
            "verbose": 1,
        }

        # TD3-specific parameters
        if algorithm == "TD3":
            self.algo_kwargs.update({
                "policy_delay": policy_delay,
                "target_policy_noise": target_policy_noise,
                "target_noise_clip": target_noise_clip,
            })

        # Paths
        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # WandB
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                import os
                
                # Get entity from environment variable if available
                wandb_entity = os.getenv('WANDB_ENTITY', None)
                
                init_config = {
                    "project": wandb_project,
                    "name": wandb_name or f"{algorithm.lower()}_{vehicle_type}_{int(time.time())}",
                    "config": {
                        "algorithm": algorithm,
                        "vehicle_type": vehicle_type,
                        **self.algo_kwargs,
                    },
                }
                
                # Add entity if specified
                if wandb_entity:
                    init_config["entity"] = wandb_entity
                
                wandb.init(**init_config)
                logger.info(f"Initialized WandB logging (project: {wandb_project}, entity: {wandb_entity or 'default'})")
            except ImportError:
                logger.warning("wandb not installed. Disabling wandb.")
                self.use_wandb = False

        # Initialize environment and model
        self.env = None
        self.eval_env = None
        self.model = None
        self.action_noise = None

        # Database logger will be initialized later (after env setup)
        # to avoid pickle issues with SubprocVecEnv
        self.db_logger = None

        logger.info(f"Initialized {algorithm} trainer for vehicle type: {vehicle_type}")

    def make_env(self, rank: int, seed: int = 0):
        """Create environment factory function."""
        def _init():
            env = EVTOLEnv(
                vehicle_type=self.vehicle_type,
                max_steps=self.max_steps,
                enable_wind=True,
                enable_sensor_noise=True,
            )
            env = Monitor(env, str(self.log_dir / f"env_{rank}"))
            if seed is not None:
                env.reset(seed=seed + rank)
            return env
        return _init

    def setup_env(self):
        """Setup training and evaluation environments."""
        if self.seed is not None:
            set_random_seed(self.seed)

        # Create vectorized training environment
        if self.n_envs > 1:
            self.env = SubprocVecEnv([
                self.make_env(i, self.seed) for i in range(self.n_envs)
            ])
        else:
            self.env = DummyVecEnv([self.make_env(0, self.seed)])

        # Create evaluation environment
        self.eval_env = DummyVecEnv([self.make_env(999, self.seed)])

        logger.info(f"Created {self.n_envs} parallel environments")

    def setup_action_noise(self):
        """Setup action noise for exploration."""
        # Get action space dimension
        n_actions = self.env.action_space.shape[-1]

        # Create action noise
        # Ornstein-Uhlenbeck noise is good for continuous control
        self.action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=self.action_noise_std * np.ones(n_actions),
        )

        logger.info(f"Created Ornstein-Uhlenbeck action noise with std={self.action_noise_std}")

    def setup_model(self, policy: str = "MlpPolicy"):
        """
        Setup DDPG/TD3/SAC model.

        Args:
            policy: Policy network type
        """
        if self.env is None:
            self.setup_env()

        self.setup_action_noise()

        # Select algorithm
        if self.algorithm == "DDPG":
            AlgoClass = DDPG
        elif self.algorithm == "TD3":
            AlgoClass = TD3
        elif self.algorithm == "SAC":
            AlgoClass = SAC
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Add action noise for DDPG and TD3
        if self.algorithm in ["DDPG", "TD3"]:
            self.algo_kwargs["action_noise"] = self.action_noise

        # Check if tensorboard is available
        try:
            import tensorboard
            tb_log = str(self.log_dir)
        except ImportError:
            logger.warning("TensorBoard not installed. Logging disabled. Install with: pip install tensorboard")
            tb_log = None

        # Create model
        self.model = AlgoClass(
            policy=policy,
            env=self.env,
            tensorboard_log=tb_log,
            **self.algo_kwargs,
        )

        logger.info(f"Created {self.algorithm} model")

    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []

        # Initialize database logger now (after env setup, to avoid pickle issues)
        if self.db_logger is None:
            try:
                self.db_logger = DatabaseLogger(enable_sensor_logging=False)
                logger.info("Database logging enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize database logger: {e}")
                self.db_logger = None

        # Database logging callback (FIRST - logs all training data)
        if self.db_logger is not None:
            db_callback = DatabaseLoggingCallback(
                db_logger=self.db_logger,
                vehicle_type=self.vehicle_type,
                arena_name="NYC_Manhattan",  # TODO: Make configurable
                algorithm=self.algorithm,
                verbose=1
            )
            callbacks.append(db_callback)
            logger.info("Added database logging callback")

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10000 // self.n_envs, 1),
            save_path=str(self.save_dir),
            name_prefix=f"{self.algorithm.lower()}_evtol",
            save_replay_buffer=True,  # Save replay buffer for off-policy algorithms
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.save_dir / "best"),
            log_path=str(self.log_dir),
            eval_freq=max(5000 // self.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        )
        callbacks.append(eval_callback)

        return CallbackList(callbacks)

    def train(self):
        """Train the model."""
        if self.model is None:
            self.setup_model()

        callbacks = self.setup_callbacks()

        logger.info(f"Starting {self.algorithm} training for {self.total_timesteps} timesteps")

        try:
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                log_interval=10,
                progress_bar=True,
            )

            # Save final model
            final_path = self.save_dir / "final_model"
            self.model.save(final_path)
            logger.info(f"Saved final model to {final_path}")

            # Save replay buffer
            if hasattr(self.model, "replay_buffer"):
                buffer_path = self.save_dir / "final_replay_buffer.pkl"
                self.model.save_replay_buffer(buffer_path)
                logger.info(f"Saved replay buffer to {buffer_path}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint("interrupted")

        finally:
            self.cleanup()

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save a checkpoint."""
        if self.model is not None:
            path = self.save_dir / f"{name}_model"
            self.model.save(path)

            if hasattr(self.model, "replay_buffer"):
                buffer_path = self.save_dir / f"{name}_replay_buffer.pkl"
                self.model.save_replay_buffer(buffer_path)

            logger.info(f"Saved checkpoint to {path}")

    def load_model(self, path: str, load_replay_buffer: bool = False):
        """Load a trained model."""
        if self.env is None:
            self.setup_env()

        # Select algorithm
        if self.algorithm == "DDPG":
            AlgoClass = DDPG
        elif self.algorithm == "TD3":
            AlgoClass = TD3
        elif self.algorithm == "SAC":
            AlgoClass = SAC
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        self.model = AlgoClass.load(path, env=self.env)

        if load_replay_buffer:
            buffer_path = Path(path).parent / (Path(path).stem + "_replay_buffer.pkl")
            if buffer_path.exists():
                self.model.load_replay_buffer(buffer_path)
                logger.info(f"Loaded replay buffer from {buffer_path}")

        logger.info(f"Loaded model from {path}")

    def evaluate(self, n_episodes: int = 10, render: bool = False):
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        if self.eval_env is None:
            # Only create eval env, not training env
            self.eval_env = DummyVecEnv([self.make_env(999, self.seed)])

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1

                if render:
                    self.eval_env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)

        logger.info(f"Evaluation over {n_episodes} episodes:")
        logger.info(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"  Mean length: {mean_length:.1f}")

        return mean_reward, std_reward

    def cleanup(self):
        """Cleanup resources."""
        # Close database logger
        if self.db_logger is not None:
            try:
                self.db_logger.close()
                logger.info("Database logger closed")
            except Exception as e:
                logger.error(f"Error closing database logger: {e}")
            self.db_logger = None

        if self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                logger.error(f"Error closing training env: {e}")
            self.env = None
            
        if self.eval_env is not None:
            try:
                self.eval_env.close()
            except Exception as e:
                logger.error(f"Error closing eval env: {e}")
            self.eval_env = None

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # TD3 is generally more stable than DDPG
    trainer = DDPGTrainer(
        algorithm="TD3",
        vehicle_type="medium",
        n_envs=4,
        total_timesteps=100000,
        use_wandb=False,
    )

    trainer.train()
    trainer.evaluate(n_episodes=5)
