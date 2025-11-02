"""
PPO (Proximal Policy Optimization) trainer for eVTOL agents.
Uses Stable Baselines3 for robust implementation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any
from pathlib import Path
import logging
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from ..environments.evtol_gym_env import EVTOLEnv

logger = logging.getLogger(__name__)


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called at each step."""
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if "r" in info:
                    self.logger.record("rollout/ep_rew_mean", info["r"])
                if "l" in info:
                    self.logger.record("rollout/ep_len_mean", info["l"])

        return True


class WandbCallback(BaseCallback):
    """Callback for logging to Weights & Biases."""

    def __init__(self, verbose=0, log_interval=100):
        super().__init__(verbose)
        self.log_interval = log_interval

        try:
            import wandb
            self.wandb = wandb
            self.wandb_available = True
        except ImportError:
            logger.warning("wandb not installed. Skipping wandb logging.")
            self.wandb_available = False

    def _on_step(self) -> bool:
        """Log metrics to wandb."""
        if not self.wandb_available:
            return True

        if self.n_calls % self.log_interval == 0:
            # Log training metrics
            metrics = {
                "train/total_timesteps": self.num_timesteps,
                "train/fps": int(self.num_timesteps / (time.time() - self.model.start_time)),
            }

            # Log episode info if available
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                if "r" in ep_info:
                    metrics["rollout/ep_reward"] = ep_info["r"]
                if "l" in ep_info:
                    metrics["rollout/ep_length"] = ep_info["l"]

            self.wandb.log(metrics, step=self.num_timesteps)

        return True


class PPOTrainer:
    """
    PPO trainer for eVTOL environments.

    Features:
    - Vectorized parallel environments
    - Automatic checkpointing
    - TensorBoard and WandB logging
    - Evaluation callbacks
    - Custom network architectures
    """

    def __init__(
        self,
        env_id: str = "evtol",
        vehicle_type: str = "medium",
        n_envs: int = 8,
        max_steps: int = 1000,
        total_timesteps: int = 1_000_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        device: str = "auto",
        seed: Optional[int] = None,
        log_dir: str = "./logs",
        save_dir: str = "./models",
        use_wandb: bool = False,
        wandb_project: str = "flying-car-rl",
        wandb_name: Optional[str] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            env_id: Environment identifier
            vehicle_type: Type of vehicle to train
            n_envs: Number of parallel environments
            max_steps: Max steps per episode
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Steps per rollout
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            use_sde: Use state-dependent exploration
            device: Device (cpu/cuda/auto)
            seed: Random seed
            log_dir: Logging directory
            save_dir: Model save directory
            use_wandb: Whether to use wandb
            wandb_project: WandB project name
            wandb_name: WandB run name
        """
        self.env_id = env_id
        self.vehicle_type = vehicle_type
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.total_timesteps = total_timesteps
        self.seed = seed

        # PPO hyperparameters
        self.ppo_kwargs = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "use_sde": use_sde,
            "device": device,
            "verbose": 1,
        }

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
                wandb.init(
                    project=wandb_project,
                    name=wandb_name or f"ppo_{vehicle_type}_{int(time.time())}",
                    config=self.ppo_kwargs,
                )
                logger.info("Initialized WandB logging")
            except ImportError:
                logger.warning("wandb not installed. Disabling wandb.")
                self.use_wandb = False

        # Initialize environment and model
        self.env = None
        self.eval_env = None
        self.model = None

        logger.info(f"Initialized PPO trainer for vehicle type: {vehicle_type}")

    def make_env(self, rank: int, seed: int = 0) -> Callable:
        """Create a function that creates an environment."""
        def _init():
            env = EVTOLEnv(
                vehicle_type=self.vehicle_type,
                max_steps=self.max_steps,
                enable_wind=True,
                enable_sensor_noise=True,
            )
            env = Monitor(env, self.log_dir / f"env_{rank}")
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

    def setup_model(self, policy: str = "MlpPolicy"):
        """
        Setup PPO model.

        Args:
            policy: Policy type ("MlpPolicy", "CnnPolicy", or custom)
        """
        if self.env is None:
            self.setup_env()

        # Create PPO model
        self.model = PPO(
            policy=policy,
            env=self.env,
            tensorboard_log=str(self.log_dir),
            **self.ppo_kwargs,
        )

        logger.info("Created PPO model")

    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []

        # Checkpoint callback (save model periodically)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10000 // self.n_envs, 1),
            save_path=str(self.save_dir),
            name_prefix="ppo_evtol",
            save_replay_buffer=False,
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

        # TensorBoard callback
        tb_callback = TensorboardCallback()
        callbacks.append(tb_callback)

        # WandB callback
        if self.use_wandb:
            wandb_callback = WandbCallback(log_interval=100)
            callbacks.append(wandb_callback)

        return CallbackList(callbacks)

    def train(self):
        """Train the PPO model."""
        if self.model is None:
            self.setup_model()

        callbacks = self.setup_callbacks()

        logger.info(f"Starting training for {self.total_timesteps} timesteps")
        self.model.start_time = time.time()

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
            logger.info(f"Saved checkpoint to {path}")

    def load_model(self, path: str):
        """Load a trained model."""
        if self.env is None:
            self.setup_env()

        self.model = PPO.load(path, env=self.env)
        logger.info(f"Loaded model from {path}")

    def evaluate(self, n_episodes: int = 10, render: bool = False):
        """
        Evaluate the trained model.

        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render episodes

        Returns:
            Mean reward and std
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        if self.eval_env is None:
            self.setup_env()

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
        if self.env is not None:
            self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    trainer = PPOTrainer(
        vehicle_type="medium",
        n_envs=4,
        total_timesteps=100000,
        use_wandb=False,
    )

    trainer.train()
    trainer.evaluate(n_episodes=5)
