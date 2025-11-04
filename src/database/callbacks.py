"""
Stable Baselines3 callbacks for database logging.
Integrates DatabaseLogger with training loops.
"""

import numpy as np
from typing import Optional, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback
import logging

from .db_logger import DatabaseLogger

logger = logging.getLogger(__name__)


class DatabaseLoggingCallback(BaseCallback):
    """
    Callback for logging training data to database.

    Logs:
    - Training run metadata
    - Episode rewards and outcomes
    - Per-timestep metrics (optional)
    """

    def __init__(
        self,
        db_logger: Optional[DatabaseLogger] = None,
        training_run_name: Optional[str] = None,
        algorithm: str = "PPO",
        vehicle_type: str = "medium",
        arena_name: str = "NYC_Manhattan",
        hyperparameters: Optional[Dict[str, Any]] = None,
        log_timesteps: bool = False,
        verbose: int = 0
    ):
        """
        Initialize callback.

        Args:
            db_logger: DatabaseLogger instance (creates new if None)
            training_run_name: Name for training run
            algorithm: RL algorithm name
            vehicle_type: Vehicle type
            arena_name: Arena name
            hyperparameters: Training hyperparameters
            log_timesteps: Whether to log individual timesteps (can be large)
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.db_logger = db_logger or DatabaseLogger()
        self.training_run_name = training_run_name
        self.algorithm = algorithm
        self.vehicle_type = vehicle_type
        self.arena_name = arena_name
        self.hyperparameters = hyperparameters or {}
        self.log_timesteps = log_timesteps

        self.training_run_id = None
        self.episode_count = 0
        self.best_reward = -float('inf')
        self.convergence_episode = None

        # Episode tracking
        self.current_episode_id = None
        self.episode_start_info = {}
        self.episode_rewards = []
        self.episode_collisions = 0
        self.episode_violations = 0

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        if self.training_run_name is None:
            import time
            self.training_run_name = f"{self.algorithm}_{self.vehicle_type}_{int(time.time())}"

        # Start training run in database
        self.training_run_id = self.db_logger.start_training_run(
            name=self.training_run_name,
            algorithm=self.algorithm,
            hyperparameters=self.hyperparameters,
            description=f"Training {self.vehicle_type} eVTOL with {self.algorithm}"
        )

        logger.info(f"Started database logging for training run: {self.training_run_name}")

    def _on_rollout_start(self) -> None:
        """Called at the start of a rollout (episode start)."""
        # Start new episode in database
        self.episode_count += 1

        self.current_episode_id = self.db_logger.start_episode(
            episode_number=self.episode_count,
            vehicle_type=self.vehicle_type,
            arena_name=self.arena_name,
            num_agents=1,
            algorithm=self.algorithm,
            model_version="v1"
        )

        # Reset episode tracking
        self.episode_rewards = []
        self.episode_collisions = 0
        self.episode_violations = 0

    def _on_step(self) -> bool:
        """
        Called at each step.

        Returns:
            True to continue training
        """
        # Log timestep metrics if enabled
        if self.log_timesteps and self.current_episode_id:
            # Get environment info
            infos = self.locals.get('infos', [])

            if len(infos) > 0:
                info = infos[0]

                # Extract metrics from info dict
                if 'position' in info:
                    position = np.array(info['position'])
                    velocity = info.get('velocity', 0.0)
                    altitude = info.get('altitude_ft', 450.0)
                    battery = info.get('battery_kwh', 1.0)
                    energy = info.get('energy_consumption_kw', 0.0)
                    reward = self.locals.get('rewards', [0.0])[0]
                    collision = info.get('collision', False)
                    violation = info.get('altitude_violation', False)

                    # Log to database
                    self.db_logger.log_timestep(
                        timestep=self.num_timesteps,
                        position=position,
                        velocity=velocity,
                        altitude_ft=altitude,
                        battery_kwh=battery,
                        energy_consumption_kw=energy,
                        step_reward=reward,
                        collision=collision,
                        altitude_violation=violation
                    )

                    # Track collisions and violations
                    if collision:
                        self.episode_collisions += 1
                    if violation:
                        self.episode_violations += 1

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout (episode end)."""
        if self.current_episode_id is None:
            return

        # Get episode info from buffer
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            episode_reward = ep_info.get('r', 0.0)
            episode_length = ep_info.get('l', 0)

            # Check if successful (reward above threshold)
            successful = episode_reward > 0

            # Track best reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                if self.convergence_episode is None:
                    # Consider converged if reward exceeds threshold
                    if episode_reward > 100:  # Adjust threshold as needed
                        self.convergence_episode = self.episode_count

            # End episode in database
            self.db_logger.end_episode(
                total_reward=episode_reward,
                collision_count=self.episode_collisions,
                altitude_violations=self.episode_violations,
                successful=successful
            )

            if self.verbose > 0 and self.episode_count % 10 == 0:
                logger.info(
                    f"Episode {self.episode_count}: "
                    f"reward={episode_reward:.2f}, "
                    f"length={episode_length}, "
                    f"collisions={self.episode_collisions}, "
                    f"violations={self.episode_violations}"
                )

        self.current_episode_id = None

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # End training run
        if self.training_run_id:
            self.db_logger.end_training_run(
                total_episodes=self.episode_count,
                best_reward=self.best_reward,
                convergence_episode=self.convergence_episode,
                status='completed'
            )

            logger.info(
                f"Training run completed: {self.episode_count} episodes, "
                f"best reward: {self.best_reward:.2f}"
            )

        # Close logger
        self.db_logger.close()


class EpisodeLoggingCallback(BaseCallback):
    """
    Lightweight callback that only logs episode-level metrics.
    Useful for high-throughput training where timestep logging is too expensive.
    """

    def __init__(
        self,
        db_logger: Optional[DatabaseLogger] = None,
        vehicle_type: str = "medium",
        arena_name: str = "NYC_Manhattan",
        algorithm: str = "PPO",
        log_interval: int = 1,
        verbose: int = 0
    ):
        """
        Initialize callback.

        Args:
            db_logger: DatabaseLogger instance
            vehicle_type: Vehicle type
            arena_name: Arena name
            algorithm: RL algorithm
            log_interval: Log every N episodes
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.db_logger = db_logger or DatabaseLogger()
        self.vehicle_type = vehicle_type
        self.arena_name = arena_name
        self.algorithm = algorithm
        self.log_interval = log_interval

        self.episode_count = 0
        self.episodes_since_log = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        # Check if episode ended
        dones = self.locals.get('dones', [])

        if len(dones) > 0 and dones[0]:
            self.episode_count += 1
            self.episodes_since_log += 1

            # Log episode if interval reached
            if self.episodes_since_log >= self.log_interval:
                self._log_episode()
                self.episodes_since_log = 0

        return True

    def _log_episode(self):
        """Log episode to database."""
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            episode_reward = ep_info.get('r', 0.0)

            # Start and immediately end episode
            episode_id = self.db_logger.start_episode(
                episode_number=self.episode_count,
                vehicle_type=self.vehicle_type,
                arena_name=self.arena_name,
                algorithm=self.algorithm
            )

            self.db_logger.end_episode(
                total_reward=episode_reward,
                successful=(episode_reward > 0)
            )

            if self.verbose > 0 and self.episode_count % 100 == 0:
                logger.info(
                    f"Logged episode {self.episode_count}: reward={episode_reward:.2f}"
                )
