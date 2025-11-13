"""
Stable Baselines3 callbacks for database logging.
Integrates DatabaseLogger with SB3 training loop.
"""

import numpy as np
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
import logging

from .db_logger import DatabaseLogger

logger = logging.getLogger(__name__)


class DatabaseLoggingCallback(BaseCallback):
    """
    Callback for logging training data to database.
    
    Automatically logs:
    - Training run metadata
    - Episode data (rewards, collisions, completion)
    - Per-timestep metrics (position, velocity, battery, altitude)
    """

    def __init__(
        self,
        db_logger: DatabaseLogger,
        vehicle_type: str = "medium",
        arena_name: str = "NYC_Manhattan",
        algorithm: str = "PPO",
        verbose: int = 0
    ):
        """
        Initialize database logging callback.

        Args:
            db_logger: DatabaseLogger instance
            vehicle_type: Vehicle type being trained
            arena_name: Arena name
            algorithm: RL algorithm name
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.db_logger = db_logger
        self.vehicle_type = vehicle_type
        self.arena_name = arena_name
        self.algorithm = algorithm
        
        # Episode tracking
        self.current_episode_num = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_collisions = 0
        self.episode_altitude_violations = 0
        self.episode_active = False
        
        logger.info("DatabaseLoggingCallback initialized")

    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Start training run in database
        try:
            hyperparameters = {
                'learning_rate': float(self.model.learning_rate),
                'gamma': float(self.model.gamma),
                'batch_size': getattr(self.model, 'batch_size', 'N/A'),
                'n_steps': getattr(self.model, 'n_steps', 'N/A'),
                'n_envs': self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else 1
            }
            
            run_name = f"{self.algorithm}_{self.vehicle_type}_{self.current_episode_num}"
            
            self.db_logger.start_training_run(
                name=run_name,
                algorithm=self.algorithm,
                hyperparameters=hyperparameters,
                description=f"Training {self.algorithm} on {self.vehicle_type} vehicle"
            )
            
            logger.info(f"Started training run: {run_name}")
        except Exception as e:
            logger.error(f"Failed to start training run in database: {e}")

    def _on_rollout_start(self) -> None:
        """Called before collecting rollout."""
        pass

    def _on_step(self) -> bool:
        """
        Called after each environment step.
        Logs per-timestep data and handles episode boundaries.
        """
        # Get environment info
        if len(self.locals.get('infos', [])) == 0:
            return True

        # Process each environment
        for env_idx, info in enumerate(self.locals['infos']):
            # Start new episode if needed
            if not self.episode_active:
                try:
                    self.current_episode_num += 1
                    self.db_logger.start_episode(
                        episode_number=self.current_episode_num,
                        vehicle_type=self.vehicle_type,
                        arena_name=self.arena_name,
                        num_agents=1,
                        algorithm=self.algorithm
                    )
                    self.episode_active = True
                    self.episode_reward = 0.0
                    self.episode_length = 0
                    self.episode_collisions = 0
                    self.episode_altitude_violations = 0
                except Exception as e:
                    logger.error(f"Failed to start episode in database: {e}")
                    return True

            # Log timestep data
            try:
                # Get observation from environment
                obs = self.locals.get('new_obs', self.locals.get('obs'))
                if obs is not None and len(obs) > env_idx:
                    current_obs = obs[env_idx]
                    
                    # Extract position, velocity, battery from observation
                    # Assuming obs format: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, ..., battery, time]
                    position = current_obs[:3] if len(current_obs) >= 3 else np.zeros(3)
                    velocity_vec = current_obs[3:6] if len(current_obs) >= 6 else np.zeros(3)
                    velocity_mag = float(np.linalg.norm(velocity_vec))
                    
                    # Battery is usually near the end of observation
                    battery_idx = -2 if len(current_obs) > 27 else len(current_obs) - 1
                    battery = float(current_obs[battery_idx]) if len(current_obs) > battery_idx else 1.0
                    
                    # Altitude (convert to feet, assuming position is in meters)
                    altitude_ft = float(position[2] * 3.28084) if len(position) > 2 else 400.0
                    
                    # Get reward
                    reward = float(self.locals.get('rewards', [0])[env_idx])
                    self.episode_reward += reward
                    
                    # Check for collision and altitude violation from info
                    collision = bool(info.get('collision', False))
                    if collision:
                        self.episode_collisions += 1
                    
                    # Check altitude compliance (400-500 ft)
                    altitude_violation = not (400.0 <= altitude_ft <= 500.0)
                    if altitude_violation:
                        self.episode_altitude_violations += 1
                    
                    # Mock energy consumption (simplified)
                    energy_consumption = 2.0  # kW (mock value)
                    battery_kwh = battery * 50.0  # Assuming max 50 kWh
                    
                    # Log to database
                    self.db_logger.log_timestep(
                        timestep=self.episode_length,
                        position=position,
                        velocity=velocity_mag,
                        altitude_ft=altitude_ft,
                        battery_kwh=battery_kwh,
                        energy_consumption_kw=energy_consumption,
                        step_reward=reward,
                        collision=collision,
                        altitude_violation=altitude_violation
                    )
                    
                    self.episode_length += 1
                    
            except Exception as e:
                logger.error(f"Failed to log timestep: {e}")

            # Check if episode ended
            if info.get('terminal', False) or info.get('TimeLimit.truncated', False):
                try:
                    # Determine success
                    successful = (
                        self.episode_reward > 0 and 
                        self.episode_collisions == 0 and
                        self.episode_altitude_violations < 10
                    )
                    
                    # End episode in database
                    self.db_logger.end_episode(
                        total_reward=self.episode_reward,
                        collision_count=self.episode_collisions,
                        altitude_violations=self.episode_altitude_violations,
                        successful=successful
                    )
                    
                    if self.verbose > 0:
                        logger.info(
                            f"Episode {self.current_episode_num} complete: "
                            f"reward={self.episode_reward:.2f}, "
                            f"collisions={self.episode_collisions}, "
                            f"success={successful}"
                        )
                    
                    self.episode_active = False
                    
                except Exception as e:
                    logger.error(f"Failed to end episode in database: {e}")
                    self.episode_active = False

        return True

    def _on_rollout_end(self) -> None:
        """Called after rollout collection."""
        pass

    def _on_training_end(self) -> None:
        """Called when training ends."""
        # End any active episode
        if self.episode_active:
            try:
                self.db_logger.end_episode(
                    total_reward=self.episode_reward,
                    collision_count=self.episode_collisions,
                    altitude_violations=self.episode_altitude_violations,
                    successful=False  # Interrupted
                )
                self.episode_active = False
            except Exception as e:
                logger.error(f"Failed to end episode on training end: {e}")

        # End training run
        try:
            self.db_logger.end_training_run(
                total_episodes=self.current_episode_num,
                best_reward=float(getattr(self.model, 'best_reward', 0.0)),
                status='completed'
            )
            logger.info(f"Training run completed: {self.current_episode_num} episodes")
        except Exception as e:
            logger.error(f"Failed to end training run: {e}")
