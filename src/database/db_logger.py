"""
Database logging module for FlyingCarRL.
Logs episodes, metrics, and sensor data during training.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# Import models from init_db
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.init_db import (
    Vehicle, Arena, Episode, SensorLog, Metric, TrainingRun,
    get_database_url
)

logger = logging.getLogger(__name__)
load_dotenv()


class DatabaseLogger:
    """
    Database logger for simulation episodes.

    Logs:
    - Episode metadata (start/end times, rewards, collisions)
    - Per-timestep metrics (position, velocity, battery)
    - Sensor data (optional, compressed)
    - Training run metadata
    """

    def __init__(self, db_type: Optional[str] = None, enable_sensor_logging: bool = False):
        """
        Initialize database logger.

        Args:
            db_type: Database type ('sqlite' or 'mysql'). Defaults to env variable.
            enable_sensor_logging: Whether to log detailed sensor data (can be large).
        """
        self.db_type = db_type or os.getenv('DB_TYPE', 'sqlite')
        self.enable_sensor_logging = enable_sensor_logging

        # Create engine and session
        db_url = get_database_url(self.db_type)
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Current episode tracking
        self.current_episode: Optional[Episode] = None
        self.current_training_run: Optional[TrainingRun] = None
        self.episode_start_time: Optional[datetime] = None
        self.episode_metrics_buffer: List[Dict] = []
        self.episode_sensor_buffer: List[Dict] = []

        logger.info(f"DatabaseLogger initialized with {self.db_type}")

    def start_training_run(
        self,
        name: str,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        description: Optional[str] = None
    ) -> int:
        """
        Start a new training run.

        Args:
            name: Training run name
            algorithm: RL algorithm (PPO, DDPG, etc.)
            hyperparameters: Training hyperparameters
            description: Optional description

        Returns:
            training_run_id
        """
        session = self.SessionLocal()
        try:
            training_run = TrainingRun(
                name=name,
                description=description,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                status='running',
                start_time=datetime.utcnow()
            )
            session.add(training_run)
            session.commit()

            self.current_training_run = training_run
            logger.info(f"Started training run: {name} (ID: {training_run.id})")

            return training_run.id

        except Exception as e:
            logger.error(f"Error starting training run: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def end_training_run(
        self,
        total_episodes: int,
        best_reward: float,
        convergence_episode: Optional[int] = None,
        status: str = 'completed'
    ):
        """
        End the current training run.

        Args:
            total_episodes: Total episodes completed
            best_reward: Best reward achieved
            convergence_episode: Episode where convergence was reached
            status: Final status ('completed', 'failed', 'interrupted')
        """
        if self.current_training_run is None:
            logger.warning("No active training run to end")
            return

        session = self.SessionLocal()
        try:
            run = session.query(TrainingRun).filter_by(
                id=self.current_training_run.id
            ).first()

            if run:
                run.end_time = datetime.utcnow()
                run.status = status
                run.total_episodes = total_episodes
                run.best_reward = best_reward
                run.convergence_episode = convergence_episode
                session.commit()

                logger.info(f"Ended training run {run.name}: {status}")

            self.current_training_run = None

        except Exception as e:
            logger.error(f"Error ending training run: {e}")
            session.rollback()
        finally:
            session.close()

    def start_episode(
        self,
        episode_number: int,
        vehicle_type: str,
        arena_name: str,
        num_agents: int = 1,
        algorithm: str = "PPO",
        model_version: str = "v1"
    ) -> int:
        """
        Start logging a new episode.

        Args:
            episode_number: Episode number
            vehicle_type: Vehicle type (small, medium, large)
            arena_name: Arena name
            num_agents: Number of agents
            algorithm: RL algorithm
            model_version: Model version

        Returns:
            episode_id
        """
        session = self.SessionLocal()
        try:
            # Get or create vehicle
            vehicle = session.query(Vehicle).filter_by(
                type=vehicle_type
            ).first()

            if not vehicle:
                logger.warning(f"Vehicle type {vehicle_type} not found, using default")
                vehicle = session.query(Vehicle).first()

            # Get or create arena
            arena = session.query(Arena).filter_by(name=arena_name).first()
            if not arena:
                logger.warning(f"Arena {arena_name} not found, using default")
                arena = session.query(Arena).first()

            # Create episode
            episode = Episode(
                episode_number=episode_number,
                vehicle_id=vehicle.id,
                arena_id=arena.id,
                start_time=datetime.utcnow(),
                num_agents=num_agents,
                algorithm=algorithm,
                model_version=model_version
            )
            session.add(episode)
            session.commit()

            self.current_episode = episode
            self.episode_start_time = datetime.utcnow()
            self.episode_metrics_buffer = []
            self.episode_sensor_buffer = []

            logger.debug(f"Started episode {episode_number} (ID: {episode.id})")

            return episode.id

        except Exception as e:
            logger.error(f"Error starting episode: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def log_timestep(
        self,
        timestep: int,
        position: np.ndarray,
        velocity: float,
        altitude_ft: float,
        battery_kwh: float,
        energy_consumption_kw: float,
        step_reward: float,
        collision: bool = False,
        altitude_violation: bool = False
    ):
        """
        Log metrics for a single timestep.

        Args:
            timestep: Current timestep
            position: 3D position [x, y, z]
            velocity: Velocity magnitude
            altitude_ft: Altitude in feet
            battery_kwh: Battery level in kWh
            energy_consumption_kw: Energy consumption in kW
            step_reward: Reward for this step
            collision: Whether collision occurred
            altitude_violation: Whether altitude was violated
        """
        if self.current_episode is None:
            logger.warning("Cannot log timestep: no active episode")
            return

        metric_data = {
            'timestep': timestep,
            'position_x': float(position[0]),
            'position_y': float(position[1]),
            'position_z': float(position[2]),
            'velocity': float(velocity),
            'altitude_ft': float(altitude_ft),
            'battery_remaining_kwh': float(battery_kwh),
            'energy_consumption_kw': float(energy_consumption_kw),
            'step_reward': float(step_reward),
            'collision': collision,
            'altitude_violation': altitude_violation
        }

        self.episode_metrics_buffer.append(metric_data)

    def log_sensor_data(
        self,
        timestep: float,
        agent_id: int,
        sensor_type: str,
        data: Dict[str, Any]
    ):
        """
        Log sensor data (optional, for detailed analysis).

        Args:
            timestep: Simulation timestamp
            agent_id: Agent ID
            sensor_type: Sensor type (camera, lidar, imu, gps)
            data: Sensor data dictionary
        """
        if not self.enable_sensor_logging or self.current_episode is None:
            return

        sensor_data = {
            'timestep': timestep,
            'agent_id': agent_id,
            'sensor_type': sensor_type,
            'data_blob': data
        }

        self.episode_sensor_buffer.append(sensor_data)

    def end_episode(
        self,
        total_reward: float,
        collision_count: int = 0,
        altitude_violations: int = 0,
        successful: bool = False,
        trajectory_data: Optional[Dict] = None
    ):
        """
        End the current episode and flush data to database.

        Args:
            total_reward: Total episode reward
            collision_count: Number of collisions
            altitude_violations: Number of altitude violations
            successful: Whether episode was successful
            trajectory_data: Optional trajectory data (positions over time)
        """
        if self.current_episode is None:
            logger.warning("Cannot end episode: no active episode")
            return

        session = self.SessionLocal()
        try:
            episode = session.query(Episode).filter_by(
                id=self.current_episode.id
            ).first()

            if episode:
                # Update episode metadata
                episode.end_time = datetime.utcnow()
                episode.duration_seconds = (
                    episode.end_time - episode.start_time
                ).total_seconds()
                episode.total_reward = total_reward
                episode.avg_reward = total_reward / max(len(self.episode_metrics_buffer), 1)
                episode.collision_count = collision_count
                episode.altitude_violations = altitude_violations
                episode.successful_completion = successful

                if trajectory_data:
                    episode.trajectory_data = trajectory_data

                # Flush metrics
                if self.episode_metrics_buffer:
                    for metric_data in self.episode_metrics_buffer:
                        metric = Metric(
                            episode_id=episode.id,
                            **metric_data
                        )
                        session.add(metric)

                # Flush sensor logs
                if self.enable_sensor_logging and self.episode_sensor_buffer:
                    for sensor_data in self.episode_sensor_buffer:
                        sensor_log = SensorLog(
                            episode_id=episode.id,
                            **sensor_data
                        )
                        session.add(sensor_log)

                session.commit()

                logger.debug(
                    f"Ended episode {episode.episode_number}: "
                    f"reward={total_reward:.2f}, "
                    f"duration={episode.duration_seconds:.1f}s, "
                    f"metrics={len(self.episode_metrics_buffer)}"
                )

            # Clear buffers
            self.current_episode = None
            self.episode_metrics_buffer = []
            self.episode_sensor_buffer = []

        except Exception as e:
            logger.error(f"Error ending episode: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get a new database session for custom queries."""
        return self.SessionLocal()

    def close(self):
        """Close database connections."""
        if self.current_episode:
            logger.warning("Closing logger with active episode - data may be lost")

        if self.engine:
            self.engine.dispose()
            logger.info("DatabaseLogger closed")
