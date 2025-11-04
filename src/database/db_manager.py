"""
Database manager for FlyingCarRL.
Provides utilities for querying and managing database data.
"""

import os
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime

from sqlalchemy import create_engine, func, and_, or_
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.init_db import (
    Vehicle, Arena, Episode, SensorLog, Metric, TrainingRun,
    get_database_url
)

logger = logging.getLogger(__name__)
load_dotenv()


class DatabaseManager:
    """
    Database manager for querying and analyzing simulation data.

    Provides high-level APIs for:
    - Querying episodes by criteria
    - Aggregating metrics
    - Exporting data for analysis
    - Database maintenance
    """

    def __init__(self, db_type: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            db_type: Database type ('sqlite' or 'mysql'). Defaults to env variable.
        """
        self.db_type = db_type or os.getenv('DB_TYPE', 'sqlite')

        # Create engine and session
        db_url = get_database_url(self.db_type)
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info(f"DatabaseManager initialized with {self.db_type}")

    def query_episodes(
        self,
        vehicle_type: Optional[str] = None,
        arena_name: Optional[str] = None,
        min_reward: Optional[float] = None,
        successful_only: bool = False,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query episodes with filters.

        Args:
            vehicle_type: Filter by vehicle type
            arena_name: Filter by arena name
            min_reward: Minimum reward threshold
            successful_only: Only include successful episodes
            limit: Maximum number of results

        Returns:
            DataFrame with episode data
        """
        session = self.SessionLocal()
        try:
            query = session.query(
                Episode.id,
                Episode.episode_number,
                Vehicle.type.label('vehicle_type'),
                Arena.name.label('arena_name'),
                Episode.start_time,
                Episode.duration_seconds,
                Episode.total_reward,
                Episode.avg_reward,
                Episode.collision_count,
                Episode.altitude_violations,
                Episode.successful_completion,
                Episode.algorithm,
                Episode.num_agents
            ).join(Vehicle).join(Arena)

            # Apply filters
            if vehicle_type:
                query = query.filter(Vehicle.type == vehicle_type)

            if arena_name:
                query = query.filter(Arena.name == arena_name)

            if min_reward is not None:
                query = query.filter(Episode.total_reward >= min_reward)

            if successful_only:
                query = query.filter(Episode.successful_completion == True)

            # Order by episode number
            query = query.order_by(Episode.episode_number.desc())

            if limit:
                query = query.limit(limit)

            # Convert to DataFrame
            results = query.all()
            df = pd.DataFrame(results, columns=[
                'id', 'episode_number', 'vehicle_type', 'arena_name',
                'start_time', 'duration_seconds', 'total_reward', 'avg_reward',
                'collision_count', 'altitude_violations', 'successful_completion',
                'algorithm', 'num_agents'
            ])

            logger.info(f"Queried {len(df)} episodes")
            return df

        finally:
            session.close()

    def get_episode_metrics(self, episode_id: int) -> pd.DataFrame:
        """
        Get all metrics for a specific episode.

        Args:
            episode_id: Episode ID

        Returns:
            DataFrame with timestep metrics
        """
        session = self.SessionLocal()
        try:
            metrics = session.query(Metric).filter_by(
                episode_id=episode_id
            ).order_by(Metric.timestep).all()

            data = []
            for m in metrics:
                data.append({
                    'timestep': m.timestep,
                    'position_x': m.position_x,
                    'position_y': m.position_y,
                    'position_z': m.position_z,
                    'velocity': m.velocity,
                    'altitude_ft': m.altitude_ft,
                    'battery_remaining_kwh': m.battery_remaining_kwh,
                    'energy_consumption_kw': m.energy_consumption_kw,
                    'step_reward': m.step_reward,
                    'collision': m.collision,
                    'altitude_violation': m.altitude_violation
                })

            df = pd.DataFrame(data)
            logger.debug(f"Retrieved {len(df)} metrics for episode {episode_id}")
            return df

        finally:
            session.close()

    def get_vehicle_performance(self, vehicle_type: str) -> Dict[str, float]:
        """
        Get aggregated performance metrics for a vehicle type.

        Args:
            vehicle_type: Vehicle type (small, medium, large)

        Returns:
            Dictionary with performance stats
        """
        session = self.SessionLocal()
        try:
            stats = session.query(
                func.count(Episode.id).label('total_episodes'),
                func.avg(Episode.total_reward).label('avg_reward'),
                func.max(Episode.total_reward).label('max_reward'),
                func.avg(Episode.duration_seconds).label('avg_duration'),
                func.sum(Episode.collision_count).label('total_collisions'),
                func.sum(Episode.altitude_violations).label('total_violations'),
                func.count(Episode.id).filter(
                    Episode.successful_completion == True
                ).label('successful_episodes')
            ).join(Vehicle).filter(
                Vehicle.type == vehicle_type
            ).first()

            result = {
                'total_episodes': stats.total_episodes or 0,
                'avg_reward': float(stats.avg_reward) if stats.avg_reward else 0.0,
                'max_reward': float(stats.max_reward) if stats.max_reward else 0.0,
                'avg_duration': float(stats.avg_duration) if stats.avg_duration else 0.0,
                'total_collisions': stats.total_collisions or 0,
                'total_violations': stats.total_violations or 0,
                'successful_episodes': stats.successful_episodes or 0,
                'success_rate': (
                    stats.successful_episodes / stats.total_episodes
                    if stats.total_episodes > 0 else 0.0
                )
            }

            logger.info(f"Performance for {vehicle_type}: {result}")
            return result

        finally:
            session.close()

    def get_training_progress(
        self,
        vehicle_type: Optional[str] = None,
        window_size: int = 100
    ) -> pd.DataFrame:
        """
        Get training progress over episodes with rolling averages.

        Args:
            vehicle_type: Filter by vehicle type
            window_size: Rolling average window size

        Returns:
            DataFrame with episode rewards and rolling averages
        """
        session = self.SessionLocal()
        try:
            query = session.query(
                Episode.episode_number,
                Episode.total_reward,
                Episode.collision_count,
                Episode.successful_completion
            ).join(Vehicle)

            if vehicle_type:
                query = query.filter(Vehicle.type == vehicle_type)

            query = query.order_by(Episode.episode_number)

            # Convert to DataFrame
            results = query.all()
            df = pd.DataFrame(results, columns=[
                'episode_number', 'reward', 'collisions', 'success'
            ])

            # Add rolling averages
            if len(df) > 0:
                df['reward_rolling_avg'] = df['reward'].rolling(
                    window=window_size, min_periods=1
                ).mean()
                df['success_rate'] = df['success'].rolling(
                    window=window_size, min_periods=1
                ).mean()

            logger.info(f"Retrieved training progress: {len(df)} episodes")
            return df

        finally:
            session.close()

    def get_training_runs(self, status: Optional[str] = None) -> pd.DataFrame:
        """
        Get all training runs.

        Args:
            status: Filter by status (running, completed, failed)

        Returns:
            DataFrame with training run data
        """
        session = self.SessionLocal()
        try:
            query = session.query(TrainingRun)

            if status:
                query = query.filter(TrainingRun.status == status)

            query = query.order_by(TrainingRun.start_time.desc())

            runs = query.all()
            data = []
            for run in runs:
                data.append({
                    'id': run.id,
                    'name': run.name,
                    'algorithm': run.algorithm,
                    'status': run.status,
                    'start_time': run.start_time,
                    'end_time': run.end_time,
                    'total_episodes': run.total_episodes,
                    'best_reward': run.best_reward,
                    'convergence_episode': run.convergence_episode
                })

            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} training runs")
            return df

        finally:
            session.close()

    def export_episodes_to_csv(
        self,
        output_path: str,
        vehicle_type: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """
        Export episodes to CSV for external analysis.

        Args:
            output_path: Output CSV file path
            vehicle_type: Filter by vehicle type
            limit: Maximum number of episodes
        """
        df = self.query_episodes(
            vehicle_type=vehicle_type,
            limit=limit
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} episodes to {output_path}")

    def export_episode_trajectories(
        self,
        episode_id: int,
        output_path: str
    ):
        """
        Export episode trajectory data to CSV.

        Args:
            episode_id: Episode ID
            output_path: Output CSV file path
        """
        df = self.get_episode_metrics(episode_id)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Exported trajectory for episode {episode_id} to {output_path}")

    def delete_old_episodes(
        self,
        days_old: int = 30,
        keep_successful: bool = True,
        dry_run: bool = True
    ) -> int:
        """
        Delete old episodes to free space.

        Args:
            days_old: Delete episodes older than this many days
            keep_successful: Keep successful episodes regardless of age
            dry_run: If True, only count episodes that would be deleted

        Returns:
            Number of episodes deleted (or would be deleted if dry_run)
        """
        from datetime import timedelta

        session = self.SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            query = session.query(Episode).filter(
                Episode.start_time < cutoff_date
            )

            if keep_successful:
                query = query.filter(Episode.successful_completion == False)

            count = query.count()

            if not dry_run and count > 0:
                query.delete()
                session.commit()
                logger.info(f"Deleted {count} old episodes")
            else:
                logger.info(f"Would delete {count} episodes (dry run)")

            return count

        finally:
            session.close()

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database stats
        """
        session = self.SessionLocal()
        try:
            stats = {
                'total_episodes': session.query(Episode).count(),
                'total_metrics': session.query(Metric).count(),
                'total_sensor_logs': session.query(SensorLog).count(),
                'total_training_runs': session.query(TrainingRun).count(),
                'total_vehicles': session.query(Vehicle).count(),
                'total_arenas': session.query(Arena).count(),
            }

            # Get most recent episode
            latest_episode = session.query(Episode).order_by(
                Episode.start_time.desc()
            ).first()

            if latest_episode:
                stats['latest_episode_time'] = latest_episode.start_time

            logger.info(f"Database stats: {stats}")
            return stats

        finally:
            session.close()

    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("DatabaseManager closed")
