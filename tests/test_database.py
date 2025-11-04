"""
Unit tests for database functionality.
"""
import pytest
import numpy as np
from datetime import datetime


@pytest.mark.database
class TestDatabaseLogger:
    """Test DatabaseLogger functionality."""

    def test_logger_initialization_sqlite(self, temp_db):
        """Test SQLite database logger initialization."""
        from src.database import DatabaseLogger

        logger = DatabaseLogger(db_type='sqlite', db_path=temp_db)
        assert logger is not None
        assert logger.db_type == 'sqlite'
        logger.close()

    def test_log_episode(self, temp_db, sample_trajectory):
        """Test logging a complete episode."""
        from src.database import DatabaseLogger

        logger = DatabaseLogger(db_type='sqlite', db_path=temp_db)

        episode_id = logger.log_episode(
            vehicle_type='medium',
            arena_name='test_arena',
            total_reward=150.5,
            episode_length=100,
            success=True,
            trajectory=sample_trajectory['positions'],
            collision_occurred=False,
            final_battery_percent=45.2
        )

        assert episode_id is not None
        assert episode_id > 0
        logger.close()

    def test_log_metrics(self, temp_db):
        """Test logging timestep metrics."""
        from src.database import DatabaseLogger

        logger = DatabaseLogger(db_type='sqlite', db_path=temp_db)

        # First create an episode
        episode_id = logger.log_episode(
            vehicle_type='small',
            arena_name='test_arena',
            total_reward=100.0,
            episode_length=50,
            success=True
        )

        # Log metrics for the episode
        logger.log_metrics(
            episode_id=episode_id,
            timestep=10,
            position=np.array([100, 200, 450]),
            velocity=np.array([5, 3, 0.5]),
            altitude=450.0,
            battery_level=85.5,
            reward=2.5,
            collision=False
        )

        logger.close()

    def test_query_episodes(self, temp_db):
        """Test querying episodes from database."""
        from src.database import DatabaseLogger

        logger = DatabaseLogger(db_type='sqlite', db_path=temp_db)

        # Log multiple episodes
        for i in range(5):
            logger.log_episode(
                vehicle_type='medium',
                arena_name='test_arena',
                total_reward=100.0 + i * 10,
                episode_length=100,
                success=True
            )

        # Query episodes
        episodes = logger.query_episodes(vehicle_type='medium', limit=10)
        assert len(episodes) == 5

        logger.close()


@pytest.mark.database
class TestDatabaseCompression:
    """Test database compression utilities."""

    def test_compress_trajectory(self):
        """Test trajectory data compression."""
        from src.database.compression import compress_trajectory, decompress_trajectory

        trajectory = np.random.randn(1000, 3).astype(np.float32)
        compressed = compress_trajectory(trajectory)

        assert len(compressed) < trajectory.nbytes

        decompressed = decompress_trajectory(compressed)
        np.testing.assert_array_almost_equal(trajectory, decompressed, decimal=5)

    def test_compress_sensor_data(self):
        """Test sensor data compression."""
        from src.database.compression import compress_sensor_data, decompress_sensor_data

        sensor_data = {
            'lidar': np.random.randn(500, 3).astype(np.float32),
            'imu_acc': np.random.randn(3).astype(np.float32),
            'imu_gyro': np.random.randn(3).astype(np.float32)
        }

        compressed = compress_sensor_data(sensor_data)
        assert isinstance(compressed, bytes)

        decompressed = decompress_sensor_data(compressed)
        assert 'lidar' in decompressed
        np.testing.assert_array_almost_equal(
            sensor_data['lidar'],
            decompressed['lidar'],
            decimal=5
        )
