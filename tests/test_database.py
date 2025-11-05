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

        logger = DatabaseLogger(db_type='sqlite')
        assert logger is not None
        assert logger.db_type == 'sqlite'
        logger.close()

    def test_log_episode(self, temp_db, sample_trajectory):
        """Test logging a complete episode."""
        from src.database import DatabaseLogger

        logger = DatabaseLogger(db_type='sqlite')

        # Start episode
        episode_id = logger.start_episode(
            episode_number=1,
            vehicle_type='medium',
            arena_name='test_arena',
            num_agents=1
        )

        # Log some timesteps
        for i in range(10):
            logger.log_timestep(
                timestep=i,
                position=sample_trajectory['positions'][i],
                velocity=5.0,
                altitude_ft=450.0,
                battery_kwh=50.0,
                energy_consumption_kw=2.0,
                step_reward=1.5
            )

        # End episode
        logger.end_episode(
            total_reward=150.5,
            collision_count=0,
            altitude_violations=0,
            successful=True
        )

        assert episode_id is not None
        assert episode_id > 0
        logger.close()

    def test_log_metrics(self, temp_db):
        """Test logging timestep metrics."""
        from src.database import DatabaseLogger

        logger = DatabaseLogger(db_type='sqlite')

        # First create an episode
        episode_id = logger.start_episode(
            episode_number=1,
            vehicle_type='small',
            arena_name='test_arena',
            num_agents=1
        )

        # Log metrics for the episode
        logger.log_timestep(
            timestep=10,
            position=np.array([100, 200, 450]),
            velocity=5.0,
            altitude_ft=450.0,
            battery_kwh=85.5,
            energy_consumption_kw=2.0,
            step_reward=2.5,
            collision=False
        )

        # End episode
        logger.end_episode(
            total_reward=100.0,
            successful=True
        )

        logger.close()

    def test_query_episodes(self, temp_db):
        """Test querying episodes from database."""
        from src.database import DatabaseLogger
        from scripts.init_db import Episode

        logger = DatabaseLogger(db_type='sqlite')

        # Log multiple episodes
        for i in range(5):
            episode_id = logger.start_episode(
                episode_number=i,
                vehicle_type='medium',
                arena_name='test_arena',
                num_agents=1
            )
            logger.end_episode(
                total_reward=100.0 + i * 10,
                successful=True
            )

        # Query episodes using session
        session = logger.get_session()
        try:
            episodes = session.query(Episode).limit(10).all()
            assert len(episodes) == 5
        finally:
            session.close()

        logger.close()


@pytest.mark.database
class TestDatabaseCompression:
    """Test database compression utilities."""

    def test_compress_trajectory(self):
        """Test trajectory data compression."""
        from src.database import CompressionUtils

        positions = np.random.randn(1000, 3).astype(np.float32)
        velocities = np.random.randn(1000, 3).astype(np.float32)
        timestamps = np.arange(1000).astype(np.float32)

        compressed = CompressionUtils.compress_trajectory(positions, velocities, timestamps)

        assert isinstance(compressed, dict)
        assert 'positions' in compressed
        assert 'velocities' in compressed
        assert 'timestamps' in compressed

        decompressed = CompressionUtils.decompress_trajectory(compressed)
        np.testing.assert_array_almost_equal(positions, decompressed['positions'], decimal=5)
        np.testing.assert_array_almost_equal(velocities, decompressed['velocities'], decimal=5)
        np.testing.assert_array_almost_equal(timestamps, decompressed['timestamps'], decimal=5)

    def test_compress_sensor_data(self):
        """Test sensor data compression."""
        from src.database import CompressionUtils

        sensor_data = {
            'lidar': np.random.randn(500, 3).astype(np.float32),
            'imu_acc': np.random.randn(3).astype(np.float32),
            'imu_gyro': np.random.randn(3).astype(np.float32)
        }

        compressed = CompressionUtils.compress_sensor_data(sensor_data)
        assert isinstance(compressed, str)

        decompressed = CompressionUtils.decompress_sensor_data(compressed)
        assert 'lidar' in decompressed
        np.testing.assert_array_almost_equal(
            sensor_data['lidar'],
            decompressed['lidar'],
            decimal=5
        )
