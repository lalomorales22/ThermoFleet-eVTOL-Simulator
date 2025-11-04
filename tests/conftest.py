"""
Pytest configuration and shared fixtures for FlyingCarRL tests.
"""
import pytest
import numpy as np
import os
import tempfile
from pathlib import Path


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_vehicle_config():
    """Sample vehicle configuration for testing."""
    return {
        'small': {
            'mass': 100,
            'max_thrust': 1200,
            'drag_coefficient': 0.3,
            'battery_capacity': 20000,
            'shape': 'sphere'
        },
        'medium': {
            'mass': 500,
            'max_thrust': 6000,
            'drag_coefficient': 0.4,
            'battery_capacity': 80000,
            'shape': 'winged'
        },
        'large': {
            'mass': 1000,
            'max_thrust': 12000,
            'drag_coefficient': 0.5,
            'battery_capacity': 150000,
            'shape': 'boxy'
        }
    }


@pytest.fixture
def sample_arena_config():
    """Sample arena configuration for testing."""
    return {
        'name': 'test_arena',
        'bounds': {
            'x_min': -1000,
            'x_max': 1000,
            'y_min': -1000,
            'y_max': 1000,
            'z_min': 400,
            'z_max': 500
        },
        'obstacles': [],
        'wind_config': {
            'enabled': True,
            'mean_velocity': [0, 0, 0],
            'std_velocity': [2, 2, 1]
        }
    }


@pytest.fixture
def sample_trajectory():
    """Generate a sample flight trajectory for testing."""
    num_steps = 100
    positions = np.random.randn(num_steps, 3) * 10 + np.array([0, 0, 450])
    velocities = np.random.randn(num_steps, 3) * 2
    return {
        'positions': positions,
        'velocities': velocities,
        'timestamps': np.arange(num_steps) * 0.1
    }


@pytest.fixture
def mock_sensor_data():
    """Generate mock sensor data for testing."""
    return {
        'camera': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        'lidar': np.random.randn(1000, 3),
        'imu': {
            'acceleration': np.random.randn(3),
            'angular_velocity': np.random.randn(3)
        },
        'gps': {
            'position': np.array([40.7128, -74.0060, 450.0]),
            'accuracy': 2.5
        }
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
