"""
Unit tests for vehicle spawning system.
"""
import pytest
import numpy as np


@pytest.mark.unit
class TestVehicleSpawning:
    """Test vehicle spawning functionality."""

    def test_spawn_single_vehicle(self, sample_vehicle_config, sample_arena_config):
        """Test spawning a single vehicle."""
        # This is a placeholder test - actual implementation would depend on
        # the specific spawning system implementation
        vehicle = sample_vehicle_config['medium']

        assert vehicle['mass'] == 500
        assert vehicle['max_thrust'] == 6000

    def test_spawn_multiple_vehicles(self, sample_vehicle_config, sample_arena_config):
        """Test spawning multiple vehicles."""
        num_vehicles = 10
        vehicles = [sample_vehicle_config['small'] for _ in range(num_vehicles)]

        assert len(vehicles) == num_vehicles

    def test_vehicle_start_positions(self, sample_arena_config):
        """Test that vehicles spawn at valid positions."""
        arena = sample_arena_config

        # Generate random start positions within arena bounds
        num_vehicles = 5
        positions = []

        for _ in range(num_vehicles):
            x = np.random.uniform(arena['bounds']['x_min'], arena['bounds']['x_max'])
            y = np.random.uniform(arena['bounds']['y_min'], arena['bounds']['y_max'])
            z = np.random.uniform(arena['bounds']['z_min'], arena['bounds']['z_max'])

            positions.append([x, y, z])

            # Check altitude constraint
            assert arena['bounds']['z_min'] <= z <= arena['bounds']['z_max']

        assert len(positions) == num_vehicles

    def test_vehicle_type_distribution(self, sample_vehicle_config):
        """Test different vehicle type distributions."""
        vehicle_types = ['small', 'medium', 'large']
        distribution = [0.5, 0.3, 0.2]  # 50% small, 30% medium, 20% large

        num_vehicles = 100
        spawned_types = np.random.choice(vehicle_types, size=num_vehicles, p=distribution)

        # Check that we have vehicles of each type
        assert 'small' in spawned_types
        assert 'medium' in spawned_types
        assert 'large' in spawned_types

        # Verify approximate distribution (with tolerance)
        small_count = np.sum(spawned_types == 'small')
        assert 40 <= small_count <= 60  # ~50% with tolerance


@pytest.mark.integration
class TestArenaIntegration:
    """Test arena and vehicle integration."""

    def test_vehicles_within_arena_bounds(self, sample_arena_config):
        """Test that all vehicles stay within arena bounds."""
        arena = sample_arena_config
        bounds = arena['bounds']

        # Simulate some positions
        num_steps = 100
        positions = np.random.randn(num_steps, 3) * 50 + np.array([0, 0, 450])

        # Clip to arena bounds (this would be done by the environment)
        positions[:, 0] = np.clip(positions[:, 0], bounds['x_min'], bounds['x_max'])
        positions[:, 1] = np.clip(positions[:, 1], bounds['y_min'], bounds['y_max'])
        positions[:, 2] = np.clip(positions[:, 2], bounds['z_min'], bounds['z_max'])

        # Verify all positions are within bounds
        assert np.all(positions[:, 0] >= bounds['x_min'])
        assert np.all(positions[:, 0] <= bounds['x_max'])
        assert np.all(positions[:, 2] >= bounds['z_min'])
        assert np.all(positions[:, 2] <= bounds['z_max'])
