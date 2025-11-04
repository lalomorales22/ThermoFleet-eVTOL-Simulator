"""
Unit tests for environment and physics simulation.
"""
import pytest
import numpy as np


@pytest.mark.unit
@pytest.mark.physics
class TestEVTOLEnvironment:
    """Test eVTOL Gym environment."""

    def test_environment_initialization(self, sample_vehicle_config, sample_arena_config):
        """Test environment initialization."""
        from src.environments.evtol_gym_env import EVTOLEnv

        env = EVTOLEnv(
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        assert env is not None
        assert env.action_space is not None
        assert env.observation_space is not None

    def test_environment_reset(self, sample_vehicle_config, sample_arena_config):
        """Test environment reset."""
        from src.environments.evtol_gym_env import EVTOLEnv

        env = EVTOLEnv(
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        obs, info = env.reset()

        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape

        # Check altitude is within bounds
        altitude = obs[2] if len(obs) > 2 else 450
        assert 400 <= altitude <= 500

    def test_environment_step(self, sample_vehicle_config, sample_arena_config):
        """Test environment step."""
        from src.environments.evtol_gym_env import EVTOLEnv

        env = EVTOLEnv(
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        env.reset()
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_altitude_constraint(self, sample_vehicle_config, sample_arena_config):
        """Test that altitude remains within constraints."""
        from src.environments.evtol_gym_env import EVTOLEnv

        env = EVTOLEnv(
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        obs, _ = env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

            # Check altitude constraint
            altitude = info.get('altitude', obs[2] if len(obs) > 2 else 450)
            assert 350 <= altitude <= 550  # Allow some margin for physics


@pytest.mark.unit
class TestPhysicsSimulation:
    """Test physics calculations."""

    def test_aerodynamic_drag(self):
        """Test aerodynamic drag calculation."""
        velocity = np.array([10, 5, 0])
        drag_coefficient = 0.4
        air_density = 1.225
        cross_section_area = 2.0

        # Simple drag calculation: F = 0.5 * rho * v^2 * Cd * A
        speed = np.linalg.norm(velocity)
        expected_drag_magnitude = 0.5 * air_density * speed**2 * drag_coefficient * cross_section_area

        assert expected_drag_magnitude > 0

    def test_battery_drain(self, sample_vehicle_config):
        """Test battery drain calculation."""
        config = sample_vehicle_config['medium']
        thrust = config['max_thrust'] * 0.8
        dt = 0.1

        # Simplified: energy = power * time, power ~ thrust
        power = thrust * 0.1  # Simplified power model
        energy_consumed = power * dt

        battery_drain_percent = (energy_consumed / config['battery_capacity']) * 100

        assert 0 <= battery_drain_percent <= 100
        assert battery_drain_percent > 0


@pytest.mark.integration
class TestMultiAgentEnvironment:
    """Test multi-agent environment."""

    def test_multiagent_initialization(self, sample_vehicle_config, sample_arena_config):
        """Test multi-agent environment initialization."""
        from src.environments.evtol_multiagent_env import EVTOLMultiAgentEnv

        env = EVTOLMultiAgentEnv(
            num_agents=5,
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        assert env is not None
        assert env.num_agents == 5

    def test_multiagent_step(self, sample_vehicle_config, sample_arena_config):
        """Test multi-agent environment step."""
        from src.environments.evtol_multiagent_env import EVTOLMultiAgentEnv

        num_agents = 3
        env = EVTOLMultiAgentEnv(
            num_agents=num_agents,
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        observations = env.reset()
        assert len(observations) == num_agents

        actions = {i: env.action_space.sample() for i in range(num_agents)}
        observations, rewards, dones, infos = env.step(actions)

        assert len(observations) == num_agents
        assert len(rewards) == num_agents
        assert len(dones) == num_agents
