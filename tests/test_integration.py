"""
Integration tests for complete workflows.
"""
import pytest
import numpy as np


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_full_episode_with_logging(self, temp_db, sample_vehicle_config, sample_arena_config):
        """Test a complete episode with database logging."""
        from src.environments.evtol_gym_env import EVTOLEnv
        from src.database import DatabaseLogger

        # Create environment and logger
        env = EVTOLEnv(
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )
        logger = DatabaseLogger(db_type='sqlite', db_path=temp_db)

        # Run an episode
        obs, info = env.reset()
        total_reward = 0
        trajectory = []

        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            trajectory.append(info.get('position', [0, 0, 450]))

            if terminated or truncated:
                break

        # Log episode
        episode_id = logger.log_episode(
            vehicle_type='medium',
            arena_name='test_arena',
            total_reward=total_reward,
            episode_length=step + 1,
            success=not terminated,
            trajectory=np.array(trajectory)
        )

        assert episode_id is not None

        # Query the episode back
        episodes = logger.query_episodes(limit=1)
        assert len(episodes) > 0

        logger.close()

    def test_multi_episode_training_workflow(self, temp_db, sample_vehicle_config, sample_arena_config):
        """Test multiple episode workflow."""
        from src.environments.evtol_gym_env import EVTOLEnv
        from src.database import DatabaseLogger

        env = EVTOLEnv(
            vehicle_config=sample_vehicle_config['small'],
            arena_config=sample_arena_config
        )
        logger = DatabaseLogger(db_type='sqlite', db_path=temp_db)

        num_episodes = 5
        episode_ids = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            total_reward = 0

            for step in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            episode_id = logger.log_episode(
                vehicle_type='small',
                arena_name='test_arena',
                total_reward=total_reward,
                episode_length=step + 1,
                success=True
            )

            episode_ids.append(episode_id)

        assert len(episode_ids) == num_episodes

        # Query all episodes
        episodes = logger.query_episodes(vehicle_type='small', limit=10)
        assert len(episodes) == num_episodes

        logger.close()


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""

    def test_episode_throughput(self, sample_vehicle_config, sample_arena_config):
        """Test episode throughput."""
        import time
        from src.environments.evtol_gym_env import EVTOLEnv

        env = EVTOLEnv(
            vehicle_config=sample_vehicle_config['small'],
            arena_config=sample_arena_config
        )

        num_episodes = 10
        start_time = time.time()

        for _ in range(num_episodes):
            obs, info = env.reset()
            for _ in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

        elapsed_time = time.time() - start_time
        episodes_per_second = num_episodes / elapsed_time

        # Just verify it completes and we can measure throughput
        assert episodes_per_second > 0

    def test_multiagent_scalability(self, sample_vehicle_config, sample_arena_config):
        """Test multi-agent environment scalability."""
        from src.environments.evtol_multiagent_env import EVTOLMultiAgentEnv

        for num_agents in [5, 10, 20]:
            env = EVTOLMultiAgentEnv(
                num_agents=num_agents,
                vehicle_config=sample_vehicle_config['medium'],
                arena_config=sample_arena_config
            )

            observations = env.reset()
            assert len(observations) == num_agents

            # Take a few steps
            for _ in range(10):
                actions = {i: env.action_space.sample() for i in range(num_agents)}
                observations, rewards, dones, infos = env.step(actions)

                assert len(observations) == num_agents
