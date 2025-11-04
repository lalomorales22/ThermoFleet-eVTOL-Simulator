"""
Unit tests for RL training pipeline.
"""
import pytest
import numpy as np


@pytest.mark.unit
class TestPPOTrainer:
    """Test PPO trainer functionality."""

    def test_trainer_initialization(self, sample_vehicle_config, sample_arena_config):
        """Test PPO trainer initialization."""
        from src.training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        assert trainer is not None

    @pytest.mark.slow
    @pytest.mark.convergence
    def test_short_training_run(self, sample_vehicle_config, sample_arena_config):
        """Test short training run (convergence not expected)."""
        from src.training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        # Run very short training
        trainer.train(total_timesteps=1000, save_model=False)

        # Just verify it completes without errors
        assert True


@pytest.mark.unit
class TestDDPGTrainer:
    """Test DDPG trainer functionality."""

    def test_trainer_initialization(self, sample_vehicle_config, sample_arena_config):
        """Test DDPG trainer initialization."""
        from src.training.ddpg_trainer import DDPGTrainer

        trainer = DDPGTrainer(
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        assert trainer is not None


@pytest.mark.integration
class TestTrainingCallbacks:
    """Test training callbacks."""

    def test_database_logging_callback(self, temp_db, sample_vehicle_config, sample_arena_config):
        """Test database logging callback."""
        from src.database import DatabaseLogger
        from src.database.callbacks import DatabaseLoggingCallback

        db_logger = DatabaseLogger(db_type='sqlite', db_path=temp_db)

        callback = DatabaseLoggingCallback(
            db_logger=db_logger,
            training_run_name="test_run",
            algorithm="PPO",
            vehicle_type="medium",
            hyperparameters={'learning_rate': 3e-4}
        )

        assert callback is not None
        assert callback.training_run_id is not None

        db_logger.close()


@pytest.mark.unit
class TestPufferWrapper:
    """Test PufferLib wrapper."""

    def test_wrapper_compatibility(self, sample_vehicle_config, sample_arena_config):
        """Test PufferLib wrapper for environment."""
        from src.training.puffer_wrapper import PufferEVTOLWrapper
        from src.environments.evtol_gym_env import EVTOLEnv

        base_env = EVTOLEnv(
            vehicle_config=sample_vehicle_config['medium'],
            arena_config=sample_arena_config
        )

        wrapped_env = PufferEVTOLWrapper(base_env)

        assert wrapped_env is not None

        # Test reset
        obs, info = wrapped_env.reset()
        assert obs is not None

        # Test step
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        assert obs is not None
        assert isinstance(reward, (int, float))
