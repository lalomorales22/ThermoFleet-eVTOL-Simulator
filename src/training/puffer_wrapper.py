"""
PufferLib wrapper for high-performance eVTOL training.
Provides optimized vectorized environments and data collection.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

# Note: PufferLib may need to be installed from GitHub
# pip install git+https://github.com/PufferAI/PufferLib.git
try:
    import pufferlib
    import pufferlib.vector
    PUFFER_AVAILABLE = True
except ImportError:
    PUFFER_AVAILABLE = False
    logging.warning("PufferLib not available. Install with: pip install git+https://github.com/PufferAI/PufferLib.git")

from ..environments.evtol_gym_env import EVTOLEnv

logger = logging.getLogger(__name__)


class PufferEVTOLWrapper:
    """
    Wrapper for eVTOL environment to work with PufferLib's optimized training.

    PufferLib provides:
    - Efficient vectorized environments
    - Optimized data collection and batching
    - Memory-efficient replay buffers
    - Fast PPO implementation
    """

    def __init__(
        self,
        num_envs: int = 128,
        vehicle_type: str = "medium",
        max_steps: int = 1000,
        enable_wind: bool = True,
        **kwargs
    ):
        """
        Initialize PufferLib wrapper.

        Args:
            num_envs: Number of parallel environments
            vehicle_type: Type of vehicle to train
            max_steps: Max steps per episode
            enable_wind: Whether to enable wind simulation
            **kwargs: Additional arguments for EVTOLEnv
        """
        if not PUFFER_AVAILABLE:
            raise ImportError("PufferLib is required but not installed")

        self.num_envs = num_envs
        self.vehicle_type = vehicle_type
        self.max_steps = max_steps
        self.enable_wind = enable_wind
        self.kwargs = kwargs

        logger.info(f"Initializing PufferLib with {num_envs} parallel environments")

    def make_env(self):
        """Factory function to create a single environment."""
        return EVTOLEnv(
            vehicle_type=self.vehicle_type,
            max_steps=self.max_steps,
            enable_wind=self.enable_wind,
            **self.kwargs
        )

    def create_vectorized_env(self):
        """Create vectorized environment for parallel training."""
        if not PUFFER_AVAILABLE:
            raise ImportError("PufferLib is required")

        # Create vectorized environment using PufferLib
        vec_env = pufferlib.vector.make(
            self.make_env,
            num_envs=self.num_envs,
            num_workers=4,  # Number of worker processes
            batch_size=self.num_envs,
        )

        logger.info(f"Created vectorized environment with {self.num_envs} instances")
        return vec_env


class PufferConfig:
    """Configuration for PufferLib training."""

    def __init__(
        self,
        # Environment settings
        num_envs: int = 128,
        num_steps: int = 128,
        num_minibatches: int = 4,

        # PPO hyperparameters
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,

        # Training settings
        total_timesteps: int = 10_000_000,
        update_epochs: int = 4,
        normalize_advantage: bool = True,
        clip_vloss: bool = True,

        # Logging
        log_interval: int = 10,
        save_interval: int = 100,

        # Device
        device: str = "cuda",
    ):
        """
        Initialize PufferLib training configuration.

        Args:
            num_envs: Number of parallel environments
            num_steps: Number of steps per environment per update
            num_minibatches: Number of minibatches for updates
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_coef: PPO clipping coefficient
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            total_timesteps: Total training timesteps
            update_epochs: Number of epochs per update
            normalize_advantage: Whether to normalize advantages
            clip_vloss: Whether to clip value loss
            log_interval: Steps between logging
            save_interval: Steps between model saves
            device: Device to use (cuda/cpu)
        """
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_minibatches = num_minibatches
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps
        self.update_epochs = update_epochs
        self.normalize_advantage = normalize_advantage
        self.clip_vloss = clip_vloss
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.device = device

        # Calculated values
        self.batch_size = num_envs * num_steps
        self.minibatch_size = self.batch_size // num_minibatches
        self.num_updates = total_timesteps // self.batch_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_envs": self.num_envs,
            "num_steps": self.num_steps,
            "num_minibatches": self.num_minibatches,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_coef": self.clip_coef,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "total_timesteps": self.total_timesteps,
            "update_epochs": self.update_epochs,
            "normalize_advantage": self.normalize_advantage,
            "clip_vloss": self.clip_vloss,
            "batch_size": self.batch_size,
            "minibatch_size": self.minibatch_size,
            "num_updates": self.num_updates,
            "device": self.device,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"PufferConfig({self.to_dict()})"


def test_puffer_integration():
    """Test function to verify PufferLib integration."""
    if not PUFFER_AVAILABLE:
        logger.error("PufferLib not available. Skipping test.")
        return False

    try:
        # Create wrapper
        wrapper = PufferEVTOLWrapper(num_envs=4, vehicle_type="medium")

        # Test single environment creation
        env = wrapper.make_env()
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        logger.info("PufferLib integration test passed!")
        return True

    except Exception as e:
        logger.error(f"PufferLib integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Test integration
    logging.basicConfig(level=logging.INFO)
    test_puffer_integration()
