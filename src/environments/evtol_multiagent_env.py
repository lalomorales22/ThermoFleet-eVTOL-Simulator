"""
Multi-Agent eVTOL Environment using PettingZoo
Supports fleet coordination and multi-vehicle scenarios.
"""

import numpy as np
from typing import Dict, Optional, Any, List
from pettingzoo import ParallelEnv
from gymnasium import spaces
import logging

from .evtol_gym_env import EVTOLEnv

logger = logging.getLogger(__name__)


class EVTOLMultiAgentEnv(ParallelEnv):
    """
    Multi-agent parallel environment for eVTOL fleet coordination.

    This environment supports:
    - Multiple vehicles operating simultaneously
    - Collision avoidance between vehicles
    - Cooperative task completion
    - Fleet coordination rewards

    Based on PettingZoo's ParallelEnv API for efficient parallel execution.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "evtol_multiagent_v0"}

    def __init__(
        self,
        num_agents: int = 4,
        vehicle_types: Optional[List[str]] = None,
        max_steps: int = 1000,
        altitude_min: float = 400.0,
        altitude_max: float = 500.0,
        arena_bounds: tuple = (1000.0, 1000.0, 100.0),
        enable_wind: bool = True,
        enable_collisions: bool = True,
        coordination_reward: bool = True,
        min_separation: float = 20.0,  # meters
        render_mode: Optional[str] = None,
    ):
        """
        Initialize multi-agent eVTOL environment.

        Args:
            num_agents: Number of vehicles in the fleet
            vehicle_types: List of vehicle types for each agent (default: all "medium")
            max_steps: Maximum steps per episode
            altitude_min: Minimum allowed altitude (feet)
            altitude_max: Maximum allowed altitude (feet)
            arena_bounds: (x, y, z) bounds in meters
            enable_wind: Whether to simulate wind turbulence
            enable_collisions: Whether to check inter-vehicle collisions
            coordination_reward: Whether to add coordination bonuses
            min_separation: Minimum safe separation between vehicles (meters)
            render_mode: Rendering mode
        """
        super().__init__()

        self.num_agents = num_agents
        self.max_steps = max_steps
        self.altitude_min = altitude_min
        self.altitude_max = altitude_max
        self.arena_bounds = np.array(arena_bounds)
        self.enable_wind = enable_wind
        self.enable_collisions = enable_collisions
        self.coordination_reward = coordination_reward
        self.min_separation = min_separation
        self.render_mode = render_mode

        # Vehicle types
        if vehicle_types is None:
            self.vehicle_types = ["medium"] * num_agents
        else:
            assert len(vehicle_types) == num_agents, "Must specify type for each agent"
            self.vehicle_types = vehicle_types

        # Agent IDs
        self.possible_agents = [f"evtol_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()

        # Create single-agent environments for each vehicle
        self.envs = {}
        for i, agent_id in enumerate(self.possible_agents):
            self.envs[agent_id] = EVTOLEnv(
                vehicle_type=self.vehicle_types[i],
                max_steps=max_steps,
                altitude_min=altitude_min,
                altitude_max=altitude_max,
                arena_bounds=arena_bounds,
                enable_wind=enable_wind,
                enable_sensor_noise=True,
                render_mode=None,  # Individual envs don't render
            )

        # Define spaces (same for all agents in this case)
        self._setup_spaces()

        # Global state
        self.current_step = 0
        self.collision_matrix = np.zeros((num_agents, num_agents))

        logger.info(f"Initialized Multi-Agent EVTOLEnv with {num_agents} agents")

    def _setup_spaces(self):
        """Setup observation and action spaces for all agents."""
        # Use the same spaces as single-agent env, but add relative positions of other agents
        base_env = self.envs[self.possible_agents[0]]

        # Extended observation includes relative positions to other agents
        # Base obs: 30 dims + (num_agents - 1) * 3 dims for relative positions
        extra_dims = (self.num_agents - 1) * 3

        self.observation_spaces = {}
        self.action_spaces = {}

        for agent_id in self.possible_agents:
            # Extended observation space
            low = np.concatenate([
                base_env.observation_space.low,
                np.full(extra_dims, -2000.0)  # Relative positions
            ])
            high = np.concatenate([
                base_env.observation_space.high,
                np.full(extra_dims, 2000.0)
            ])

            self.observation_spaces[agent_id] = spaces.Box(
                low=low,
                high=high,
                dtype=np.float32,
            )

            # Same action space as single-agent
            self.action_spaces[agent_id] = base_env.action_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset all agents."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.collision_matrix = np.zeros((self.num_agents, self.num_agents))

        observations = {}
        infos = {}

        # Reset each agent with different random seeds
        for i, agent_id in enumerate(self.agents):
            agent_seed = None if seed is None else seed + i
            obs, info = self.envs[agent_id].reset(seed=agent_seed)
            observations[agent_id] = obs
            infos[agent_id] = info

        # Add relative observations
        observations = self._add_relative_observations(observations)

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Execute one step for all agents.

        Args:
            actions: Dictionary mapping agent_id to action array

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        self.current_step += 1

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Step each agent
        for agent_id in self.agents:
            action = actions.get(agent_id, np.zeros(4))
            obs, reward, term, trunc, info = self.envs[agent_id].step(action)

            observations[agent_id] = obs
            rewards[agent_id] = reward
            terminations[agent_id] = term
            truncations[agent_id] = trunc
            infos[agent_id] = info

        # Add relative observations
        observations = self._add_relative_observations(observations)

        # Check inter-agent collisions
        if self.enable_collisions:
            self._check_inter_agent_collisions(rewards, terminations)

        # Add coordination rewards
        if self.coordination_reward:
            self._add_coordination_rewards(rewards)

        # Remove terminated agents
        self.agents = [
            agent_id for agent_id in self.agents
            if not terminations.get(agent_id, False)
        ]

        return observations, rewards, terminations, truncations, infos

    def _add_relative_observations(self, observations: Dict) -> Dict:
        """Add relative positions of other agents to observations."""
        positions = {}
        for agent_id in observations.keys():
            # Extract position from observation (first 3 values)
            positions[agent_id] = self.envs[agent_id].position

        # Add relative positions to each agent's observation
        extended_obs = {}
        for agent_id in observations.keys():
            base_obs = observations[agent_id]
            my_pos = positions[agent_id]

            # Calculate relative positions to all other agents
            relative_positions = []
            for other_id in self.possible_agents:
                if other_id != agent_id:
                    other_pos = positions.get(other_id, my_pos)
                    rel_pos = other_pos - my_pos
                    relative_positions.extend(rel_pos)

            # Concatenate
            extended_obs[agent_id] = np.concatenate([
                base_obs,
                np.array(relative_positions, dtype=np.float32)
            ])

        return extended_obs

    def _check_inter_agent_collisions(self, rewards: Dict, terminations: Dict):
        """Check for collisions between agents."""
        positions = {
            agent_id: self.envs[agent_id].position
            for agent_id in self.agents
        }

        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i >= j:
                    continue

                if agent_i not in positions or agent_j not in positions:
                    continue

                # Calculate distance
                distance = np.linalg.norm(positions[agent_i] - positions[agent_j])

                # Check collision
                if distance < self.min_separation:
                    # Record collision
                    self.collision_matrix[i, j] = 1
                    self.collision_matrix[j, i] = 1

                    # Apply penalties
                    if agent_i in rewards:
                        rewards[agent_i] -= 5.0
                    if agent_j in rewards:
                        rewards[agent_j] -= 5.0

                    # Terminate both agents
                    terminations[agent_i] = True
                    terminations[agent_j] = True

                    logger.warning(f"Collision between {agent_i} and {agent_j} at distance {distance:.2f}m")

    def _add_coordination_rewards(self, rewards: Dict):
        """Add bonuses for coordinated behavior."""
        if len(self.agents) < 2:
            return

        positions = {
            agent_id: self.envs[agent_id].position
            for agent_id in self.agents
        }

        # Reward for maintaining safe separation (not too close, not too far)
        for agent_id in self.agents:
            my_pos = positions[agent_id]
            distances = []

            for other_id in self.agents:
                if other_id == agent_id:
                    continue
                other_pos = positions[other_id]
                dist = np.linalg.norm(my_pos - other_pos)
                distances.append(dist)

            if distances:
                avg_distance = np.mean(distances)
                # Ideal separation: 30-50 meters
                if 30.0 <= avg_distance <= 50.0:
                    rewards[agent_id] += 0.2
                elif avg_distance < 25.0:
                    rewards[agent_id] -= 0.1  # Too close

        # Bonus for all agents maintaining altitude compliance
        all_in_compliance = all(
            self.altitude_min <= self.envs[agent_id].position[2] <= self.altitude_max
            for agent_id in self.agents
        )
        if all_in_compliance:
            for agent_id in self.agents:
                rewards[agent_id] += 0.3

    def render(self):
        """Render the multi-agent environment."""
        if self.render_mode == "human":
            print(f"\n=== Step {self.current_step} ===")
            for agent_id in self.agents:
                env = self.envs[agent_id]
                print(f"{agent_id}: Pos={env.position}, Bat={env.battery_level:.2f}")

    def close(self):
        """Clean up resources."""
        for env in self.envs.values():
            env.close()

    def observation_space(self, agent: str):
        """Get observation space for specific agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Get action space for specific agent."""
        return self.action_spaces[agent]
