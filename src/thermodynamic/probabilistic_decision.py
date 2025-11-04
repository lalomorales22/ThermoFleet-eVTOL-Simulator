"""
Probabilistic Decision Making using Thermodynamic Computing

Implements decision-making for eVTOL agents using energy-based models and
probabilistic inference inspired by thermodynamic principles.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import jax
import jax.numpy as jnp
from enum import Enum


class Action(Enum):
    """Possible eVTOL actions."""
    HOVER = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    UP = 5
    DOWN = 6
    LAND = 7
    EMERGENCY_STOP = 8


class ThermodynamicDecisionMaker:
    """
    Makes flight decisions using thermodynamic computing principles.

    Treats decision-making as sampling from a probability distribution where
    better decisions have lower energy (analogous to thermodynamic equilibrium).

    Uses concepts from THRML:
    - Energy-based models for action selection
    - Gibbs sampling for exploration
    - Block-wise updates for multi-objective optimization
    """

    def __init__(
        self,
        beta: float = 2.0,  # Inverse temperature (higher = more deterministic)
        energy_horizon: int = 10,  # Look-ahead steps
    ):
        """
        Initialize thermodynamic decision maker.

        Args:
            beta: Inverse temperature for Boltzmann distribution
            energy_horizon: Number of steps to look ahead for energy computation
        """
        self.beta = beta
        self.energy_horizon = energy_horizon

        # Energy weights for decision factors
        self.decision_weights = {
            'goal_progress': 2.0,      # Progress toward goal
            'safety': 5.0,             # Safety/collision avoidance
            'energy_efficiency': 1.0,  # Battery conservation
            'altitude_compliance': 3.0, # Stay in 400-500 ft range
            'smoothness': 0.5,         # Avoid jerky movements
            'coordination': 1.5,       # Multi-agent coordination
        }

    def compute_action_energy(
        self,
        action: Action,
        state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> float:
        """
        Compute energy (cost) of taking an action in current state.

        Lower energy = better action (thermodynamic minimum principle).

        Args:
            action: Proposed action
            state: Current agent state (position, velocity, battery, etc.)
            context: Environmental context (obstacles, goal, other agents, etc.)

        Returns:
            Energy value (scalar)
        """
        energy = 0.0

        # Extract state information
        position = np.array(state.get('position', [0, 0, 0]))
        velocity = np.array(state.get('velocity', [0, 0, 0]))
        battery = state.get('battery', 100.0)
        altitude_ft = position[2] * 3.28084  # m to ft

        # Extract context
        goal = np.array(context.get('goal', position))
        obstacles = context.get('obstacles', [])
        nearby_agents = context.get('nearby_agents', [])

        # Simulate action effect (simplified)
        next_position = self._simulate_action(position, velocity, action)

        # 1. Goal progress energy (negative = good)
        current_dist = np.linalg.norm(position - goal)
        next_dist = np.linalg.norm(next_position - goal)
        progress = current_dist - next_dist
        energy -= self.decision_weights['goal_progress'] * progress

        # 2. Safety energy
        min_obstacle_dist = float('inf')
        for obs in obstacles:
            obs_pos = np.array(obs['position'])
            dist = np.linalg.norm(next_position - obs_pos)
            min_obstacle_dist = min(min_obstacle_dist, dist)

        if min_obstacle_dist < 50:  # Within 50m of obstacle
            energy += self.decision_weights['safety'] * (50 - min_obstacle_dist)

        # 3. Energy efficiency
        action_cost = self._get_action_energy_cost(action, velocity)
        energy += self.decision_weights['energy_efficiency'] * action_cost

        # 4. Altitude compliance
        target_altitude_ft = (400 + 500) / 2  # Target middle of range
        next_altitude_ft = next_position[2] * 3.28084

        if next_altitude_ft < 400 or next_altitude_ft > 500:
            altitude_violation = abs(next_altitude_ft - np.clip(next_altitude_ft, 400, 500))
            energy += self.decision_weights['altitude_compliance'] * altitude_violation
        else:
            # Prefer middle of altitude range
            energy += self.decision_weights['altitude_compliance'] * 0.1 * abs(next_altitude_ft - target_altitude_ft)

        # 5. Smoothness (penalize sudden changes)
        if action in [Action.EMERGENCY_STOP]:
            energy += self.decision_weights['smoothness'] * 10

        # 6. Multi-agent coordination
        for agent in nearby_agents:
            agent_pos = np.array(agent['position'])
            dist = np.linalg.norm(next_position - agent_pos)
            if dist < 30:  # Minimum separation 30m
                energy += self.decision_weights['coordination'] * (30 - dist)

        return energy

    def _simulate_action(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        action: Action,
        dt: float = 1.0,
    ) -> np.ndarray:
        """Simulate the effect of an action (simplified kinematics)."""
        action_to_accel = {
            Action.HOVER: np.array([0, 0, 0]),
            Action.FORWARD: np.array([5, 0, 0]),
            Action.BACKWARD: np.array([-5, 0, 0]),
            Action.LEFT: np.array([0, -5, 0]),
            Action.RIGHT: np.array([0, 5, 0]),
            Action.UP: np.array([0, 0, 2]),
            Action.DOWN: np.array([0, 0, -2]),
            Action.LAND: np.array([0, 0, -5]),
            Action.EMERGENCY_STOP: np.array([0, 0, 0]),
        }

        accel = action_to_accel.get(action, np.array([0, 0, 0]))
        next_velocity = velocity + accel * dt
        next_position = position + next_velocity * dt

        return next_position

    def _get_action_energy_cost(self, action: Action, velocity: np.ndarray) -> float:
        """Get battery energy cost of an action."""
        base_costs = {
            Action.HOVER: 1.0,
            Action.FORWARD: 2.0,
            Action.BACKWARD: 2.5,
            Action.LEFT: 2.0,
            Action.RIGHT: 2.0,
            Action.UP: 3.0,
            Action.DOWN: 1.0,
            Action.LAND: 1.5,
            Action.EMERGENCY_STOP: 0.5,
        }
        return base_costs.get(action, 2.0)

    def sample_action_thermodynamic(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any],
        available_actions: List[Action] = None,
    ) -> Tuple[Action, float, Dict]:
        """
        Sample an action from the Boltzmann distribution over actions.

        P(action) ∝ exp(-beta * energy(action))

        This is the core thermodynamic computing principle: actions with lower
        energy are exponentially more likely.

        Args:
            state: Current agent state
            context: Environmental context
            available_actions: List of actions to consider (None = all actions)

        Returns:
            (selected_action, action_energy, metadata)
        """
        if available_actions is None:
            available_actions = list(Action)

        # Compute energy for each action
        energies = []
        for action in available_actions:
            energy = self.compute_action_energy(action, state, context)
            energies.append(energy)

        energies = np.array(energies)

        # Boltzmann distribution: P(action) ∝ exp(-beta * energy)
        log_probabilities = -self.beta * energies
        # Numerical stability: subtract max
        log_probabilities -= log_probabilities.max()
        probabilities = np.exp(log_probabilities)
        probabilities /= probabilities.sum()

        # Sample action
        selected_idx = np.random.choice(len(available_actions), p=probabilities)
        selected_action = available_actions[selected_idx]
        selected_energy = energies[selected_idx]

        metadata = {
            'energies': dict(zip([a.name for a in available_actions], energies)),
            'probabilities': dict(zip([a.name for a in available_actions], probabilities)),
            'beta': self.beta,
            'thermodynamic': True,
        }

        return selected_action, selected_energy, metadata

    def get_greedy_action(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any],
        available_actions: List[Action] = None,
    ) -> Tuple[Action, float]:
        """
        Get the action with minimum energy (greedy/deterministic).

        Equivalent to thermodynamic ground state (zero temperature).

        Args:
            state: Current agent state
            context: Environmental context
            available_actions: List of actions to consider

        Returns:
            (best_action, min_energy)
        """
        if available_actions is None:
            available_actions = list(Action)

        best_action = None
        min_energy = float('inf')

        for action in available_actions:
            energy = self.compute_action_energy(action, state, context)
            if energy < min_energy:
                min_energy = energy
                best_action = action

        return best_action, min_energy

    def anneal_temperature(self, step: int, max_steps: int, beta_min: float = 0.5, beta_max: float = 5.0):
        """
        Simulated annealing: adjust temperature over time.

        Start with high temperature (exploration) and cool down (exploitation).

        Args:
            step: Current step
            max_steps: Total steps
            beta_min: Minimum beta (high temperature, more exploration)
            beta_max: Maximum beta (low temperature, more exploitation)
        """
        # Linear annealing
        progress = step / max_steps
        self.beta = beta_min + (beta_max - beta_min) * progress
