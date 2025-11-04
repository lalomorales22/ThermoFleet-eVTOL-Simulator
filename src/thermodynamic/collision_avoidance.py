"""
Energy-Based Collision Avoidance using Thermodynamic Computing

Implements collision avoidance for eVTOL vehicles using energy-based models
that create repulsive potential fields around obstacles and other agents.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import jax
import jax.numpy as jnp


class EnergyBasedCollisionAvoidance:
    """
    Collision avoidance system using thermodynamic energy fields.

    Creates a potential energy landscape where:
    - Obstacles and other vehicles create high-energy (repulsive) regions
    - Goals create low-energy (attractive) regions
    - The agent naturally flows to low-energy states

    This is inspired by thermodynamic systems seeking equilibrium at minimum energy.
    """

    def __init__(
        self,
        safety_distance: float = 30.0,  # meters
        critical_distance: float = 15.0,  # meters
        energy_scale: float = 100.0,
    ):
        """
        Initialize energy-based collision avoidance.

        Args:
            safety_distance: Distance at which repulsion starts
            critical_distance: Distance at which repulsion becomes very strong
            energy_scale: Scaling factor for energy values
        """
        self.safety_distance = safety_distance
        self.critical_distance = critical_distance
        self.energy_scale = energy_scale

    def compute_obstacle_energy(
        self,
        position: np.ndarray,
        obstacle: Dict,
    ) -> float:
        """
        Compute repulsive energy from a single obstacle.

        Uses Lennard-Jones-like potential: very strong at close range,
        negligible at long range.

        Args:
            position: Agent position (x, y, z)
            obstacle: Obstacle dict with 'position' and 'radius'

        Returns:
            Repulsive energy (scalar)
        """
        obs_pos = np.array(obstacle['position'])
        obs_radius = obstacle.get('radius', 10.0)

        distance = np.linalg.norm(position - obs_pos)
        effective_distance = max(distance - obs_radius, 0.1)  # Avoid division by zero

        if effective_distance > self.safety_distance:
            return 0.0

        # Inverse square law with exponential barrier
        energy = self.energy_scale * (
            (self.critical_distance / effective_distance) ** 6
        )

        return energy

    def compute_agent_repulsion(
        self,
        position: np.ndarray,
        other_agent: Dict,
    ) -> float:
        """
        Compute repulsive energy from another agent.

        Args:
            position: This agent's position
            other_agent: Other agent dict with 'position' and optionally 'velocity'

        Returns:
            Repulsive energy
        """
        other_pos = np.array(other_agent['position'])
        distance = np.linalg.norm(position - other_pos)

        if distance > self.safety_distance:
            return 0.0

        # Similar to obstacle repulsion but slightly weaker
        # (agents can coordinate, obstacles cannot)
        energy = 0.7 * self.energy_scale * (
            (self.critical_distance / max(distance, 0.1)) ** 4
        )

        # If other agent is moving toward us, increase repulsion
        if 'velocity' in other_agent:
            other_vel = np.array(other_agent['velocity'])
            direction_to_us = position - other_pos
            if np.dot(other_vel, direction_to_us) > 0:
                # Other agent approaching us
                energy *= 1.5

        return energy

    def compute_total_repulsion_energy(
        self,
        position: np.ndarray,
        obstacles: List[Dict],
        other_agents: List[Dict],
    ) -> float:
        """
        Compute total repulsive energy from all obstacles and agents.

        Args:
            position: Agent position
            obstacles: List of obstacles
            other_agents: List of other agents

        Returns:
            Total repulsive energy
        """
        total_energy = 0.0

        # Obstacles
        for obs in obstacles:
            total_energy += self.compute_obstacle_energy(position, obs)

        # Other agents
        for agent in other_agents:
            total_energy += self.compute_agent_repulsion(position, agent)

        return total_energy

    def compute_energy_gradient(
        self,
        position: np.ndarray,
        obstacles: List[Dict],
        other_agents: List[Dict],
        epsilon: float = 0.1,
    ) -> np.ndarray:
        """
        Compute gradient of energy field at current position.

        The gradient points in the direction of steepest energy increase.
        Moving opposite to gradient reduces collision risk.

        Args:
            position: Current position
            obstacles: List of obstacles
            other_agents: List of other agents
            epsilon: Finite difference step size

        Returns:
            Energy gradient vector (3D)
        """
        gradient = np.zeros(3)

        current_energy = self.compute_total_repulsion_energy(
            position, obstacles, other_agents
        )

        # Finite differences for each dimension
        for dim in range(3):
            pos_plus = position.copy()
            pos_plus[dim] += epsilon

            energy_plus = self.compute_total_repulsion_energy(
                pos_plus, obstacles, other_agents
            )

            gradient[dim] = (energy_plus - current_energy) / epsilon

        return gradient

    def get_avoidance_vector(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        obstacles: List[Dict],
        other_agents: List[Dict],
        goal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get collision avoidance vector using energy gradient descent.

        The avoidance vector points away from high-energy regions.

        Args:
            position: Current position
            velocity: Current velocity
            obstacles: List of obstacles
            other_agents: List of other agents
            goal: Optional goal position (creates attractive force)

        Returns:
            (avoidance_vector, metadata)
        """
        # Compute repulsive gradient
        repulsion_gradient = self.compute_energy_gradient(
            position, obstacles, other_agents
        )

        # Move opposite to gradient (down the energy landscape)
        avoidance_vector = -repulsion_gradient

        # If goal provided, add attractive force
        if goal is not None:
            goal_direction = goal - position
            goal_distance = np.linalg.norm(goal_direction)
            if goal_distance > 0:
                # Attractive force proportional to distance
                attraction = (goal_direction / goal_distance) * min(goal_distance, 10.0)
                avoidance_vector += attraction * 0.5

        # Normalize
        magnitude = np.linalg.norm(avoidance_vector)
        if magnitude > 0:
            avoidance_vector = avoidance_vector / magnitude

        # Metadata
        metadata = {
            'repulsion_energy': self.compute_total_repulsion_energy(
                position, obstacles, other_agents
            ),
            'gradient_magnitude': np.linalg.norm(repulsion_gradient),
            'thermodynamic': True,
        }

        return avoidance_vector, metadata

    def is_collision_imminent(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        obstacles: List[Dict],
        other_agents: List[Dict],
        lookahead_time: float = 2.0,
    ) -> Tuple[bool, List[str]]:
        """
        Check if collision is imminent using energy-based prediction.

        Args:
            position: Current position
            velocity: Current velocity
            obstacles: List of obstacles
            other_agents: List of other agents
            lookahead_time: How far ahead to predict (seconds)

        Returns:
            (is_imminent, collision_sources)
        """
        is_imminent = False
        collision_sources = []

        # Predict future position
        future_position = position + velocity * lookahead_time

        # Check obstacles
        for i, obs in enumerate(obstacles):
            obs_pos = np.array(obs['position'])
            obs_radius = obs.get('radius', 10.0)
            distance = np.linalg.norm(future_position - obs_pos)

            if distance < (obs_radius + self.critical_distance):
                is_imminent = True
                collision_sources.append(f"obstacle_{i}")

        # Check other agents
        for i, agent in enumerate(other_agents):
            agent_pos = np.array(agent['position'])

            # If agent has velocity, predict their position too
            if 'velocity' in agent:
                agent_vel = np.array(agent['velocity'])
                future_agent_pos = agent_pos + agent_vel * lookahead_time
            else:
                future_agent_pos = agent_pos

            distance = np.linalg.norm(future_position - future_agent_pos)

            if distance < self.critical_distance:
                is_imminent = True
                collision_sources.append(f"agent_{i}")

        return is_imminent, collision_sources

    def compute_minimum_safe_distance(
        self,
        velocity1: np.ndarray,
        velocity2: np.ndarray,
    ) -> float:
        """
        Compute minimum safe distance based on relative velocity.

        Faster relative velocity requires larger separation.

        Args:
            velocity1: First agent velocity
            velocity2: Second agent velocity

        Returns:
            Minimum safe distance (meters)
        """
        relative_velocity = velocity1 - velocity2
        relative_speed = np.linalg.norm(relative_velocity)

        # Base safety distance plus velocity-dependent term
        # Assumes 2 second reaction time
        min_distance = self.critical_distance + relative_speed * 2.0

        return min_distance
