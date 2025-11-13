"""
Thermodynamic Multi-Agent Coordination

Implements fleet coordination using thermodynamic computing and energy-based models
for scalable, efficient multi-agent decision making inspired by THRML.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import jax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class AgentState:
    """State of a single agent in the fleet."""
    agent_id: str
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    battery: float
    status: str = "active"


class ThermodynamicCoordinator:
    """
    Coordinates multiple eVTOL agents using thermodynamic principles.

    Treats the fleet as a thermodynamic system where agents interact through
    energy-based potentials. The system naturally converges to low-energy
    configurations that avoid collisions and optimize global objectives.

    Inspired by THRML's block Gibbs sampling for probabilistic graphical models.
    """

    def __init__(
        self,
        coordination_radius: float = 100.0,  # meters
        beta: float = 1.0,  # Inverse temperature
        update_strategy: str = "block_gibbs",  # or "mean_field"
    ):
        """
        Initialize thermodynamic multi-agent coordinator.

        Args:
            coordination_radius: Radius within which agents coordinate
            beta: Inverse temperature for Boltzmann distribution
            update_strategy: Coordination strategy ('block_gibbs' or 'mean_field')
        """
        self.coordination_radius = coordination_radius
        self.beta = beta
        self.update_strategy = update_strategy

        # Energy weights for fleet coordination
        self.weights = {
            'separation': 5.0,     # Maintain minimum separation
            'goal_progress': 2.0,  # Make progress toward goals
            'energy_balance': 1.0, # Balance battery usage across fleet
            'traffic_flow': 1.5,   # Smooth traffic patterns
            'congestion': 3.0,     # Avoid congested areas
        }

    def compute_pairwise_interaction_energy(
        self,
        agent1: AgentState,
        agent2: AgentState,
    ) -> float:
        """
        Compute interaction energy between two agents.

        Lower energy = better configuration.

        Args:
            agent1: First agent state
            agent2: Second agent state

        Returns:
            Interaction energy (scalar)
        """
        energy = 0.0

        # Distance between agents
        distance = np.linalg.norm(agent1.position - agent2.position)

        # 1. Separation energy (repulsive at short range)
        min_separation = 30.0  # meters
        if distance < min_separation * 2:
            # Inverse square repulsion
            energy += self.weights['separation'] * (min_separation / max(distance, 1.0)) ** 2

        # 2. Traffic flow energy (prefer aligned velocities if moving in same direction)
        velocity_alignment = np.dot(agent1.velocity, agent2.velocity)
        direction_alignment = np.dot(
            agent1.goal - agent1.position,
            agent2.goal - agent2.position
        )

        if direction_alignment > 0 and distance < self.coordination_radius:
            # Agents going in similar directions should align their velocities
            energy -= self.weights['traffic_flow'] * 0.01 * velocity_alignment

        return energy

    def compute_agent_energy(
        self,
        agent: AgentState,
        all_agents: List[AgentState],
        obstacles: List[Dict],
    ) -> float:
        """
        Compute total energy of an agent given fleet configuration.

        Args:
            agent: Agent to compute energy for
            all_agents: All agents in fleet
            obstacles: List of obstacles

        Returns:
            Total agent energy
        """
        energy = 0.0

        # 1. Goal progress energy
        distance_to_goal = np.linalg.norm(agent.position - agent.goal)
        energy += self.weights['goal_progress'] * distance_to_goal

        # 2. Interaction with other agents
        for other in all_agents:
            if other.agent_id != agent.agent_id:
                energy += self.compute_pairwise_interaction_energy(agent, other)

        # 3. Obstacle avoidance
        for obs in obstacles:
            obs_pos = np.array(obs['position'])
            obs_radius = obs.get('radius', 10.0)
            distance = np.linalg.norm(agent.position - obs_pos)

            if distance < obs_radius * 3:
                energy += 10.0 * np.exp(-(distance - obs_radius))

        # 4. Battery consideration (penalize low battery)
        if agent.battery < 20:
            energy += self.weights['energy_balance'] * (20 - agent.battery)

        return energy

    def compute_fleet_energy(
        self,
        agents: List[AgentState],
        obstacles: List[Dict],
    ) -> float:
        """
        Compute total energy of entire fleet (global state).

        Args:
            agents: List of all agent states
            obstacles: List of obstacles

        Returns:
            Total fleet energy
        """
        total_energy = 0.0

        # Individual agent energies
        for agent in agents:
            total_energy += self.compute_agent_energy(agent, agents, obstacles)

        # Pairwise interactions (avoid double counting)
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                total_energy += self.compute_pairwise_interaction_energy(agent1, agent2)

        # Fleet-level congestion penalty
        total_energy += self._compute_congestion_energy(agents)

        return total_energy

    def _compute_congestion_energy(
        self,
        agents: List[AgentState],
        grid_size: float = 50.0,
    ) -> float:
        """
        Compute energy penalty for congested regions.

        Divides space into grid cells and penalizes cells with many agents.

        Args:
            agents: List of agent states
            grid_size: Size of grid cells (meters)

        Returns:
            Congestion energy
        """
        # Count agents in each grid cell
        cell_counts = {}

        for agent in agents:
            cell_x = int(agent.position[0] / grid_size)
            cell_y = int(agent.position[1] / grid_size)
            cell_z = int(agent.position[2] / grid_size)
            cell = (cell_x, cell_y, cell_z)

            cell_counts[cell] = cell_counts.get(cell, 0) + 1

        # Quadratic penalty for congestion
        congestion_energy = 0.0
        for count in cell_counts.values():
            if count > 3:  # More than 3 agents in same cell
                congestion_energy += self.weights['congestion'] * (count - 3) ** 2

        return congestion_energy

    def find_coordination_groups(
        self,
        agents: List[AgentState],
    ) -> List[Set[str]]:
        """
        Partition agents into coordination groups based on proximity.

        Agents in the same group need to coordinate their actions.
        This is analogous to block partitioning in block Gibbs sampling.

        Args:
            agents: List of agent states

        Returns:
            List of sets, each containing agent IDs in a coordination group
        """
        # Build proximity graph
        n = len(agents)
        adjacency = np.zeros((n, n), dtype=bool)

        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    distance = np.linalg.norm(agent1.position - agent2.position)
                    if distance < self.coordination_radius:
                        adjacency[i, j] = True

        # Find connected components (coordination groups)
        visited = set()
        groups = []

        def dfs(node, group):
            visited.add(node)
            group.add(agents[node].agent_id)
            for neighbor in range(n):
                if adjacency[node, neighbor] and neighbor not in visited:
                    dfs(neighbor, group)

        for i in range(n):
            if i not in visited:
                group = set()
                dfs(i, group)
                groups.append(group)

        return groups

    def coordinate_block_gibbs(
        self,
        agents: List[AgentState],
        obstacles: List[Dict],
        n_iterations: int = 10,
    ) -> List[AgentState]:
        """
        Coordinate agents using block Gibbs sampling.

        Updates coordination groups iteratively, sampling from Boltzmann distribution.
        This is inspired by THRML's block Gibbs sampling for PGMs.

        Args:
            agents: Current agent states
            obstacles: List of obstacles
            n_iterations: Number of sampling iterations

        Returns:
            Updated agent states with coordinated velocities
        """
        # Find coordination groups
        groups = self.find_coordination_groups(agents)

        # Create agent lookup
        agent_dict = {a.agent_id: a for a in agents}

        for iteration in range(n_iterations):
            # Update each group
            for group in groups:
                # Sample velocity adjustments for agents in this group
                for agent_id in group:
                    agent = agent_dict[agent_id]

                    # Propose velocity changes
                    candidates = []
                    energies = []

                    for _ in range(5):  # Sample 5 candidates
                        # Perturb velocity
                        delta_v = np.random.normal(0, 1.0, size=3)
                        new_velocity = agent.velocity + delta_v

                        # Clamp velocity
                        speed = np.linalg.norm(new_velocity)
                        max_speed = 20.0  # m/s
                        if speed > max_speed:
                            new_velocity = new_velocity * (max_speed / speed)

                        # Create temporary updated agent
                        temp_agent = AgentState(
                            agent_id=agent.agent_id,
                            position=agent.position + new_velocity * 1.0,  # 1 second lookahead
                            velocity=new_velocity,
                            goal=agent.goal,
                            battery=agent.battery,
                            status=agent.status,
                        )

                        # Compute energy
                        energy = self.compute_agent_energy(
                            temp_agent,
                            [agent_dict[aid] for aid in agent_dict],
                            obstacles
                        )

                        candidates.append(new_velocity)
                        energies.append(energy)

                    # Sample from Boltzmann distribution (with numerical stability)
                    energies = np.array(energies)
                    
                    # Use log-sum-exp trick for numerical stability
                    # Subtract max energy to prevent overflow/underflow
                    energies_shifted = -self.beta * (energies - np.min(energies))
                    probabilities = np.exp(energies_shifted)
                    
                    # Normalize probabilities with safety check
                    prob_sum = probabilities.sum()
                    if prob_sum > 0 and not np.isnan(prob_sum):
                        probabilities /= prob_sum
                    else:
                        # Fallback to uniform distribution if numerical issues
                        probabilities = np.ones(len(candidates)) / len(candidates)

                    # Select velocity
                    selected_idx = np.random.choice(len(candidates), p=probabilities)
                    agent.velocity = candidates[selected_idx]

        return agents

    def coordinate_mean_field(
        self,
        agents: List[AgentState],
        obstacles: List[Dict],
    ) -> List[AgentState]:
        """
        Coordinate agents using mean-field approximation.

        Each agent updates based on average effect of all other agents.
        Faster but less accurate than block Gibbs.

        Args:
            agents: Current agent states
            obstacles: List of obstacles

        Returns:
            Updated agent states
        """
        for agent in agents:
            # Compute mean field from other agents
            mean_repulsion = np.zeros(3)
            count = 0

            for other in agents:
                if other.agent_id != agent.agent_id:
                    distance = np.linalg.norm(agent.position - other.position)
                    if distance < self.coordination_radius and distance > 0:
                        # Repulsive force
                        direction = (agent.position - other.position) / distance
                        force = direction * (self.coordination_radius / distance)
                        mean_repulsion += force
                        count += 1

            if count > 0:
                mean_repulsion /= count

                # Adjust velocity
                agent.velocity += mean_repulsion * 0.1

                # Clamp velocity
                speed = np.linalg.norm(agent.velocity)
                max_speed = 20.0
                if speed > max_speed:
                    agent.velocity = agent.velocity * (max_speed / speed)

        return agents

    def coordinate_fleet(
        self,
        agents: List[AgentState],
        obstacles: List[Dict],
    ) -> Tuple[List[AgentState], Dict]:
        """
        Main coordination function using configured strategy.

        Args:
            agents: Current agent states
            obstacles: List of obstacles

        Returns:
            (coordinated_agents, metadata)
        """
        initial_energy = self.compute_fleet_energy(agents, obstacles)

        if self.update_strategy == "block_gibbs":
            coordinated_agents = self.coordinate_block_gibbs(agents, obstacles)
        elif self.update_strategy == "mean_field":
            coordinated_agents = self.coordinate_mean_field(agents, obstacles)
        else:
            coordinated_agents = agents

        final_energy = self.compute_fleet_energy(coordinated_agents, obstacles)

        metadata = {
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_reduction': initial_energy - final_energy,
            'strategy': self.update_strategy,
            'n_agents': len(agents),
            'thermodynamic': True,
        }

        return coordinated_agents, metadata
