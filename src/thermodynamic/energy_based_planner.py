"""
Energy-Based Path Planning using Thermodynamic Computing

This module implements path planning for eVTOL vehicles using energy-based models
and thermodynamic principles for efficient, low-energy trajectory optimization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import jax
import jax.numpy as jnp


class EnergyBasedPathPlanner:
    """
    Path planner using energy-based models (EBMs) for thermodynamically-efficient
    trajectory planning in 3D urban airspace.

    The planner treats path planning as an energy minimization problem where:
    - Lower energy = more desirable paths
    - Energy considers: distance, obstacles, altitude constraints, wind, battery usage
    - Uses Gibbs sampling for exploring path space efficiently
    """

    def __init__(
        self,
        arena_bounds: Tuple[float, float, float, float, float, float],
        altitude_range: Tuple[float, float] = (400.0, 500.0),
        beta: float = 1.0,  # Inverse temperature for Boltzmann distribution
        grid_resolution: float = 10.0,  # meters
    ):
        """
        Initialize the energy-based path planner.

        Args:
            arena_bounds: (x_min, x_max, y_min, y_max, z_min, z_max) in meters
            altitude_range: (min_altitude, max_altitude) in feet
            beta: Inverse temperature parameter (higher = more deterministic)
            grid_resolution: Spatial discretization resolution in meters
        """
        self.arena_bounds = arena_bounds
        self.altitude_range = altitude_range
        self.beta = beta
        self.grid_resolution = grid_resolution

        # Energy weights for different path components
        self.weights = {
            'distance': 1.0,        # Prefer shorter paths
            'altitude': 0.5,        # Penalty for altitude violations
            'smoothness': 0.3,      # Prefer smooth trajectories
            'obstacles': 10.0,      # Strong penalty for obstacles
            'wind': 0.2,           # Consider wind resistance
            'energy': 0.4,         # Battery energy consumption
        }

        # Cache for obstacle map
        self.obstacle_energy_map: Optional[np.ndarray] = None

    def compute_path_energy(
        self,
        path: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Dict],
        wind_field: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the total energy of a path using thermodynamic principles.

        Lower energy = more desirable path (thermodynamic minimum).

        Args:
            path: Array of shape (n_waypoints, 3) representing the trajectory
            goal: Target position (x, y, z)
            obstacles: List of obstacle dictionaries with 'position' and 'radius'
            wind_field: Optional wind vector field

        Returns:
            Total path energy (scalar)
        """
        energy = 0.0

        # 1. Distance energy (how far from goal)
        final_distance = np.linalg.norm(path[-1] - goal)
        energy += self.weights['distance'] * final_distance

        # 2. Altitude constraint energy
        altitude_ft = path[:, 2] * 3.28084  # Convert m to ft
        altitude_violations = np.sum(
            np.maximum(0, self.altitude_range[0] - altitude_ft) +
            np.maximum(0, altitude_ft - self.altitude_range[1])
        )
        energy += self.weights['altitude'] * altitude_violations

        # 3. Path smoothness energy (penalize sharp turns)
        if len(path) > 2:
            accelerations = np.diff(path, n=2, axis=0)
            smoothness_cost = np.sum(np.linalg.norm(accelerations, axis=1))
            energy += self.weights['smoothness'] * smoothness_cost

        # 4. Obstacle avoidance energy
        for obs in obstacles:
            obs_pos = np.array(obs['position'])
            obs_radius = obs['radius']

            for waypoint in path:
                dist = np.linalg.norm(waypoint - obs_pos)
                # Exponential penalty near obstacles (thermodynamic barrier)
                if dist < obs_radius * 3:
                    energy += self.weights['obstacles'] * np.exp(-dist / obs_radius)

        # 5. Wind resistance energy
        if wind_field is not None:
            wind_cost = 0.0
            for i in range(len(path) - 1):
                segment = path[i+1] - path[i]
                # Simplified wind interpolation
                wind_cost += np.dot(segment, -wind_field)  # Negative because going against wind costs energy
            energy += self.weights['wind'] * max(0, wind_cost)

        # 6. Energy consumption (simplified battery model)
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        energy += self.weights['energy'] * path_length

        return energy

    def sample_path_gibbs(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Dict],
        n_waypoints: int = 10,
        n_samples: int = 100,
        wind_field: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Sample optimal path using Gibbs sampling (thermodynamic sampling).

        This is inspired by THRML's block Gibbs sampling for discrete PGMs.
        We discretize the path into waypoints and sample from the Boltzmann distribution.

        Args:
            start: Starting position (x, y, z)
            goal: Target position (x, y, z)
            obstacles: List of obstacles
            n_waypoints: Number of intermediate waypoints
            n_samples: Number of samples for Monte Carlo estimation
            wind_field: Optional wind field

        Returns:
            (best_path, energy): Best sampled path and its energy
        """
        # Initialize with linear interpolation
        t = np.linspace(0, 1, n_waypoints + 2)
        current_path = np.outer(1 - t, start) + np.outer(t, goal)

        best_path = current_path.copy()
        best_energy = self.compute_path_energy(current_path, goal, obstacles, wind_field)

        # Gibbs sampling: update each waypoint conditioned on others
        rng = np.random.default_rng()

        for _ in range(n_samples):
            # Randomly select a waypoint to update (skip start and end)
            wp_idx = rng.integers(1, n_waypoints + 1)

            # Sample new position from Boltzmann distribution
            # Try multiple candidate positions
            candidates = []
            energies = []

            for _ in range(10):
                # Perturb waypoint
                perturbation = rng.normal(0, self.grid_resolution, size=3)
                new_waypoint = current_path[wp_idx] + perturbation

                # Clamp to arena bounds
                new_waypoint = np.clip(
                    new_waypoint,
                    [self.arena_bounds[0], self.arena_bounds[2], self.arena_bounds[4]],
                    [self.arena_bounds[1], self.arena_bounds[3], self.arena_bounds[5]],
                )

                # Update path
                test_path = current_path.copy()
                test_path[wp_idx] = new_waypoint

                # Compute energy
                energy = self.compute_path_energy(test_path, goal, obstacles, wind_field)
                candidates.append(test_path)
                energies.append(energy)

            # Sample from Boltzmann distribution: P(path) ∝ exp(-beta * energy)
            # Use log-sum-exp trick for numerical stability
            energies = np.array(energies)
            energies_shifted = -self.beta * (energies - np.min(energies))
            probabilities = np.exp(energies_shifted)
            
            # Normalize with safety check
            prob_sum = probabilities.sum()
            if prob_sum > 0 and not np.isnan(prob_sum):
                probabilities /= prob_sum
            else:
                # Fallback to uniform distribution if numerical issues
                probabilities = np.ones(len(candidates)) / len(candidates)

            # Select path according to probability
            selected_idx = rng.choice(len(candidates), p=probabilities)
            current_path = candidates[selected_idx]

            # Track best path
            if energies[selected_idx] < best_energy:
                best_energy = energies[selected_idx]
                best_path = current_path.copy()

        return best_path, best_energy

    def plan_trajectory(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Dict],
        n_waypoints: int = 10,
        method: str = 'gibbs',
    ) -> Dict:
        """
        Plan a trajectory from start to goal using thermodynamic computing.

        Args:
            start: Starting position (x, y, z) in meters
            goal: Goal position (x, y, z) in meters
            obstacles: List of obstacles
            n_waypoints: Number of waypoints
            method: Planning method ('gibbs' or 'greedy')

        Returns:
            Dictionary with 'path', 'energy', and 'metadata'
        """
        if method == 'gibbs':
            path, energy = self.sample_path_gibbs(start, goal, obstacles, n_waypoints)
        else:
            # Fallback to simple linear path
            t = np.linspace(0, 1, n_waypoints + 2)
            path = np.outer(1 - t, start) + np.outer(t, goal)
            energy = self.compute_path_energy(path, goal, obstacles)

        return {
            'path': path,
            'energy': energy,
            'waypoints': path,
            'n_waypoints': len(path),
            'method': method,
            'thermodynamic': True,
        }

    def update_energy_weights(self, new_weights: Dict[str, float]):
        """Update energy function weights dynamically."""
        self.weights.update(new_weights)
