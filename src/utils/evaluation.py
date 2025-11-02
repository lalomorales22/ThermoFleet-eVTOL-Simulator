"""
Evaluation utilities for trained eVTOL agents.
Provides comprehensive testing and metrics.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class EVTOLEvaluator:
    """
    Comprehensive evaluator for eVTOL agents.

    Metrics:
    - Success rate (goal reached)
    - Average reward
    - Collision rate
    - Altitude compliance
    - Energy efficiency
    - Flight smoothness
    - Scenario-specific performance
    """

    def __init__(
        self,
        env,
        agent,
        n_eval_episodes: int = 100,
        save_trajectories: bool = True,
        output_dir: str = "./eval_results",
    ):
        """
        Initialize evaluator.

        Args:
            env: Evaluation environment
            agent: Trained agent (with .predict() method)
            n_eval_episodes: Number of evaluation episodes
            save_trajectories: Whether to save trajectory data
            output_dir: Directory to save results
        """
        self.env = env
        self.agent = agent
        self.n_eval_episodes = n_eval_episodes
        self.save_trajectories = save_trajectories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.episode_data = []
        self.trajectories = []

        logger.info(f"Initialized EVTOLEvaluator with {n_eval_episodes} episodes")

    def evaluate(self, deterministic: bool = True, render: bool = False) -> Dict[str, Any]:
        """
        Run full evaluation.

        Args:
            deterministic: Use deterministic policy
            render: Render episodes

        Returns:
            Comprehensive metrics dictionary
        """
        logger.info(f"Starting evaluation over {self.n_eval_episodes} episodes...")

        for episode in range(self.n_eval_episodes):
            episode_metrics = self._run_episode(
                episode_num=episode,
                deterministic=deterministic,
                render=render,
            )
            self.episode_data.append(episode_metrics)

            if (episode + 1) % 10 == 0:
                logger.info(f"Completed {episode + 1}/{self.n_eval_episodes} episodes")

        # Compute aggregate metrics
        results = self._compute_metrics()

        # Save results
        self._save_results(results)

        # Generate plots
        self._generate_plots()

        logger.info("Evaluation complete!")
        return results

    def _run_episode(
        self,
        episode_num: int,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        obs, info = self.env.reset()

        trajectory = {
            "positions": [],
            "velocities": [],
            "actions": [],
            "rewards": [],
        }

        episode_reward = 0.0
        episode_length = 0
        collisions = 0
        altitude_violations = 0
        done = False

        while not done and episode_length < 2000:
            # Get action
            action, _ = self.agent.predict(obs, deterministic=deterministic)

            # Store trajectory
            if self.save_trajectories:
                trajectory["positions"].append(info.get("position", np.zeros(3)))
                trajectory["velocities"].append(info.get("velocity", np.zeros(3)))
                trajectory["actions"].append(action)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if self.save_trajectories:
                trajectory["rewards"].append(reward)

            # Track metrics
            altitude = info.get("position", [0, 0, 0])[2]
            if altitude < 400.0 or altitude > 500.0:
                altitude_violations += 1

            if info.get("collision_count", 0) > collisions:
                collisions = info.get("collision_count", 0)

            if render:
                self.env.render()

        # Episode summary
        episode_metrics = {
            "episode": episode_num,
            "reward": episode_reward,
            "length": episode_length,
            "success": info.get("goal_reached", False),
            "collisions": collisions,
            "altitude_violations": altitude_violations,
            "final_battery": info.get("battery_level", 0.0),
            "distance_to_goal": info.get("distance_to_goal", float('inf')),
        }

        if self.save_trajectories:
            self.trajectories.append(trajectory)

        return episode_metrics

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics from episode data."""
        df = pd.DataFrame(self.episode_data)

        metrics = {
            # Success metrics
            "success_rate": df["success"].mean(),
            "collision_rate": (df["collisions"] > 0).mean(),

            # Reward metrics
            "mean_reward": df["reward"].mean(),
            "std_reward": df["reward"].std(),
            "min_reward": df["reward"].min(),
            "max_reward": df["reward"].max(),

            # Episode length
            "mean_length": df["length"].mean(),
            "std_length": df["length"].std(),

            # Altitude compliance
            "mean_altitude_violations": df["altitude_violations"].mean(),
            "altitude_compliance_rate": (df["altitude_violations"] == 0).mean(),

            # Energy efficiency
            "mean_final_battery": df["final_battery"].mean(),
            "battery_depletion_rate": (df["final_battery"] == 0.0).mean(),

            # Distance to goal (for unsuccessful episodes)
            "mean_distance_to_goal": df[~df["success"]]["distance_to_goal"].mean()
            if len(df[~df["success"]]) > 0 else 0.0,

            # Quartile statistics
            "reward_q25": df["reward"].quantile(0.25),
            "reward_median": df["reward"].median(),
            "reward_q75": df["reward"].quantile(0.75),
        }

        return metrics

    def _save_results(self, metrics: Dict[str, Any]):
        """Save evaluation results to file."""
        # Save metrics as JSON
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")

        # Save episode data as CSV
        df = pd.DataFrame(self.episode_data)
        episodes_file = self.output_dir / "episodes.csv"
        df.to_csv(episodes_file, index=False)
        logger.info(f"Saved episode data to {episodes_file}")

        # Save trajectories (if enabled)
        if self.save_trajectories:
            trajectories_file = self.output_dir / "trajectories.npz"
            np.savez_compressed(trajectories_file, trajectories=self.trajectories)
            logger.info(f"Saved trajectories to {trajectories_file}")

    def _generate_plots(self):
        """Generate evaluation plots."""
        df = pd.DataFrame(self.episode_data)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Reward distribution
        axes[0, 0].hist(df["reward"], bins=30, edgecolor="black")
        axes[0, 0].set_title("Reward Distribution")
        axes[0, 0].set_xlabel("Reward")
        axes[0, 0].set_ylabel("Frequency")

        # Episode length distribution
        axes[0, 1].hist(df["length"], bins=30, edgecolor="black")
        axes[0, 1].set_title("Episode Length Distribution")
        axes[0, 1].set_xlabel("Length")
        axes[0, 1].set_ylabel("Frequency")

        # Success rate over episodes
        window = 10
        rolling_success = df["success"].rolling(window).mean()
        axes[0, 2].plot(rolling_success)
        axes[0, 2].set_title(f"Success Rate (Rolling {window})")
        axes[0, 2].set_xlabel("Episode")
        axes[0, 2].set_ylabel("Success Rate")

        # Collision count
        axes[1, 0].hist(df["collisions"], bins=range(int(df["collisions"].max()) + 2), edgecolor="black")
        axes[1, 0].set_title("Collision Distribution")
        axes[1, 0].set_xlabel("Collisions")
        axes[1, 0].set_ylabel("Frequency")

        # Battery levels
        axes[1, 1].hist(df["final_battery"], bins=30, edgecolor="black")
        axes[1, 1].set_title("Final Battery Level Distribution")
        axes[1, 1].set_xlabel("Battery Level")
        axes[1, 1].set_ylabel("Frequency")

        # Altitude violations
        axes[1, 2].hist(df["altitude_violations"], bins=30, edgecolor="black")
        axes[1, 2].set_title("Altitude Violations")
        axes[1, 2].set_xlabel("Violations")
        axes[1, 2].set_ylabel("Frequency")

        plt.tight_layout()
        plot_file = self.output_dir / "evaluation_plots.png"
        plt.savefig(plot_file, dpi=150)
        logger.info(f"Saved plots to {plot_file}")
        plt.close()

        # 3D trajectory plot (if trajectories saved)
        if self.save_trajectories and len(self.trajectories) > 0:
            self._plot_trajectories()

    def _plot_trajectories(self, n_traj: int = 5):
        """Plot 3D trajectories."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot first n_traj trajectories
        for i in range(min(n_traj, len(self.trajectories))):
            traj = self.trajectories[i]
            positions = np.array(traj["positions"])

            if len(positions) > 0:
                ax.plot(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    alpha=0.6,
                    label=f"Episode {i}",
                )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Sample Trajectories")
        ax.legend()

        # Add altitude bounds
        ax.axhline(y=400, color='r', linestyle='--', alpha=0.3, label='Min Altitude')
        ax.axhline(y=500, color='r', linestyle='--', alpha=0.3, label='Max Altitude')

        traj_file = self.output_dir / "trajectories_3d.png"
        plt.savefig(traj_file, dpi=150)
        logger.info(f"Saved 3D trajectory plot to {traj_file}")
        plt.close()

    def print_summary(self):
        """Print evaluation summary."""
        if not self.episode_data:
            logger.warning("No evaluation data available")
            return

        metrics = self._compute_metrics()

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nEpisodes: {self.n_eval_episodes}")
        print(f"\nSuccess Metrics:")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Collision Rate: {metrics['collision_rate']:.2%}")
        print(f"  Altitude Compliance: {metrics['altitude_compliance_rate']:.2%}")
        print(f"\nReward Statistics:")
        print(f"  Mean: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Min: {metrics['min_reward']:.2f}")
        print(f"  Max: {metrics['max_reward']:.2f}")
        print(f"  Median: {metrics['reward_median']:.2f}")
        print(f"\nEpisode Length:")
        print(f"  Mean: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
        print(f"\nEnergy Efficiency:")
        print(f"  Mean Final Battery: {metrics['mean_final_battery']:.2%}")
        print(f"  Battery Depletion Rate: {metrics['battery_depletion_rate']:.2%}")
        print("\n" + "=" * 60)


def compare_models(
    env,
    models: Dict[str, Any],
    n_episodes: int = 50,
    output_dir: str = "./model_comparison",
) -> pd.DataFrame:
    """
    Compare multiple trained models.

    Args:
        env: Evaluation environment
        models: Dictionary of {model_name: model}
        n_episodes: Episodes per model
        output_dir: Output directory

    Returns:
        DataFrame with comparison results
    """
    results = []

    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")

        evaluator = EVTOLEvaluator(
            env=env,
            agent=model,
            n_eval_episodes=n_episodes,
            save_trajectories=False,
            output_dir=f"{output_dir}/{name}",
        )

        metrics = evaluator.evaluate(deterministic=True)
        metrics["model_name"] = name
        results.append(metrics)

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Save comparison
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_file = output_path / "model_comparison.csv"
    df.to_csv(comparison_file, index=False)

    logger.info(f"Saved model comparison to {comparison_file}")

    return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    from ..environments.evtol_gym_env import EVTOLEnv

    # Create environment
    env = EVTOLEnv(vehicle_type="medium")

    # Create dummy agent for testing
    class RandomAgent:
        def predict(self, obs, deterministic=False):
            return env.action_space.sample(), None

    agent = RandomAgent()

    # Evaluate
    evaluator = EVTOLEvaluator(env, agent, n_eval_episodes=10)
    results = evaluator.evaluate()
    evaluator.print_summary()
