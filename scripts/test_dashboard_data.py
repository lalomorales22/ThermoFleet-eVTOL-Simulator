#!/usr/bin/env python3
"""
Test script to populate database with sample data for dashboard testing.
Run this to quickly populate the database for dashboard visualization.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_logger import DatabaseLogger
import numpy as np
from datetime import datetime
import time

def generate_test_data():
    """Generate test training data for dashboard."""
    
    print("=" * 60)
    print("Generating Test Data for Dashboard")
    print("=" * 60)
    
    # Initialize database logger
    try:
        db_logger = DatabaseLogger()
        print("✅ Database logger initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return
    
    # Start training run
    try:
        run_id = db_logger.start_training_run(
            name="Test_Run_PPO_Medium",
            algorithm="PPO",
            hyperparameters={
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "batch_size": 64,
                "n_steps": 2048
            },
            description="Test data for dashboard validation"
        )
        print(f"✅ Started training run (ID: {run_id})")
    except Exception as e:
        print(f"❌ Failed to start training run: {e}")
        return
    
    # Generate 20 episodes
    n_episodes = 20
    vehicle_types = ['small', 'medium', 'large']
    
    print(f"\n🚁 Generating {n_episodes} episodes...")
    
    for episode_num in range(1, n_episodes + 1):
        vehicle_type = vehicle_types[episode_num % 3]
        
        try:
            # Start episode
            episode_id = db_logger.start_episode(
                episode_number=episode_num,
                vehicle_type=vehicle_type,
                arena_name="NYC_Manhattan",
                num_agents=1,
                algorithm="PPO",
                model_version="v1.0"
            )
            
            # Generate trajectory (100-500 timesteps)
            n_steps = np.random.randint(100, 500)
            
            # Starting position
            position = np.array([
                np.random.uniform(-500, 500),
                np.random.uniform(-500, 500),
                np.random.uniform(120, 150)  # Altitude in meters (~400-500 ft)
            ])
            
            # Goal position
            goal = np.array([
                np.random.uniform(-500, 500),
                np.random.uniform(-500, 500),
                np.random.uniform(120, 150)
            ])
            
            total_reward = 0
            collision_count = 0
            altitude_violations = 0
            battery = 50.0  # kWh
            
            # Generate timesteps
            for step in range(n_steps):
                # Move towards goal with some noise
                direction = goal - position
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                
                # Update position
                position += direction * 5.0 + np.random.randn(3) * 2.0
                
                # Calculate velocity
                velocity = np.linalg.norm(direction * 5.0)
                
                # Altitude in feet
                altitude_ft = position[2] * 3.28084
                
                # Check altitude compliance (400-500 ft)
                altitude_violation = not (400.0 <= altitude_ft <= 500.0)
                if altitude_violation:
                    altitude_violations += 1
                
                # Random collision (5% chance)
                collision = np.random.random() < 0.05
                if collision:
                    collision_count += 1
                
                # Calculate reward
                distance_to_goal = np.linalg.norm(goal - position)
                step_reward = max(0, 10.0 - distance_to_goal / 100.0)
                if altitude_violation:
                    step_reward -= 1.0
                if collision:
                    step_reward -= 5.0
                
                total_reward += step_reward
                
                # Energy consumption
                energy_consumption = np.random.uniform(1.5, 3.5)  # kW
                battery -= energy_consumption * 0.02 / 3600  # Decrease battery
                battery = max(0, battery)
                
                # Log timestep
                db_logger.log_timestep(
                    timestep=step,
                    position=position,
                    velocity=velocity,
                    altitude_ft=altitude_ft,
                    battery_kwh=battery,
                    energy_consumption_kw=energy_consumption,
                    step_reward=step_reward,
                    collision=collision,
                    altitude_violation=altitude_violation
                )
            
            # Determine success
            final_distance = np.linalg.norm(goal - position)
            successful = (
                final_distance < 50.0 and
                collision_count == 0 and
                altitude_violations < 10 and
                battery > 5.0
            )
            
            # End episode
            db_logger.end_episode(
                total_reward=total_reward,
                collision_count=collision_count,
                altitude_violations=altitude_violations,
                successful=successful
            )
            
            status = "✅ SUCCESS" if successful else "❌ FAILED"
            print(f"  Episode {episode_num:2d} ({vehicle_type:6s}): "
                  f"reward={total_reward:6.1f}, "
                  f"steps={n_steps:3d}, "
                  f"collisions={collision_count}, "
                  f"{status}")
            
        except Exception as e:
            print(f"  ❌ Episode {episode_num} failed: {e}")
    
    # End training run
    try:
        db_logger.end_training_run(
            total_episodes=n_episodes,
            best_reward=max([100.0]),  # Mock value
            convergence_episode=15,
            status='completed'
        )
        print(f"\n✅ Training run completed")
    except Exception as e:
        print(f"\n❌ Failed to end training run: {e}")
    
    # Close logger
    db_logger.close()
    
    print("\n" + "=" * 60)
    print("✅ Test Data Generation Complete!")
    print("=" * 60)
    print("\nYou can now:")
    print("  1. Run the dashboard: streamlit run dashboard.py")
    print("  2. Check the Overview tab for episode data")
    print("  3. View Training Monitor for progress")
    print("  4. Explore Thermodynamic Analysis")
    print("=" * 60)


if __name__ == "__main__":
    generate_test_data()

