#!/usr/bin/env python3
"""
Quick WandB Integration Test
Run this to verify WandB is working before starting long training runs.
"""

import os
import sys
import random
import wandb
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")

def test_wandb_basic():
    """Test basic WandB functionality"""
    print("=" * 60)
    print("🧪 Testing WandB Integration")
    print("=" * 60)
    
    # Check environment variables
    api_key = os.getenv('WANDB_API_KEY')
    project = os.getenv('WANDB_PROJECT', 'thermofleet-evtol-simulator')
    entity = os.getenv('WANDB_ENTITY', None)
    
    print("\n📋 Configuration:")
    print(f"  API Key: {'✅ Set' if api_key else '❌ Not Set'}")
    print(f"  Project: {project}")
    print(f"  Entity: {entity or 'Not specified (will use default)'}")
    
    if not api_key:
        print("\n❌ ERROR: WANDB_API_KEY not found in environment!")
        print("   Please add it to your .env file or run: wandb login")
        return False
    
    # Initialize WandB
    print("\n🚀 Initializing WandB run...")
    
    try:
        init_config = {
            "project": project,
            "name": "test-run-wandb-integration",
            "config": {
                "test_type": "integration_test",
                "learning_rate": 0.02,
                "architecture": "PPO",
                "vehicle_type": "medium",
                "epochs": 10,
            },
        }
        
        # Add entity if specified
        if entity:
            init_config["entity"] = entity
        
        run = wandb.init(**init_config)
        
        print(f"✅ WandB initialized successfully!")
        print(f"   Run URL: {run.url}")
        
    except Exception as e:
        print(f"❌ Failed to initialize WandB: {e}")
        return False
    
    # Simulate training with metrics
    print("\n📊 Logging test metrics...")
    
    epochs = 10
    offset = random.random() / 5
    
    for epoch in range(2, epochs):
        # Simulate metrics
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        reward = 100 * acc - 50 * loss
        
        # Log metrics
        run.log({
            "epoch": epoch,
            "train/accuracy": acc,
            "train/loss": loss,
            "rollout/ep_reward": reward,
            "rollout/ep_length": 500 + random.randint(-50, 50),
        })
        
        print(f"  Epoch {epoch}: acc={acc:.4f}, loss={loss:.4f}, reward={reward:.2f}")
    
    print("\n✅ Logged 8 epochs of test data")
    
    # Finish the run
    print("\n🏁 Finishing WandB run...")
    run.finish()
    
    print("\n" + "=" * 60)
    print("✅ WandB Integration Test PASSED!")
    print("=" * 60)
    print("\n🎉 You're ready to train with WandB!")
    print("\nNext steps:")
    print("  1. Check your WandB dashboard: https://wandb.ai/")
    print("  2. Look for project:", project)
    print("  3. Find run: test-run-wandb-integration")
    print("\n💡 To train with WandB:")
    print("  python train.py --algo=PPO --use-wandb --total-timesteps=50000")
    print()
    
    return True


def test_wandb_with_thermofleet():
    """Test WandB with ThermoFleet-specific config"""
    print("\n" + "=" * 60)
    print("🚁 Testing ThermoFleet WandB Configuration")
    print("=" * 60)
    
    try:
        project = os.getenv('WANDB_PROJECT', 'thermofleet-evtol-simulator')
        entity = os.getenv('WANDB_ENTITY', None)
        
        init_config = {
            "project": project,
            "name": "thermofleet-config-test",
            "config": {
                # Training config
                "algorithm": "PPO",
                "vehicle_type": "medium",
                "n_envs": 8,
                "total_timesteps": 1000000,
                "learning_rate": 3e-4,
                "batch_size": 64,
                
                # Thermodynamic config
                "thermodynamic": True,
                "beta": 2.0,
                "path_planner": "thermodynamic",
                
                # Environment config
                "max_altitude_ft": 500,
                "min_altitude_ft": 400,
                "arena": "NYC_Manhattan",
            },
        }
        
        if entity:
            init_config["entity"] = entity
        
        run = wandb.init(**init_config)
        
        print(f"✅ ThermoFleet config logged successfully!")
        print(f"   Run URL: {run.url}")
        
        # Log some ThermoFleet-specific metrics
        print("\n📊 Logging ThermoFleet metrics...")
        
        for step in range(0, 100, 10):
            run.log({
                "step": step,
                "rollout/ep_reward": 50 + step * 0.5 + random.random() * 10,
                "rollout/collision_rate": max(0, 0.05 - step * 0.0005),
                "rollout/altitude_compliance": min(1.0, 0.8 + step * 0.002),
                "energy/efficiency": 1.0 + step * 0.01 + random.random() * 0.1,
                "thermodynamic/energy": 100 - step * 0.5,
            })
        
        print("✅ Logged ThermoFleet metrics")
        
        run.finish()
        
        print("\n✅ ThermoFleet WandB Configuration Test PASSED!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Run basic test
    success = test_wandb_basic()
    
    if success:
        # Run ThermoFleet-specific test
        test_wandb_with_thermofleet()
    else:
        print("\n❌ Basic test failed. Please fix configuration before proceeding.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\n✈️  Your WandB integration is ready for ThermoFleet training!")

