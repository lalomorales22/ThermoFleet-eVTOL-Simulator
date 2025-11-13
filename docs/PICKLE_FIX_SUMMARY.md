# Training Pickle Error Fix Summary

**Date**: November 13, 2025  
**Issue**: `TypeError: cannot pickle '_thread.RLock' object`  
**Status**: ✅ RESOLVED

---

## Problem Description

When running training tests with parallel environments (e.g., `python train.py --algo=PPO --n-envs=4`), the simulator encountered a pickle error:

```
TypeError: cannot pickle '_thread.RLock' object
```

This error occurred because:
1. The `DatabaseLogger` was initialized in the trainer's `__init__` method
2. The `DatabaseLogger` contains SQLAlchemy engine and sessionmaker objects with thread locks
3. When using `SubprocVecEnv` for parallel training, Python's multiprocessing module needs to pickle everything to send to subprocesses
4. Thread locks (RLock objects) cannot be pickled, causing the error

---

## Root Cause Analysis

### Files Affected
- `src/training/ppo_trainer.py`
- `src/training/ddpg_trainer.py`
- `train.py`

### The Issue

The `DatabaseLogger` class uses SQLAlchemy for database connections:
- SQLAlchemy's `create_engine()` creates connection pools with thread locks
- `sessionmaker` also creates objects with thread-local state
- These objects are unpickleable

When `SubprocVecEnv` was created, it tried to pickle the entire trainer context (including the database logger) to send to worker processes, causing the pickle error.

---

## Solution Implemented

### Fix 1: Lazy Database Logger Initialization

**Changed**: Initialize the database logger AFTER environment setup, not during trainer initialization.

#### PPO Trainer (`src/training/ppo_trainer.py`)

**Before:**
```python
def __init__(self, ...):
    # ... other initialization ...
    
    # Initialize database logger
    try:
        self.db_logger = DatabaseLogger(enable_sensor_logging=False)
        logger.info("Database logging enabled")
    except Exception as e:
        logger.warning(f"Failed to initialize database logger: {e}")
        self.db_logger = None
```

**After:**
```python
def __init__(self, ...):
    # ... other initialization ...
    
    # Database logger will be initialized later (after env setup)
    # to avoid pickle issues with SubprocVecEnv
    self.db_logger = None

def setup_callbacks(self):
    """Setup training callbacks."""
    callbacks = []
    
    # Initialize database logger now (after env setup, to avoid pickle issues)
    if self.db_logger is None:
        try:
            self.db_logger = DatabaseLogger(enable_sensor_logging=False)
            logger.info("Database logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize database logger: {e}")
            self.db_logger = None
    
    # ... rest of callback setup ...
```

The same changes were applied to `DDPGTrainer` in `src/training/ddpg_trainer.py`.

---

### Fix 2: Proper Resource Cleanup

**Changed**: Set environment references to `None` after closing them in cleanup.

#### Both Trainers

**Before:**
```python
def cleanup(self):
    """Cleanup resources."""
    if self.db_logger is not None:
        self.db_logger.close()
    
    if self.env is not None:
        self.env.close()
    if self.eval_env is not None:
        self.eval_env.close()
```

**After:**
```python
def cleanup(self):
    """Cleanup resources."""
    if self.db_logger is not None:
        try:
            self.db_logger.close()
            logger.info("Database logger closed")
        except Exception as e:
            logger.error(f"Error closing database logger: {e}")
        self.db_logger = None
    
    if self.env is not None:
        try:
            self.env.close()
        except Exception as e:
            logger.error(f"Error closing training env: {e}")
        self.env = None
        
    if self.eval_env is not None:
        try:
            self.eval_env.close()
        except Exception as e:
            logger.error(f"Error closing eval env: {e}")
        self.eval_env = None
```

**Why this matters**: When `evaluate()` checks `if self.eval_env is None`, it needs to know whether to recreate the environment. Setting to `None` after closing ensures proper recreation.

---

### Fix 3: Evaluation Environment Creation

**Changed**: Only create evaluation environment during evaluation, not entire training setup.

#### Both Trainers

**Before:**
```python
def evaluate(self, n_episodes: int = 10, render: bool = False):
    if self.model is None:
        raise ValueError("No model loaded. Train or load a model first.")
    
    if self.eval_env is None:
        self.setup_env()  # This recreates BOTH training and eval envs!
```

**After:**
```python
def evaluate(self, n_episodes: int = 10, render: bool = False):
    if self.model is None:
        raise ValueError("No model loaded. Train or load a model first.")
    
    if self.eval_env is None:
        # Only create eval env, not training env
        self.eval_env = DummyVecEnv([self.make_env(999, self.seed)])
```

**Why this matters**: 
- `setup_env()` would try to create `SubprocVecEnv` for training
- This could trigger pickle errors if the model had unpickleable state
- Evaluation only needs a single environment (using `DummyVecEnv` avoids multiprocessing entirely)

---

### Fix 4: Better Error Handling in train.py

**Changed**: Wrapped cleanup in try-except to prevent cascading errors.

```python
finally:
    # Always cleanup resources
    try:
        trainer.cleanup()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
```

---

## Verification

All training scenarios now work correctly:

### ✅ Test 1: PPO with 4 parallel environments
```bash
python train.py --algo=PPO --vehicle-type=medium --total-timesteps=2000 --n-envs=4 --device=cpu
```
**Result**: Success! No pickle errors, training and evaluation complete.

### ✅ Test 2: TD3 with 4 parallel environments
```bash
python train.py --algo=TD3 --vehicle-type=small --total-timesteps=2000 --n-envs=4 --device=cpu
```
**Result**: Success! No pickle errors, training and evaluation complete.

### ✅ Test 3: PPO with 8 parallel environments
```bash
python train.py --algo=PPO --vehicle-type=large --total-timesteps=5000 --n-envs=8 --device=cpu
```
**Result**: Success! Even with high parallelism, no pickle errors.

---

## Key Takeaways

1. **Pickle-aware design**: When using multiprocessing (`SubprocVecEnv`), ensure all objects that need to be pickled are pickleable
2. **Lazy initialization**: Initialize database connections and other unpickleable resources AFTER environment setup
3. **Resource management**: Always set references to `None` after cleanup to enable proper recreation
4. **Separate concerns**: Use `DummyVecEnv` for evaluation (no multiprocessing needed) and `SubprocVecEnv` only for training

---

## Related Issues

### Minor Issue: Database Unique Constraint
During testing, you may see:
```
ERROR: UNIQUE constraint failed: training_runs.name
```

This is not critical - it occurs when running multiple training sessions with the same configuration. The training continues successfully. To fix, either:
- Use unique `--wandb-name` flags for each run
- Or clear the database between runs: `rm data/database/thermofleet_evtol.db`

---

## Testing Recommendations

Before deploying, test with:
```bash
# Quick test (2-3 minutes)
python train.py --algo=PPO --vehicle-type=medium --total-timesteps=10000 --n-envs=4

# Full test (10-15 minutes)
python train.py --algo=PPO --vehicle-type=medium --total-timesteps=100000 --n-envs=8

# Multi-algorithm test
python train.py --algo=TD3 --vehicle-type=small --total-timesteps=50000 --n-envs=4
python train.py --algo=SAC --vehicle-type=large --total-timesteps=50000 --n-envs=2
```

All should complete without pickle errors.

---

## Future Improvements

1. **Optional multiprocessing**: Add a flag to disable `SubprocVecEnv` and use `DummyVecEnv` for easier debugging
2. **Pickle-safe database logger**: Implement `__getstate__` and `__setstate__` methods to make `DatabaseLogger` properly pickleable
3. **Better error messages**: Add more descriptive error messages when pickle issues occur

---

**Status**: All training tests from `GETTING_STARTED.md` now work correctly! 🎉

