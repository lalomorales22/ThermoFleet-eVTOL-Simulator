# WandB Permission Error Fix

**Date**: November 13, 2025  
**Issue**: WandB 403 Permission Error  
**Status**: ✅ FIXED - Training now continues even if WandB fails

---

## Problem

When running training with `--use-wandb`, you encountered:

```
Error uploading run: returned error 403: permission denied
```

Previously, this would **crash the entire training run**. Now it gracefully falls back to training without WandB.

---

## What Was Fixed

### Code Changes

Updated both trainers to catch WandB initialization errors:

**Files Modified:**
- `src/training/ppo_trainer.py`
- `src/training/ddpg_trainer.py`

**Before:**
```python
try:
    wandb.init(**init_config)
except ImportError:
    logger.warning("wandb not installed. Disabling wandb.")
    self.use_wandb = False
```

**After:**
```python
try:
    wandb.init(**init_config)
except ImportError:
    logger.warning("wandb not installed. Disabling wandb.")
    self.use_wandb = False
except Exception as e:
    logger.error(f"Failed to initialize WandB: {e}")
    logger.warning("Continuing training without WandB logging...")
    self.use_wandb = False
```

Now **any** WandB error (not just ImportError) will be caught, and training will continue.

---

## How to Use WandB Successfully

### Option 1: Set Your WandB Entity

Add this to your `.env` file:

```bash
WANDB_ENTITY=lalopenguin  # Your WandB username
WANDB_API_KEY=your_api_key_here
```

Then train:
```bash
python train.py --algo=PPO --use-wandb
```

---

### Option 2: Use Your Own Project Name

Create a project on WandB first, then:

```bash
python train.py --algo=PPO --use-wandb --wandb-project=my-evtol-project
```

---

### Option 3: Train Without WandB (Use TensorBoard)

Simply **omit** the `--use-wandb` flag:

```bash
# Train without WandB
python train.py --algo=PPO --total-timesteps=10000

# View with TensorBoard
tensorboard --logdir=./logs
# Open: http://localhost:6006
```

**TensorBoard works great and requires no setup!**

---

### Option 4: Let It Fail Gracefully

Just run with `--use-wandb` - if it fails, training continues anyway:

```bash
python train.py --algo=PPO --use-wandb

# You'll see:
# [ERROR] Failed to initialize WandB: permission denied
# [WARNING] Continuing training without WandB logging...
# [INFO] Starting training for 1000 timesteps
# ... training continues normally ...
```

---

## Why Does the 403 Error Happen?

The 403 "permission denied" error typically occurs when:

1. **Project doesn't exist** - The project `thermofleet-evtol-simulator` doesn't exist in your account
2. **Wrong entity** - You're trying to access a team/organization you don't belong to
3. **Project is private** - The project belongs to someone else
4. **API key issues** - Your API key doesn't have write permissions

---

## Verification

The fix was tested and works correctly:

```bash
$ python train.py --algo=PPO --use-wandb --total-timesteps=2000 --n-envs=4

# Output:
[ERROR] Failed to initialize WandB: Error uploading run: returned error 403
[WARNING] Continuing training without WandB logging...
[INFO] Starting training for 2000 timesteps
...
[INFO] Training completed successfully!
[INFO] Final performance: 10.91 ± 135.51

✅ SUCCESS - Training completed despite WandB error!
```

---

## Recommended Approach

### For Development/Testing
Use **TensorBoard** (no setup required):
```bash
python train.py --algo=PPO --total-timesteps=100000
tensorboard --logdir=./logs
```

### For Experiment Tracking
Use **WandB** with your own project:
```bash
# 1. Create project on wandb.ai
# 2. Add to .env:
WANDB_ENTITY=your_username

# 3. Train
python train.py --algo=PPO --use-wandb --wandb-project=your-project
```

### For Production
Use **both**:
```bash
# Get local logs AND cloud tracking
python train.py --algo=PPO --use-wandb --total-timesteps=1000000

# TensorBoard for real-time monitoring
tensorboard --logdir=./logs &

# WandB for experiment comparison and team sharing
# View at: https://wandb.ai/your-username/your-project
```

---

## Database Logging Still Works!

Even without WandB, you still get:
- ✅ **SQLite/MySQL database logging** - All episode data saved
- ✅ **TensorBoard logs** - Real-time training metrics
- ✅ **Model checkpoints** - Saved in `./models/`
- ✅ **Evaluation metrics** - Logged to console and files

So you're not missing out on data!

---

## Summary

**Before**: WandB errors crashed the entire training run ❌

**After**: WandB errors are caught, training continues successfully ✅

**You can now:**
1. Train without worrying about WandB permissions
2. Use TensorBoard as a reliable alternative
3. Fix WandB setup at your convenience
4. Still get all your training data logged

**No more crashes!** 🎉

---

## Documentation Updated

- ✅ `GETTING_STARTED.md` - Added WandB troubleshooting section
- ✅ `src/training/ppo_trainer.py` - Graceful WandB error handling
- ✅ `src/training/ddpg_trainer.py` - Graceful WandB error handling
- ✅ `docs/WANDB_FIX.md` - This document

---

**Next Steps:**
1. Continue training (WandB is optional now!)
2. Use TensorBoard for visualization: `tensorboard --logdir=./logs`
3. Fix WandB permissions when convenient (see options above)
4. Enjoy uninterrupted training! 🚁✨

