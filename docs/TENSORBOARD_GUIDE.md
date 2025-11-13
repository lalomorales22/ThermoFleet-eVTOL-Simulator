# TensorBoard Usage Guide

## ✅ TensorBoard is Working!

The warning you see is **normal and safe to ignore**:
```
TensorFlow installation not found - running with reduced feature set.
```

This just means you don't have TensorFlow installed (which is fine - we use PyTorch). TensorBoard works perfectly without it!

## 📊 Viewing Your Training Logs

### 1. Start TensorBoard

```bash
tensorboard --logdir=./logs
```

Then open your browser to: **http://localhost:6006/**

### 2. What You'll See

Your training logs are in `logs/PPO_1/` and contain:

- **Scalars**: Training metrics (rewards, losses, etc.)
- **Time Series**: Performance over training steps
- **Distributions**: Policy/value network statistics

### 3. Available Metrics

From your recent training run, you should see:

- **rollout/ep_rew_mean**: Average episode reward
- **rollout/ep_len_mean**: Average episode length
- **train/learning_rate**: Current learning rate
- **train/policy_loss**: Policy network loss
- **train/value_loss**: Value network loss
- **train/entropy_loss**: Entropy (exploration) loss

### 4. Tips for Better Visualization

#### Multiple Runs Comparison
```bash
# Create different log directories for different experiments
python train.py --algo=PPO --log-dir=./logs/ppo_baseline --total-timesteps=100000
python train.py --algo=PPO --thermodynamic --log-dir=./logs/ppo_thermodynamic --total-timesteps=100000

# View both
tensorboard --logdir=./logs
```

#### Expose to Network
```bash
# Access from other machines
tensorboard --logdir=./logs --bind_all
```

#### Custom Port
```bash
# Use different port
tensorboard --logdir=./logs --port=6007
```

## 🔥 Thermodynamic Training Visualization

When training with thermodynamic features:

```bash
# Train with thermodynamic decision making
python train.py --algo=PPO --thermodynamic --beta=2.0 --total-timesteps=500000

# View results
tensorboard --logdir=./logs
```

**What to Look For**:
- Energy efficiency improvements (fewer steps to goal)
- Smoother learning curves (lower variance)
- Better collision avoidance (higher rewards)

## 📁 Log File Structure

```
logs/
├── PPO_1/                          # Run 1 (your recent training)
│   └── events.out.tfevents.*       # TensorBoard event file
├── PPO_2/                          # Run 2 (next training)
│   └── events.out.tfevents.*
├── env_0.monitor.csv               # Environment 0 episodes
├── env_1.monitor.csv               # Environment 1 episodes
└── evaluations.npz                 # Evaluation results
```

## 🧹 Cleaning Up Old Logs

```bash
# Remove all logs
rm -rf logs/

# Keep logs but start fresh numbering
mkdir -p logs
```

## 📊 Current Training Run

Your recent training run created logs at:
- **Directory**: `logs/PPO_1/`
- **Event file**: `events.out.tfevents.1762327069.minibrain.23541.0`
- **Training time**: ~30 seconds
- **Total steps**: 110,000

These logs are ready to view in TensorBoard!

## 🐛 Troubleshooting

### "No dashboards are active"

This means no data yet. Solutions:
1. Run a training: `python train.py --algo=PPO --total-timesteps=10000`
2. Wait a few seconds for logs to be written
3. Refresh TensorBoard

### "TensorFlow installation not found"

**This is normal!** TensorBoard works fine without TensorFlow. Just ignore this warning.

### Port already in use

```bash
# Kill existing TensorBoard
pkill -f tensorboard

# Or use different port
tensorboard --logdir=./logs --port=6007
```

## 🎯 Best Practices

1. **Use descriptive log directories** for experiments:
   ```bash
   --log-dir=./logs/ppo_lr_1e-4_gamma_0.99
   ```

2. **Keep logs organized** by date or experiment:
   ```bash
   --log-dir=./logs/2025-11-04/ppo_thermodynamic
   ```

3. **Monitor during training** - Start TensorBoard before training to watch in real-time

4. **Save important runs** - Copy good training logs to a separate folder

## 🚀 Example Workflow

```bash
# Terminal 1: Start TensorBoard
tensorboard --logdir=./logs

# Terminal 2: Run training
python train.py --algo=PPO --thermodynamic --beta=2.0 --total-timesteps=1000000

# Browser: Open http://localhost:6006/
# Watch training progress in real-time!
```

---

**Your logs are ready to view!** Just open http://localhost:6006/ in your browser. 📊
