# WandB Setup Guide for ThermoFleet

> **Quick answer**: YES, you have WandB logging built in! Just follow the steps below. ✅

---

## ✅ What You Already Have

Your ThermoFleet simulator already has **complete WandB integration**! Both trainers log:
- 📊 Training metrics (loss, rewards, episode length)
- ⚡ Performance stats (FPS, timesteps)
- 🎯 Algorithm hyperparameters
- 🚁 Vehicle configurations
- 🔥 Thermodynamic computing parameters

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Update Your `.env` File

Open your `.env` file and make sure you have (or add):

```bash
# ===========================
# Experiment Tracking
# ===========================
# Weights & Biases
WANDB_API_KEY=e5dff0811a2b1b733eac3ee3678c9aa958970a94
WANDB_PROJECT=thermofleet-evtol-simulator
WANDB_ENTITY=your_actual_wandb_username  # ⬅️ ADD YOUR USERNAME HERE!

# Enable/disable experiment tracking
ENABLE_WANDB=true
```

**To find your username:**
1. Go to https://wandb.ai/
2. Click your profile icon (top right)
3. Your username is shown there (e.g., `@minibrain` → username is `minibrain`)

### Step 2: Test WandB Integration

Run the test script I just created:

```bash
cd /Users/minibrain/Desktop/ThermoFleet-eVTOL-Simulator
python test_wandb.py
```

**Expected output:**
```
🧪 Testing WandB Integration
============================================================
📋 Configuration:
  API Key: ✅ Set
  Project: thermofleet-evtol-simulator
  Entity: your_username

🚀 Initializing WandB run...
✅ WandB initialized successfully!
   Run URL: https://wandb.ai/your_username/thermofleet-evtol-simulator/runs/xxxxx

📊 Logging test metrics...
  Epoch 2: acc=0.7500, loss=0.2500, reward=50.00
  Epoch 3: acc=0.8750, loss=0.1250, reward=68.75
  ...

✅ WandB Integration Test PASSED!
🎉 ALL TESTS PASSED!
```

### Step 3: Train with WandB

Now you're ready! Run training with WandB enabled:

```bash
python train.py \
  --algo=PPO \
  --vehicle-type=medium \
  --total-timesteps=100000 \
  --use-wandb \
  --device=cpu
```

---

## 📊 What Gets Logged

### Automatically Logged Metrics

#### Training Metrics
- `train/total_timesteps` - Total training steps
- `train/fps` - Frames per second
- `train/learning_rate` - Current learning rate (if scheduled)

#### Episode Metrics (from PPO/DDPG)
- `rollout/ep_reward` - Episode reward
- `rollout/ep_length` - Episode length
- `rollout/ep_rew_mean` - Mean episode reward

#### Policy Metrics (from SB3)
- `train/policy_loss` - Policy network loss
- `train/value_loss` - Value network loss
- `train/entropy_loss` - Entropy bonus
- `train/approx_kl` - KL divergence
- `train/clip_fraction` - Clipped samples fraction

#### Custom Metrics (you can add more!)
- `thermodynamic/energy` - Energy-based planning metrics
- `collision/rate` - Collision statistics
- `altitude/compliance` - Altitude regulation compliance

### Configuration Logged

All hyperparameters are automatically saved:
```python
{
    "algorithm": "PPO",
    "vehicle_type": "medium",
    "n_envs": 8,
    "learning_rate": 0.0003,
    "batch_size": 64,
    "gamma": 0.99,
    # ... and many more!
}
```

---

## 🎯 Usage Examples

### Example 1: Basic Training with WandB
```bash
python train.py \
  --algo=PPO \
  --use-wandb \
  --total-timesteps=50000
```

### Example 2: Custom Project and Run Name
```bash
python train.py \
  --algo=PPO \
  --use-wandb \
  --wandb-project=my-evtol-experiments \
  --wandb-name=ppo_medium_baseline \
  --total-timesteps=100000
```

### Example 3: Thermodynamic Training with WandB
```bash
python train.py \
  --algo=PPO \
  --use-wandb \
  --thermodynamic \
  --beta=2.0 \
  --path-planner=thermodynamic \
  --wandb-name=thermodynamic_v1 \
  --total-timesteps=200000
```

### Example 4: Compare Multiple Algorithms
```bash
# PPO
python train.py --algo=PPO --use-wandb --wandb-name=ppo_baseline --total-timesteps=100000

# DDPG  
python train.py --algo=DDPG --use-wandb --wandb-name=ddpg_baseline --total-timesteps=100000

# TD3
python train.py --algo=TD3 --use-wandb --wandb-name=td3_baseline --total-timesteps=100000
```

Then compare them in WandB dashboard!

---

## 🔍 View Your Results

### In WandB Dashboard

1. Go to https://wandb.ai/
2. Click on your project: `thermofleet-evtol-simulator`
3. See all your runs with:
   - Real-time training curves
   - Hyperparameter comparison
   - System metrics (CPU, RAM, GPU if available)
   - Custom charts and reports

### Key Views to Check

**Training Progress:**
- `rollout/ep_rew_mean` - Is reward increasing?
- `train/policy_loss` - Is it decreasing?
- `train/fps` - How fast is training?

**Performance:**
- Compare multiple runs side-by-side
- Create custom charts
- Share reports with collaborators

---

## 🆚 WandB vs TensorBoard

You can use **BOTH** simultaneously!

### TensorBoard (Local)
```bash
tensorboard --logdir=./logs
# Open: http://localhost:6006
```

### WandB (Cloud)
```bash
# Automatically logs when you use --use-wandb
# View at: https://wandb.ai/
```

**Advantages of WandB over TensorBoard:**
- ☁️ Cloud storage (access anywhere)
- 🤝 Team collaboration
- 📊 Better comparison tools
- 🔔 Slack/email alerts
- 📝 Notes and reports
- 🎨 Custom dashboards

**When to use TensorBoard:**
- 💻 Want local-only logging
- 📴 No internet connection
- 🆓 Don't want cloud account

---

## 🎓 Advanced: Custom Logging

### Add Custom Metrics

If you want to log additional metrics, you can modify the WandB callback:

```python
# In src/training/ppo_trainer.py or ddpg_trainer.py

class WandbCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            metrics = {
                "train/total_timesteps": self.num_timesteps,
                "train/fps": ...,
                
                # Add your custom metrics here!
                "custom/collision_rate": self.locals.get('collision_rate', 0),
                "custom/altitude_violations": self.locals.get('altitude_violations', 0),
                "thermodynamic/energy": self.locals.get('energy', 0),
            }
            
            self.wandb.log(metrics, step=self.num_timesteps)
```

### Log Images/Videos

```python
import wandb

# Log trajectory visualization
wandb.log({
    "trajectory": wandb.Image("path/to/trajectory.png"),
    "video": wandb.Video("path/to/episode.mp4"),
})
```

### Log Histograms

```python
import wandb

# Log weight distributions
wandb.log({
    "weights/policy": wandb.Histogram(policy_weights),
})
```

---

## 🐛 Troubleshooting

### Issue 1: "wandb not installed"
```bash
pip install wandb
```

### Issue 2: "Not logged in to wandb"
```bash
wandb login
# Enter your API key when prompted
```

### Issue 3: "No WANDB_API_KEY found"
Make sure it's in your `.env` file:
```bash
WANDB_API_KEY=your_key_here
```

### Issue 4: Runs not showing up
- Check you're using the right project name
- Check your entity (username) is correct
- Visit: https://wandb.ai/your_username/your_project

### Issue 5: "Failed to initialize WandB"
```bash
# Test your connection
python test_wandb.py

# If that fails, try:
wandb login --relogin
```

---

## 📚 Learn More

### WandB Resources
- [WandB Documentation](https://docs.wandb.ai/)
- [Python API Reference](https://docs.wandb.ai/ref/python)
- [Example Projects](https://wandb.ai/gallery)

### ThermoFleet Guides
- `GETTING_STARTED.md` - Quick start guide
- `TENSORBOARD_GUIDE.md` - TensorBoard setup
- `THERMODYNAMIC_USAGE.md` - Thermodynamic computing

---

## ✅ Quick Checklist

Before your first training run with WandB:

- [ ] Updated `.env` with `WANDB_ENTITY`
- [ ] Ran `wandb login` successfully
- [ ] Ran `python test_wandb.py` - all tests pass
- [ ] Checked WandB dashboard - test runs visible
- [ ] Ready to train with `--use-wandb` flag!

---

## 🎉 You're All Set!

Your ThermoFleet simulator is fully integrated with WandB. Now you can:
- ✅ Track all experiments automatically
- ✅ Compare different algorithms
- ✅ Monitor training in real-time
- ✅ Share results with collaborators
- ✅ Create beautiful reports for papers

**Happy training!** 🚁🔥

---

## 💡 Pro Tips

1. **Organize with Tags**: Add tags to runs for easy filtering
   ```bash
   # In code: wandb.init(tags=["baseline", "thermodynamic", "v1"])
   ```

2. **Use Run Groups**: Group related runs together
   ```bash
   # In code: wandb.init(group="ablation_study")
   ```

3. **Save Best Models**: WandB can store your best model checkpoints

4. **Create Reports**: Combine multiple runs into shareable reports

5. **Set Up Alerts**: Get notified when training completes or fails

---

**Questions?** Check the troubleshooting section above or visit [WandB Support](https://wandb.ai/support)

