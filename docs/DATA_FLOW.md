# ThermoFleet Data Flow & Connections

> **Quick Answer**: The dashboard connects to **SQLite by default** (or MySQL if configured), NOT WandB.

---

## 🗂️ Data Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    THERMOFLEET SIMULATOR                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────┬─────────────────┬──────────────────┐
                              ▼                 ▼                 ▼                  ▼
                    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                    │   SQLite DB  │  │   MySQL DB   │  │  WandB Cloud │  │  TensorBoard │
                    │   (Local)    │  │  (Optional)  │  │  (Optional)  │  │    (Local)   │
                    └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
                            │                 │                 │                  │
                            └────────┬────────┘                 │                  │
                                     ▼                           ▼                  ▼
                            ┌──────────────┐          ┌──────────────┐  ┌──────────────┐
                            │  DASHBOARD   │          │  WandB Web   │  │ TensorBoard  │
                            │  (Streamlit) │          │     UI       │  │     Web      │
                            └──────────────┘          └──────────────┘  └──────────────┘
```

---

## 📊 Three Separate Data Systems

### 1. **Primary Database: SQLite/MySQL** 
**Used by: Dashboard, Training Scripts, Analysis Tools**

**Location**: 
- SQLite: `data/database/thermofleet_evtol.db` (default)
- MySQL: Remote server (if configured in `.env`)

**What it stores**:
- Episode data (rewards, collisions, violations)
- Detailed metrics (position, velocity, battery, altitude)
- Training runs metadata
- Sensor logs
- Vehicle configurations
- Arena definitions

**Who uses it**:
✅ **Streamlit Dashboard** - Reads from here for all visualizations
✅ Training scripts (via `DatabaseLogger`)
✅ Analysis scripts (`analyze_db.py`)
✅ Replay system metadata

**Configuration**:
```bash
# In .env file
DB_TYPE=sqlite  # or 'mysql'
DB_HOST=localhost
DB_PORT=3306
DB_NAME=thermofleet_evtol
DB_USER=your_user
DB_PASSWORD=your_password
```

---

### 2. **WandB (Weights & Biases)** 
**Used by: Training Scripts ONLY**

**Location**: Cloud (wandb.ai)

**What it stores**:
- Training metrics (loss, rewards, entropy)
- Hyperparameters
- System metrics (CPU, RAM, GPU usage)
- Model checkpoints (optional)
- Custom plots and charts

**Who uses it**:
✅ Training scripts when `--use-wandb` flag is set
❌ **Dashboard does NOT connect to WandB**

**Why separate?**
- WandB is for experiment tracking during training
- Dashboard is for real-time monitoring and analysis
- WandB requires internet; Dashboard works offline
- WandB is optional; Database is required

**Configuration**:
```bash
# Run once to login
wandb login

# Then train with WandB
python train.py --algo=PPO --use-wandb
```

---

### 3. **TensorBoard**
**Used by: Training Scripts, Standalone Viewer**

**Location**: Local `logs/` directory

**What it stores**:
- Training curves (rewards, loss, learning rate)
- Histograms of network weights
- Scalars and distributions
- Custom metrics

**Who uses it**:
✅ Training scripts (automatic logging)
✅ TensorBoard viewer (`tensorboard --logdir=./logs`)
❌ **Dashboard does NOT read TensorBoard logs**

**Why separate?**
- TensorBoard is optimized for deep learning metrics
- Database is better for episode-level data
- Different use cases and visualizations

---

## 🔌 Dashboard Connections - Detailed

### What the Dashboard Actually Does:

```python
# In dashboard.py (line ~95)
if 'db_manager' not in st.session_state:
    try:
        st.session_state.db_manager = DatabaseManager()  # ← Connects to SQLite/MySQL
        st.session_state.db_connected = True
    except Exception as e:
        st.session_state.db_connected = False
```

**The `DatabaseManager` class**:
- Reads `DB_TYPE` from environment (defaults to `sqlite`)
- Connects to either:
  - `data/database/thermofleet_evtol.db` (SQLite)
  - MySQL server (if configured)
- Provides methods like:
  - `query_episodes()` - Get episode data
  - `get_training_progress()` - Get rewards over time
  - `get_vehicle_performance()` - Get stats by vehicle type
  - `get_episode_metrics()` - Get detailed trajectory data

---

## 📍 Where Each Data System Lives

### SQLite Database (Default)
```
ThermoFleet-eVTOL-Simulator/
├── data/
│   └── database/
│       └── thermofleet_evtol.db  ← Dashboard reads from HERE
```

### WandB Data
```
Cloud: https://wandb.ai/<your-username>/thermofleet-evtol-simulator
```

### TensorBoard Logs
```
ThermoFleet-eVTOL-Simulator/
├── logs/
│   ├── PPO_1/
│   │   └── events.out.tfevents.*
│   ├── DDPG_1/
│   │   └── events.out.tfevents.*
│   └── evaluations.npz
```

### Replay Files
```
ThermoFleet-eVTOL-Simulator/
├── replays/
│   ├── episode_*.pkl.gz           ← Episode data
│   └── episode_*_metadata.json    ← Episode metadata
```

---

## 🔄 Data Flow During Training

```
┌──────────────┐
│  train.py    │
└──────────────┘
       │
       ├─► (1) Save to SQLite/MySQL via DatabaseLogger
       │        └─► episodes, metrics, sensor_logs tables
       │
       ├─► (2) Log to WandB (if --use-wandb)
       │        └─► wandb.log({'reward': ..., 'loss': ...})
       │
       ├─► (3) Write to TensorBoard
       │        └─► logs/PPO_1/events.out.tfevents.*
       │
       └─► (4) Save model checkpoints
                └─► models/ppo_medium_*.zip
```

---

## 🎯 Which System Should You Use When?

### Use **Dashboard** (SQLite/MySQL) for:
- ✅ Real-time monitoring of training
- ✅ Analyzing episode data
- ✅ Comparing vehicle types
- ✅ Viewing 3D trajectories
- ✅ Energy consumption analysis
- ✅ Success rate tracking
- ✅ Database maintenance

### Use **WandB** for:
- ✅ Comparing multiple experiments
- ✅ Hyperparameter sweeps
- ✅ Team collaboration (sharing results)
- ✅ Cloud-based experiment tracking
- ✅ System resource monitoring
- ✅ Long-term experiment history

### Use **TensorBoard** for:
- ✅ Deep learning metrics (loss curves)
- ✅ Network architecture visualization
- ✅ Weight distributions
- ✅ Gradient monitoring
- ✅ Custom scalar tracking

### Use **Replay System** for:
- ✅ Episode playback
- ✅ Trajectory visualization
- ✅ Debugging specific episodes
- ✅ Creating demo videos

---

## ⚙️ Configuration Priority

The dashboard follows this priority for database selection:

```python
# 1. Check .env file
DB_TYPE=sqlite  # or 'mysql'

# 2. If no .env, defaults to:
db_type = os.getenv('DB_TYPE', 'sqlite')  # ← Default: SQLite

# 3. Database URL is constructed:
# SQLite: sqlite:///data/database/thermofleet_evtol.db
# MySQL: mysql+pymysql://{user}:{password}@{host}:{port}/{database}
```

---

## 🔍 How to Check Current Connections

### Check Dashboard Connection:
```python
# In dashboard.py sidebar:
if st.session_state.db_connected:
    st.success("✅ Database Connected")
    stats = st.session_state.db_manager.get_database_stats()
    st.metric("Total Episodes", stats['total_episodes'])
else:
    st.error("❌ Database Connection Failed")
```

### Check Database Type:
```bash
# View current database type
cat .env | grep DB_TYPE

# Or check directly in Python:
python -c "from src.database.db_manager import DatabaseManager; db = DatabaseManager(); print(f'Connected to: {db.db_type}')"
```

### Check if WandB is Active:
```bash
# Check WandB status
wandb status

# View WandB runs
ls -la wandb/

# Check WandB config
cat ~/.netrc | grep wandb
```

---

## 🚀 Complete Setup Example

```bash
# 1. Initialize SQLite database (required for dashboard)
python scripts/init_db.py

# 2. (Optional) Set up WandB for cloud tracking
wandb login

# 3. Train with ALL systems enabled
python train.py \
    --algo=PPO \
    --vehicle-type=medium \
    --total-timesteps=1000000 \
    --use-wandb \
    --record-episodes

# 4. Monitor training (3 separate windows):

# Terminal 1: Streamlit Dashboard (SQLite)
streamlit run dashboard.py

# Terminal 2: TensorBoard (local logs)
tensorboard --logdir=./logs

# Terminal 3: WandB (cloud)
# Open: https://wandb.ai/<your-username>/thermofleet-evtol-simulator
```

---

## 📊 Data Flow Summary

| System | Storage | Used By | Purpose | Connection |
|--------|---------|---------|---------|------------|
| **SQLite** | `data/database/` | Dashboard, Scripts | Episode data, metrics | ✅ Dashboard Default |
| **MySQL** | Remote server | Dashboard, Scripts | Production database | ✅ Dashboard (if configured) |
| **WandB** | Cloud | Training only | Experiment tracking | ❌ NOT used by dashboard |
| **TensorBoard** | `logs/` | Training only | Deep learning metrics | ❌ NOT used by dashboard |
| **Replays** | `replays/` | Dashboard, Scripts | Episode playback | ✅ Dashboard reads metadata |

---

## 🎯 Key Takeaways

1. **Dashboard = SQLite/MySQL ONLY**
   - Does NOT read from WandB
   - Does NOT read from TensorBoard
   - Only connects to relational database

2. **Three Independent Systems**
   - Database: Persistent storage
   - WandB: Cloud experiment tracking
   - TensorBoard: ML metrics visualization

3. **Use All Three Together**
   - They complement each other
   - Different strengths for different needs
   - No conflicts or redundancy

4. **SQLite is Default**
   - Works out of the box
   - No configuration needed
   - Perfect for local development

---

## 🔧 Troubleshooting

### "Database not connected" in Dashboard
```bash
# Initialize the database
python scripts/init_db.py

# Check database file exists
ls -la data/database/thermofleet_evtol.db

# Test connection
python -c "from src.database.db_manager import DatabaseManager; DatabaseManager()"
```

### Want to switch from SQLite to MySQL?
```bash
# 1. Update .env
echo "DB_TYPE=mysql" >> .env
echo "DB_HOST=localhost" >> .env
echo "DB_NAME=thermofleet_evtol" >> .env
echo "DB_USER=your_user" >> .env
echo "DB_PASSWORD=your_password" >> .env

# 2. Create MySQL database
mysql -u root -p
CREATE DATABASE thermofleet_evtol;

# 3. Initialize tables
python scripts/init_db.py --db-type mysql

# 4. Restart dashboard
streamlit run dashboard.py
```

### See WandB data in Dashboard?
Not currently supported, but you could add it:
```python
# Would need to add to dashboard.py:
import wandb
api = wandb.Api()
runs = api.runs("your-username/thermofleet-evtol-simulator")
```

---

**Need help?** Check the main [README.md](../README.md) or open an issue!

