# CPR Position Assessment — Reinforcement Learning

An RL system that learns to recognize and evaluate correct first-aid (CPR) positions using MediaPipe body landmark observations. Three algorithms (DQN, REINFORCE, PPO) are trained in a custom Gymnasium environment and compared across 10 hyperparameter experiments each.

![Demo](demo.gif)

---

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment (53-dim obs, 12 discrete actions)
│   └── rendering.py         # Pygame visualization GUI
├── training/
│   ├── dqn_training.py      # DQN — 10 experiments, auto-resume
│   └── pg_training.py       # REINFORCE + PPO — 10 experiments each, auto-resume
├── models/
│   ├── dqn/                 # Saved DQN models
│   └── pg/                  # REINFORCE & PPO models
├── results/                 # JSON result files per algorithm
├── logs/                    # TensorBoard logs
├── main.py                  # Entry point — runs best model with GUI + terminal
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.10 or 3.11
- macOS / Linux / Windows

```bash
# Clone the repository
git clone <your-repo-url>
cd cpr_rl_project

# Create virtual environment
python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Random Agent Demo (no training required)
Demonstrates the environment and Pygame GUI without any trained model:
```bash
python main.py --random --episodes 3
```

### 2. Train All Algorithms
```bash
# DQN (10 experiments, ~30–60 min)
python training/dqn_training.py

# REINFORCE + PPO (10 experiments each, ~60–120 min)
python training/pg_training.py --algo all

# Or individually:
python training/pg_training.py --algo reinforce
python training/pg_training.py --algo ppo
```

Training is **resume-safe** — if interrupted, re-run the same command and it will continue from the last completed experiment.

### 3. Run Best Trained Agent
```bash
# Auto-detect best algorithm
python main.py

# Specify algorithm
python main.py --algo dqn
python main.py --algo ppo
python main.py --algo reinforce

# Terminal only (no GUI)
python main.py --no-render

# Multiple episodes
python main.py --episodes 5
```

### 4. TensorBoard
```bash
tensorboard --logdir logs/
```

---

## Environment Design

### Observation Space
- **Dimension:** 53 continuous values ∈ [0, 1]
- **Components:** 17 MediaPipe pose keypoints × 3 (x, y, visibility) + heart rate proxy + chest rise rate
- **Noise:** Gaussian jitter (σ=0.02) simulates real-world MediaPipe detection variance

### Action Space
| ID | Action | Description |
|----|--------|-------------|
| 0  | ASSESS_CONSCIOUSNESS | Check patient responsiveness |
| 1  | CALL_EMERGENCY | Instruct bystander to call 911 |
| 2  | OPEN_AIRWAY | Head-tilt chin-lift maneuver |
| 3  | CHECK_BREATHING | Look/listen/feel ≤10 seconds |
| 4  | BEGIN_CHEST_COMPRESSIONS | 30× at 100-120 bpm, 5-6 cm depth |
| 5  | DELIVER_RESCUE_BREATHS | 2 breaths after 30 compressions |
| 6  | DEFIBRILLATE | AED shock if shockable rhythm |
| 7  | MONITOR_PULSE | Check carotid pulse |
| 8  | RECOVERY_POSITION | Roll to lateral decubitus |
| 9  | REPOSITION_HANDS | Correct lower-sternum hand placement |
| 10 | TILT_HEAD_BACK | Extend neck to open airway |
| 11 | WAIT_OBSERVE | Passive — penalized if prolonged |

### Reward Structure
- **Sequential protocol compliance:** +2.0 per correct stage progression
- **Chest compressions with good hand placement:** up to +3.0
- **ROSC (Return of Spontaneous Circulation):** +10.0 terminal bonus
- **Skipped steps / wrong order:** −1.0
- **Rescue breaths with closed airway:** −1.5
- **Prolonged inaction:** −0.5 × (1 + t×0.1) escalating

### Terminal Conditions
- **Success:** heart_rate ≥ 0.9 AND consciousness_level ≥ 0.7 (ROSC achieved)
- **Failure:** inaction > 15 consecutive steps
- **Truncation:** max_steps reached (default 200)

---

## MediaPipe Integration

For production use, replace `_generate_collapse_pose()` in `custom_env.py` with live or video-extracted MediaPipe landmarks:

```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture("cpr_video.mp4")
ret, frame = cap.read()
results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    landmarks = np.array([
        [lm.x, lm.y, lm.visibility]
        for lm in results.pose_landmarks.landmark[:17]
    ]).flatten()
```

---

## Results

After training, results are stored in `results/`:
- `dqn_results.json`
- `ppo_results.json`
- `reinforce_results.json`

Each file contains per-experiment metrics: mean reward, max reward, hyperparameters, and reward curves for plotting.
