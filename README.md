# CPR Position Assessment with Reinforcement Learning

A reinforcement learning pipeline that assesses CPR (Cardiopulmonary Resuscitation) procedural correctness in real time. A custom Gymnasium environment models a supine patient using MediaPipe pose landmarks as observations. Three algorithms: DQN, REINFORCE, and PPO are each trained across 10 hyperparameter experiments and compared objectively.

The target application is a real-time clinical advisory tool: a camera feed extracts pose landmarks and the trained policy advises rescuers on procedural correctness, promoting optimal outcomes in pre-hospital emergencies.

<p align="center">
  <img src="demo.gif" width="900">
</p>

---

## Results Summary

| Algorithm | Best Exp | Mean Reward | Max Reward | Convergence |
|-----------|----------|------------|------------|-------------|
| **PPO** ★ | Exp 10   | **162.6**  | **218.2**  | ~900 eps    |
| DQN       | Exp 8    | 119.0      | 193.98     | ~1,200 eps  |
| REINFORCE | Exp 2    | 36.2       | 113.6      | ~1,750 eps  |

PPO dominates across all metrics: highest reward, fastest convergence, and best generalisation on 100 unseen hard-difficulty states (21.3±3.1 vs DQN 14.9±4.2 vs REINFORCE 8.2±5.8).

---

## Features

| Component | Details |
|-----------|---------|
| **Environment** | Gymnasium, 12 CPR actions, 56-dim obs (51 landmarks + vitals + stage + history) |
| **Algorithms** | DQN (SB3), PPO (SB3), vanilla REINFORCE (PyTorch) |
| **Curriculum** | `easy` → `medium` → `hard` difficulty progression across experiments |
| **Compression rhythm** | Step-count rhythm reward (100–120 bpm equivalent, deterministic) |
| **Anti-collapse** | Consecutive-action caps, diversity bonus (+0.05 per unique action in last 10 steps) |
| **Rendering** | Pygame: scrolling ECG, ROSC particles, compression depth meter |
| **Unity 3D** | WebSocket bridge, animated humanoids, particle burst on ROSC |
| **Resume-safe** | Training auto-resumes from last completed experiment |

---

## Project Structure

```
NoviceRL/
├── environment/
│   ├── custom_env.py          # Gymnasium environment (v4 — exploit-proof)
│   ├── rendering.py           # Pygame dashboard
│   ├── mediapipe_extractor.py # Landmark extraction (video / webcam / synthetic)
│   └── unity_bridge.py        # WebSocket broadcast server
│
├── training/
│   ├── dqn_training.py        # DQN — 10 hyperparameter experiments (n_envs=1)
│   └── pg_training.py         # PPO + REINFORCE — 10 experiments each
│
├── unity/
│   ├── BridgeClient.cs        # WebSocket client + event dispatcher
│   ├── CPR_HUD.cs             # Canvas UI overlay
│   ├── CPRSceneEnvironment.cs # Procedural room + particle system
│   ├── CPRSceneManager.cs     # Scene + camera orchestration
│   ├── HumanoidBuilder.cs     # Shared avatar factory
│   ├── HumanoidPatient.cs     # Patient animations
│   └── HumanoidRescuer.cs     # Rescuer IK + fatigue system
│
├── models/                    # Saved model checkpoints
├── results/                   # JSON experiment logs
│   ├── dqn_results.json
│   ├── ppo_results.json
│   ├── reinforce_results.json
│   ├── training_summary.json
│   └── plot_data.json
├── logs/                      # TensorBoard logs
│
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.10 or 3.11
- macOS / Linux / Windows

### 1 — Clone & install

```bash
git clone https://github.com/Gatwaza/Jean_Robert_Gatwaza_rl_summative.git
cd NoviceRL
python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2 — Standalone Pygame demo (no Unity bridge)

```bash
# Random agent — see the environment
python main.py --random --no-bridge

# Demo best trained model (auto-detects PPO)
python main.py --demo --no-bridge

# Specify algorithm
python main.py --algo ppo
python main.py --algo dqn
python main.py --algo reinforce

# Terminal only (no GUI)
python main.py --no-render

# Multiple episodes
python main.py --episodes 5

# Use MediaPipe on a video file
python main.py --demo --no-bridge --video data/cpr_video.mp4 --mediapipe
```

### 3 — Train all algorithms

```bash
# Train DQN (10 experiments × 750k steps ≈ 40 min on GPU / 2–4 h on CPU)
python training/dqn_training.py

# Train PPO + REINFORCE
python training/pg_training.py --algo all

# Train a specific algorithm
python training/pg_training.py --algo ppo
python training/pg_training.py --algo reinforce
```

> **Resume safety:** Training automatically resumes from the last completed experiment. Re-running a script skips already-completed experiments based on the results JSON.

### 4 — Unity 3D visualisation

1. Open the project `NoviceUnityRL/` in (Unity 6.0+ recommended)
2. Install **NativeWebSocket** via Package Manager:
   ```
   https://github.com/endel/NativeWebSocket.git#upm
   ```
3. Press **Play** in the Unity editor
4. In a separate terminal:
   ```bash
   python main.py --demo
   # or
   python main.py --random
   ```

The Python bridge auto-connects on `ws://localhost:8765`.

---

## Environment Design

### Observation Space

**56 continuous values ∈ [0, 1]** constructed as:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Pose landmarks | 51 | 17 MediaPipe keypoints × 3 (x, y, visibility) |
| Vital proxies | 2 | Heart rate estimate + chest rise rate |
| Protocol stage | 1 | Normalised stage (0–5 → 0.0–1.0) |
| Action history | 2 | Last 2 actions (normalised action IDs) |

Gaussian jitter (σ=0.02) is injected each step to simulate real-world MediaPipe detection variance and prevent overfitting to clean simulation data.

### Action Space

| ID | Action | Description |
|----|--------|-------------|
| 0  | ASSESS_CONSCIOUSNESS | Check patient responsiveness |
| 1  | CALL_EMERGENCY | Instruct bystander to call 911 |
| 2  | OPEN_AIRWAY | Head-tilt chin-lift maneuver |
| 3  | CHECK_BREATHING | Look/listen/feel ≤10 seconds |
| 4  | BEGIN_CHEST_COMPRESSIONS | 30× at 100–120 bpm, 5–6 cm depth |
| 5  | DELIVER_RESCUE_BREATHS | 2 breaths after 30 compressions |
| 6  | DEFIBRILLATE | AED shock if shockable rhythm |
| 7  | MONITOR_PULSE | Check carotid pulse |
| 8  | RECOVERY_POSITION | Roll to lateral decubitus |
| 9  | REPOSITION_HANDS | Correct lower-sternum hand placement |
| 10 | TILT_HEAD_BACK | Extend neck to open airway |
| 11 | WAIT_OBSERVE | Passive — penalised if prolonged |

### Reward Function

| Action | Condition | Reward |
|--------|-----------|--------|
| ASSESS_CONSCIOUSNESS | Stage 0 | +3.0 × scale |
| CALL_EMERGENCY | Stage 1 | +3.0 × scale |
| OPEN_AIRWAY | Stage ≥ 2, airway closed | +1.5 × scale |
| CHECK_BREATHING | Stage 3 | +1.5 × scale |
| BEGIN_COMPRESSIONS | Stage ≥ 4 | +3.5 × depth × scale |
| → compression rhythm | Step-gap bonus | +1.0 × scale |
| RESCUE_BREATHS | Stage ≥ 5, airway open | +2.5 × scale |
| DEFIBRILLATE | ≥30 compressions first | +4.5 × scale |
| RECOVERY_POSITION | HR ≥ 0.7 | +5.0 × scale |
| ROSC terminal | HR ≥ 0.9 sustained | +8.0 |
| Any action out of order | — | −0.5 to −2.0 |
| Diversity bonus | per unique action in last 10 | +0.05 |

**Reward clip:** [−3.0, +10.0]. Difficulty scales: `easy` = 1.5×, `medium` = 1.0×, `hard` = 0.70×.

### Terminal Conditions

| Condition | Type |
|-----------|------|
| heart_rate ≥ 0.9 AND consciousness_level ≥ 0.7 | Success (ROSC) |
| time_without_action > 12 consecutive steps | Failure |
| cumulative_reward < −120 | Failure |
| step_count > max_steps (200) | Truncation |

### Anti-Exploit Mechanisms (v4)

The v4 environment includes several fixes for reward exploitation discovered in live training runs:

- **Consecutive-action caps:** MONITOR_PULSE capped at 3 in a row (was being farmed at +0.5/step for 100+ steps), WAIT_OBSERVE at 2, CHECK_BREATHING at 2, TILT_HEAD_BACK at 2
- **Compression streak multiplier:** consecutive compressions give up to 1.5× HR gain, simulating clinical 30:2 rhythm bonus
- **Diversity bonus:** +0.05 per unique action in the last 10 steps (max +0.25 at 6+ unique actions)
- **Step-count rhythm:** compression rate reward uses episode step-gap, not wall-clock time (which measured Python execution speed, not CPR rate)

---

## Algorithm Details

### DQN

- **Architecture:** MLP with configurable hidden layers (best: [128, 128, 64])
- **n_envs = 1** (required for correct target update semantics)
- **ε-greedy** exploration decaying from 1.0 → ε_final over `exploration_fraction` of training
- **Replay buffer:** 50K–100K transitions
- **Target update interval:** 1,000 steps
- **Key stability fix:** reward clip [−3, +10] and ROSC bonus 8.0 (prevents Q-value overestimation)

### REINFORCE

- **Custom PyTorch implementation** with Monte Carlo returns
- **Baseline:** optional exponential moving average (α=0.05) of episode mean return
- **Normalisation fix:** when baseline is active, divide by std only (not full re-centering)
- **Entropy regularisation:** β ∈ {0.05, 0.10, 0.15, 0.20}
- **Gradient clipping:** max_norm=1.0

### PPO (SB3)

- **Actor-critic** with Generalised Advantage Estimation (GAE, λ=0.95–0.98)
- **Clipped surrogate objective:** ε_clip ∈ {0.15, 0.20}
- **Entropy coefficient:** 0.15–0.30 (primary anti-collapse mechanism)
- **n_steps:** 2048–4096 per rollout collection
- **4 parallel environments** for trajectory diversity

---

## Licence

GNU General Public License v3.0 — see [LICENSE](LICENSE).

> **Medical Disclaimer:** This tool is a training and simulation aid only. It does not replace formal CPR certification or professional medical advice. Always call emergency services first.