# Your visual CPR Position Assessment tool with Reinforcement Learning

This thorough work demonstrates the potential of RL to assess and provide guidance on procedures like CPR, which could have real-world applications in training and emergency response. It's built with custom environment utilizing MediaPipe and Unity for visualization how the model applies its learned policy in a simulated CPR scenario. The discrete action are 11 actions that represent different steps in the CPR protocol, and the reward structure encourages correct sequential actions while penalizing mistakes and inaction.

Three algorithms (DQN, REINFORCE, PPO) are trained in a custom Gymnasium environment and compared across 10 hyperparameter experiments each.

# Demo

<p align="center">
  <img src="demo.gif" width="900">
</p>

---
# Features

| Component | Details |
|-----------|---------|
| **Environment** | Gymnasium, 12 CPR actions, 56-dim obs (landmarks + vitals + temporal) |
| **Algorithms** | DQN (SB3), PPO (SB3), vanilla REINFORCE (PyTorch) |
| **Compression rate** | Real-time bpm reward signal (100–120 target) |
| **Curriculum** | `env.set_difficulty('easy' / 'medium' / 'hard')` |
| **Rendering** | Pygame: scrolling ECG, ROSC particles, compression depth meter |
| **Unity 3D** | WebSocket bridge, animated humanoids, particle burst on ROSC |
| **Resume-safe** | Training auto-resumes from last completed experiment |

# Project Structure

```
NoviceRL/
├── environment/
│   ├── custom_env.py          # Gymnasium environment (v3)
│   ├── rendering.py           # Pygame dashboard (v4)
│   ├── mediapipe_extractor.py # Landmark extraction (video / webcam / synthetic)
│   └── unity_bridge.py        # WebSocket broadcast server
│
├── training/
│   ├── dqn_training.py        # DQN — 10 hyperparameter experiments
│   └── pg_training.py         # PPO + REINFORCE — 10 experiments each
│
├── unity/
│   ├── BridgeClient.cs        # WebSocket client + event dispatcher
│   ├── CPR_HUD.cs             # Canvas UI overlay
│   ├── CPRSceneEnvironment.cs # Procedural room + particle system (v3)
│   ├── CPRSceneManager.cs     # Scene + camera orchestration
│   ├── HumanoidBuilder.cs     # Shared avatar factory
│   ├── HumanoidPatient.cs     # Patient animations
│   └── HumanoidRescuer.cs     # Rescuer IK + fatigue system (v6)
│
├── models/                    # Saved model checkpoints
├── results/                   # JSON experiment logs
├── logs/                      # TensorBoard logs
│
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

---

# Setup

## Prerequisites
- Python 3.10 or 3.11
- macOS / Linux / Windows

### 1 Clone & install

```bash
git clone https://github.com/Gatwaza/Jean_Robert_Gatwaza_rl_summative.git
cd NoviceRL
python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
##### 2 Standalone Pygame demo without Unity bridge

```bash
# Random agent see the environment
python main.py --random --no-bridge

# Demo best trained model
python main.py --demo --no-bridge
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

# Use MediaPipe on a video file
python main.py --demo --no-bridge --video data/upload_cpr_like_video_to_this_folder.mp4 --mediapipe
```
###### 3 Train all algorithms

```bash
# Train DQN (10 experiments × 750k steps ≈ 40 mins on GPU laptop & 2–4 hours on CPU)
python training/dqn_training.py

# Train PPO + REINFORCE
python training/pg_training.py --algo all

# Or train a specific algorithm
python training/pg_training.py --algo ppo
```

###### 4 Unity 3D visualization

1. Open the NoviceUnityRL in `unity/` (Unity 6.0+ recommended)
2. Install **NativeWebSocket** via Package Manager:
   ```
   https://github.com/endel/NativeWebSocket.git#upm
   ```
3. Press **Play** in Unity find the button at middle of top of unity editor. 
4. In a separate terminal:
   ```bash
   python main.py --demo
   # or
   python main.py --random
   ```

The Python bridge auto-connects on `ws://localhost:8765`.
---

# Environment Design

## Observation Space
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

#### Reward Function Summary

| Action | Condition | Reward |
|--------|-----------|--------|
| ASSESS_CONSCIOUSNESS | Stage 0 | +3.0 × scale |
| CALL_EMERGENCY | Stage 1 | +3.0 × scale |
| OPEN_AIRWAY | Stage ≥ 2, airway closed | +1.5 × scale |
| CHECK_BREATHING | Stage 3 | +1.5 × scale |
| BEGIN_COMPRESSIONS | Stage ≥ 4 | +3.5 × depth × scale |
| → at 100–120 bpm | Rate bonus | +1.0 × scale |
| RESCUE_BREATHS | Stage ≥ 5, airway open | +2.5 × scale |
| DEFIBRILLATE | ≥ 30 compressions first | +4.5 × scale |
| RECOVERY_POSITION | HR ≥ 0.7 | +5.0 × scale |
| ROSC terminal | HR ≥ 0.9 sustained | +20.0 |
| Any action out of order | — | −0.5 to −2.0 |

Difficulty scales: `easy` = 1.6×, `medium` = 1.0×, `hard` = 0.65×

# Terminal Conditions
- **Success:** heart_rate ≥ 0.9 AND consciousness_level ≥ 0.7 (ROSC achieved)
- **Failure:** inaction > 15 consecutive steps
- **Truncation:** max_steps reached (default 200)

# License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

> **Medical Disclaimer:** This tool is a training and simulation aid only. It does not replace formal CPR certification or professional medical advice. Always call emergency services first.