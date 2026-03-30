# Unity 3D Scene Setup Guide

## Overview

The Unity project receives real-time state from the Python RL engine
via WebSocket (ws://localhost:8765) and renders a 3D CPR simulation
with animated patient and rescuer avatars.

---

## Step 1 — Create a New Unity Project

1. Open **Unity Hub**
2. Click **New Project** → Select **3D (URP)** or **3D (Built-in)** template
3. Name it `CPR_RL_Simulation`
4. Click **Create Project**

---

## Step 2 — Install NativeWebSocket

In the Unity **Package Manager** (`Window → Package Manager`):

1. Click the `+` button → **Add package from git URL**
2. Paste:
   ```
   https://github.com/endel/NativeWebSocket.git#upm
   ```
3. Click **Add**. Wait for installation.

Alternatively, edit `Packages/manifest.json` and add:
```json
"com.endel.nativewebsocket": "https://github.com/endel/NativeWebSocket.git#upm"
```

---

## Step 3 — Copy Scripts

Copy everything from `unity_project/Assets/Scripts/` into your Unity project's `Assets/Scripts/` folder:

```
BridgeClient.cs       ← WebSocket receiver
PatientController.cs  ← Patient avatar animation
RescuerController.cs  ← Rescuer avatar animation
CPRSceneManager.cs    ← Scene orchestration
CPR_HUD.cs            ← Layperson UI overlay
```

---

## Step 4 — Build the Scene

### Create the hierarchy:
```
Scene
├── BridgeClient         (empty GameObject + BridgeClient.cs)
├── SceneManager         (empty GameObject + CPRSceneManager.cs)
├── Patient              (empty GameObject + PatientController.cs)
│   └── [auto-created]   primitive body parts
├── Rescuer              (empty GameObject + RescuerController.cs)
│   └── [auto-created]   primitive body parts
├── UI Canvas            (Canvas + CPR_HUD.cs)
│   ├── PhaseLabel       (Text)
│   ├── FeedbackPanel    (Image + Text)
│   ├── VitalsPanel      (Image + Sliders)
│   └── ProtocolGuide    (Image + Text[6])
├── Main Camera
├── Directional Light
└── Floor                (Plane, scale 5,1,5)
```

### Quick Setup (no asset store required):
1. `PatientController` and `RescuerController` will **auto-build** primitive avatars if no body part references are assigned — just attach the scripts and press Play.
2. The floor mat, AED prop, and camera rigs are also auto-created by `CPRSceneManager`.

---

## Step 5 — Optional: Human Avatar (Mixamo)

For a realistic human avatar:
1. Go to [mixamo.com](https://www.mixamo.com), download a character + animations:
   - **Idle** animation
   - **Push Up** animation (closest to compressions)
   - **Kneel** animation
2. Import into Unity as **Humanoid** rig
3. Create an **Animator Controller** with states: `Idle`, `Compress`, `Assess`, `OpenAirway`, `Recovery`
4. Assign the Animator to the `PatientController.humanoidAnimator` and `RescuerController.humanoidAnimator` fields in the Inspector

---

## Step 6 — UI Canvas Setup

Create a **Canvas** (`UI → Canvas`, set to **Screen Space - Overlay`):

| GameObject         | Component     | Assign to CPR_HUD field  |
|--------------------|---------------|--------------------------|
| PhaseLabel         | Text          | `phaseLabel`             |
| ConnectionStatus   | Text          | `connectionStatus`       |
| PhaseBanner        | Image         | `phaseBanner`            |
| FeedbackText       | Text          | `feedbackText`           |
| FeedbackPanel      | Image         | `feedbackPanel`          |
| ActionNameText     | Text          | `actionNameText`         |
| StepCounter        | Text          | `stepCounter`            |
| RewardText         | Text          | `rewardText`             |
| HeartRateSlider    | Slider        | `heartRateSlider`        |
| ChestRiseSlider    | Slider        | `chestRiseSlider`        |
| HandPlacementSlider| Slider        | `handPlacementSlider`    |
| ExperimentText     | Text          | `experimentText`         |
| ROSCPanel          | GameObject    | `roscPanel`              |

---

## Step 7 — Connect Components

In the **Inspector**, assign references between GameObjects:
- `CPRSceneManager.patientController` → Patient GameObject
- `CPRSceneManager.rescuerController` → Rescuer GameObject
- `CPRSceneManager.mainCamera` → Main Camera

---

## Step 8 — Run the Full System

### Terminal 1 — Python Bridge:
```bash
# Random agent demo (no training needed)
python main.py --random

# Demo best trained model
python main.py --demo --algo ppo

# With MediaPipe / CPR-Coach video
python main.py --demo --mediapipe
python main.py --demo --video data/cpr_coach/cpr_sample.mp4
```

### Unity:
1. Press **Play** in Unity
2. Unity connects to `ws://localhost:8765`
3. The scene animates in sync with the Python RL agent

---

## Architecture Diagram

```
┌──────────────────────────────┐         WebSocket
│       PYTHON BACKEND         │    ws://localhost:8765
│                              │ ─────────────────────→  ┌────────────────────┐
│  CPREnv (Gymnasium)          │                          │   UNITY 3D SCENE   │
│    ↕ observations/actions    │   JSON state packets     │                    │
│  RL Agent (DQN/PPO/REINFORCE)│ ─────────────────────→  │  PatientController │
│    ↕                         │                          │  RescuerController │
│  MediaPipe Extractor         │                          │  CPR_HUD           │
│  (video / webcam / synthetic)│                          │  CPRSceneManager   │
└──────────────────────────────┘                          └────────────────────┘
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NativeWebSocket not found` | Add package from git URL in Package Manager |
| Unity can't connect to Python | Ensure Python is running first, check firewall |
| No avatar appears | Just press Play — primitives auto-create |
| Blank HUD | Assign Text/Image references in Inspector |
| `mediapipe not installed` | `pip install mediapipe opencv-python` |
| CPR-Coach download fails | Set `HF_TOKEN` env var or use `--video` with local file |

---

## Phase Behaviour

| Phase | Command | Unity Scene |
|-------|---------|-------------|
| **Random** | `python main.py --random` | Grey ambient, random chaotic actions |
| **Training** | `python main.py --train` | Blue ambient, experiment counter updates |
| **Demo** | `python main.py --demo` | Green ambient, cinematic camera, smooth protocol |
| **ROSC** | (auto) | Green glow, celebration particles, patient sits up |
