"""
CPR Position Assessment Environment
=====================================
A custom Gymnasium environment for reinforcement learning-based
CPR position and procedure correctness assessment.

The environment models a patient on the ground and evaluates
whether observed body landmarks (via MediaPipe-style pose data)
correspond to correct first-aid / CPR procedures.

Observation: 17 normalized body landmark coordinates (x, y, visibility)
             + patient vitals proxy (HR estimate, chest rise rate)
             = 17*3 + 2 = 53-dimensional continuous vector

Action Space (Discrete, 12 actions):
  0  - ASSESS_CONSCIOUSNESS   : Check responsiveness
  1  - CALL_EMERGENCY         : Instruct bystander to call 911
  2  - OPEN_AIRWAY            : Head-tilt chin-lift
  3  - CHECK_BREATHING        : Look/listen/feel ≤10 sec
  4  - BEGIN_CHEST_COMPRESSIONS : 30x at 100-120 bpm, 5-6 cm depth
  5  - DELIVER_RESCUE_BREATHS : 2 breaths after 30 compressions
  6  - DEFIBRILLATE           : AED shock if shockable rhythm
  7  - MONITOR_PULSE          : Check carotid pulse
  8  - RECOVERY_POSITION      : Roll to lateral decubitus
  9  - REPOSITION_HANDS       : Correct hand placement (lower sternum)
  10 - TILT_HEAD_BACK         : Extend neck to open airway
  11 - WAIT_OBSERVE           : Passive observation (penalized if prolonged)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List


# ---------------------------------------------------------------------------
# Patient State Dataclass
# ---------------------------------------------------------------------------
@dataclass
class PatientState:
    """Encodes the physiological and positional state of the CPR patient."""

    # Pose landmarks (17 MediaPipe keypoints: nose, eyes, ears, shoulders,
    # elbows, wrists, hips, knees, ankles) — normalised [0,1]
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros(51))

    # Vital proxies
    heart_rate: float = 0.0          # 0 = no pulse, 1 = normal
    chest_rise_rate: float = 0.0     # breaths detected per minute (norm)
    airway_open: bool = False
    head_position: float = 0.0       # 0=neutral, 1=tilted back (ideal)
    hand_placement: float = 0.0      # 0=wrong, 1=correct sternal position
    compression_depth: float = 0.0   # 0=none, 1=adequate (5-6 cm)
    recovery_position: bool = False
    consciousness_level: float = 0.0 # 0=unresponsive, 1=responsive

    # Episode counters
    compressions_delivered: int = 0
    breaths_delivered: int = 0
    time_without_action: int = 0


# ---------------------------------------------------------------------------
# Action metadata
# ---------------------------------------------------------------------------
ACTION_NAMES = [
    "ASSESS_CONSCIOUSNESS",
    "CALL_EMERGENCY",
    "OPEN_AIRWAY",
    "CHECK_BREATHING",
    "BEGIN_CHEST_COMPRESSIONS",
    "DELIVER_RESCUE_BREATHS",
    "DEFIBRILLATE",
    "MONITOR_PULSE",
    "RECOVERY_POSITION",
    "REPOSITION_HANDS",
    "TILT_HEAD_BACK",
    "WAIT_OBSERVE",
]

N_ACTIONS = len(ACTION_NAMES)
OBS_DIM = 53  # 51 landmark floats + heart_rate + chest_rise_rate


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class CPREnv(gym.Env):
    """
    Custom Gymnasium environment for CPR position assessment.

    Parameters
    ----------
    max_steps : int
        Maximum number of steps per episode (default 200).
    noise_std : float
        Gaussian noise added to landmark observations to simulate
        MediaPipe detection jitter (default 0.02).
    difficulty : str
        'easy'   — patient responds faster, tolerant rewards
        'medium' — standard scenario
        'hard'   — delayed responses, strict sequencing required
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        max_steps: int = 200,
        noise_std: float = 0.02,
        difficulty: str = "medium",
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.noise_std = noise_std
        self.difficulty = difficulty
        self.render_mode = render_mode

        # --- Spaces --------------------------------------------------------
        # Continuous observation: landmark coords + vitals
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        # Discrete action space: 12 CPR actions
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Internal state
        self._patient: Optional[PatientState] = None
        self._step_count: int = 0
        self._actions_done: set = set()
        self._protocol_stage: int = 0  # 0–5 protocol phases
        self._cumulative_reward: float = 0.0

        # Difficulty multipliers
        _diff_map = {"easy": 1.5, "medium": 1.0, "hard": 0.6}
        self._diff_scale = _diff_map.get(difficulty, 1.0)

        # Rendering
        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._step_count = 0
        self._protocol_stage = 0
        self._cumulative_reward = 0.0
        self._actions_done = set()

        # Initialise patient in collapsed state
        self._patient = PatientState(
            landmarks=self._generate_collapse_pose(),
            heart_rate=0.0,
            chest_rise_rate=0.0,
            airway_open=False,
            head_position=0.0,
            hand_placement=0.0,
            compression_depth=0.0,
            recovery_position=False,
            consciousness_level=0.0,
            compressions_delivered=0,
            breaths_delivered=0,
            time_without_action=0,
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count += 1
        reward = self._compute_reward(action)
        self._apply_action(action)
        self._patient.time_without_action = (
            0 if action != 11 else self._patient.time_without_action + 1
        )

        self._cumulative_reward += reward

        obs = self._get_obs()
        terminated = self._is_terminal()
        truncated = self._step_count >= self.max_steps
        info = self._get_info()
        info["action_name"] = ACTION_NAMES[action]
        info["reward"] = reward

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            from environment.rendering import CPRRenderer
            self._renderer = CPRRenderer()
        self._renderer.render(self._patient, self._step_count, self._cumulative_reward)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_collapse_pose(self) -> np.ndarray:
        """
        Generate realistic normalised landmark coordinates of a person
        lying supine on the ground (y-values near 0.8–1.0 = bottom of frame).
        Coordinates: [x0,y0,v0, x1,y1,v1, ...] for 17 keypoints.
        """
        rng = self.np_random if hasattr(self, "np_random") else np.random
        # Rough supine layout (MediaPipe 17-keypoint order)
        base_landmarks = np.array([
            # nose
            0.50, 0.82, 0.95,
            # left_eye_inner, left_eye, left_eye_outer
            0.53, 0.80, 0.90,
            0.56, 0.79, 0.88,
            0.58, 0.78, 0.85,
            # right_eye_inner, right_eye, right_eye_outer
            0.47, 0.80, 0.90,
            0.44, 0.79, 0.88,
            0.42, 0.78, 0.85,
            # left_ear
            0.60, 0.81, 0.80,
            # right_ear
            0.40, 0.81, 0.80,
            # mouth_left
            0.54, 0.84, 0.90,
            # mouth_right
            0.46, 0.84, 0.90,
            # left_shoulder
            0.60, 0.72, 0.95,
            # right_shoulder
            0.40, 0.72, 0.95,
            # left_elbow
            0.65, 0.60, 0.90,
            # right_elbow
            0.35, 0.60, 0.90,
            # left_wrist
            0.62, 0.50, 0.85,
            # right_wrist
            0.38, 0.50, 0.85,
        ], dtype=np.float32)

        # Add jitter
        noise = rng.normal(0, 0.015, size=base_landmarks.shape).astype(np.float32)
        return np.clip(base_landmarks + noise, 0.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        p = self._patient
        base = np.concatenate([
            p.landmarks,
            [p.heart_rate, p.chest_rise_rate],
        ]).astype(np.float32)

        # Sensor noise
        noise = self.np_random.normal(0, self.noise_std, size=base.shape).astype(np.float32)
        return np.clip(base + noise, 0.0, 1.0)

    def _get_info(self) -> Dict[str, Any]:
        p = self._patient
        return {
            "step": self._step_count,
            "protocol_stage": self._protocol_stage,
            "heart_rate": p.heart_rate,
            "compressions": p.compressions_delivered,
            "breaths": p.breaths_delivered,
            "airway_open": p.airway_open,
            "cumulative_reward": self._cumulative_reward,
        }

    def _is_terminal(self) -> bool:
        p = self._patient
        # Success: ROSC — heart rate restored is sufficient.
        # consciousness_level is a bonus, not a requirement.
        if p.heart_rate >= 0.85:
            return True
        # Failure: prolonged inaction
        if p.time_without_action > 15:
            return True
        # Failure: cumulative reward too negative (agent is trapped in wrong loop)
        if self._cumulative_reward < -150:
            return True
        return False

    def _compute_reward(self, action: int) -> float:
        """
        Reward function v2 — all exploits closed.

        Key fixes vs v1:
          - CALL_EMERGENCY now penalised when out of sequence (was +0.5 always)
          - OPEN_AIRWAY now penalised when out of sequence (was 0 always)
          - TILT_HEAD_BACK penalised when out of sequence
          - CHECK_BREATHING penalised out of sequence (was +0.3 always)
          - MONITOR_PULSE penalised stage < 2 (was only -0.3 stage < 3)
          - Step-level time penalty added to prevent stalling
          - ROSC terminal bonus increased to strongly reward protocol completion
          - Repeating any already-completed action is penalised
        """
        p = self._patient
        r = 0.0
        stage = self._protocol_stage
        scale = self._diff_scale

        # ── Per-step living penalty — discourages stalling/looping ───────────
        # Grows linearly with steps so early exploration is tolerated but
        # long unproductive loops are costly.
        step_penalty = -0.02 * (self._step_count / self.max_steps)

        # ── Action already done this stage → penalty for repeating ───────────
        already_done = self._actions_done

        # ── Protocol-sequenced rewards ────────────────────────────────────────
        if action == 0:  # ASSESS_CONSCIOUSNESS
            if stage == 0:
                r = 3.0 * scale
                self._protocol_stage = 1
                already_done.add(0)
            elif action in already_done:
                r = -1.0  # already completed, don't repeat
            else:
                r = -0.8  # out of order

        elif action == 1:  # CALL_EMERGENCY
            if stage == 1:
                r = 3.0 * scale
                self._protocol_stage = 2
                already_done.add(1)
            elif stage == 0:
                r = -0.5  # calling before assessing
            elif action in already_done:
                r = -1.0
            else:
                r = -0.5  # no free reward for out-of-order calling

        elif action == 10:  # TILT_HEAD_BACK
            if stage >= 2:
                if p.head_position < 0.9:
                    r = 1.5 * scale
                    p.head_position = min(1.0, p.head_position + 0.4)
                    p.airway_open = p.head_position > 0.5
                else:
                    r = -0.3  # already tilted
            else:
                r = -0.8  # out of order

        elif action == 2:  # OPEN_AIRWAY
            if stage >= 2:
                if not p.airway_open:
                    r = 1.5 * scale
                    p.head_position = min(1.0, p.head_position + 0.5)
                    p.airway_open = True
                    if stage == 2:
                        self._protocol_stage = 3
                        already_done.add(2)
                else:
                    r = -0.3  # airway already open
            else:
                r = -0.8  # out of order — significant penalty

        elif action == 3:  # CHECK_BREATHING
            if stage == 3:
                r = 1.5 * scale
                self._protocol_stage = 4
                already_done.add(3)
            elif stage > 3:
                r = -0.3  # already checked
            else:
                r = -0.8  # out of order

        elif action == 9:  # REPOSITION_HANDS
            if stage >= 4:
                if p.hand_placement < 1.0:
                    p.hand_placement = min(1.0, p.hand_placement + 0.5)
                    r = 1.0 * scale
                else:
                    r = -0.2  # already positioned
            else:
                r = -0.5  # out of order

        elif action == 4:  # BEGIN_CHEST_COMPRESSIONS
            if stage >= 4:
                depth_bonus = p.hand_placement
                p.compression_depth = min(1.0, 0.4 + 0.6 * depth_bonus)
                p.compressions_delivered += 30
                p.heart_rate = min(1.0, p.heart_rate + 0.15 * p.compression_depth * scale)
                r = 3.0 * p.compression_depth * scale
                if stage == 4:
                    self._protocol_stage = 5
            else:
                r = -2.0  # strong penalty for skipping assessment

        elif action == 5:  # DELIVER_RESCUE_BREATHS
            if stage >= 5 and p.airway_open:
                p.breaths_delivered += 2
                p.chest_rise_rate = min(1.0, p.chest_rise_rate + 0.2 * scale)
                p.heart_rate = min(1.0, p.heart_rate + 0.05 * scale)
                r = 2.0 * scale
            elif not p.airway_open:
                r = -2.0  # airway closed — dangerous
            else:
                r = -1.0  # out of order

        elif action == 6:  # DEFIBRILLATE
            if p.heart_rate >= 0.80:
                r = -1.0   # HR already high — defibrillation unnecessary
            elif stage >= 4 and p.compressions_delivered >= 30:
                p.heart_rate = min(1.0, p.heart_rate + 0.3 * scale)
                r = 4.0 * scale
                already_done.add(6)
            else:
                r = -2.0  # defibrillating without compressions

        elif action == 7:  # MONITOR_PULSE
            if stage >= 4:
                r = 0.5
            elif stage >= 2:
                r = -0.2
            else:
                r = -0.5  # no point monitoring before any intervention

        elif action == 8:  # RECOVERY_POSITION
            if 8 in already_done:
                r = -1.5   # already done — repeating it is wrong
            elif p.heart_rate >= 0.7:
                p.recovery_position = True
                p.consciousness_level = min(1.0, p.consciousness_level + 0.3)
                r = 5.0 * scale  # big reward — this is success
                already_done.add(8)
            else:
                r = -1.0  # premature recovery position

        elif action == 11:  # WAIT_OBSERVE
            r = -0.5 * (1.0 + p.time_without_action * 0.15)  # escalating

        # ── Stage progress bonus — small reward for advancing protocol ────────
        # (only fires on the step that advances the stage)
        # Already incorporated in the stage transitions above.

        # ── ROSC terminal bonus ───────────────────────────────────────────────
        if p.heart_rate >= 0.9 and p.consciousness_level >= 0.7:
            r += 20.0  # strong terminal signal

        # ── Continuous state decay without active intervention ─────────────────
        if action not in [4, 5, 6]:
            p.heart_rate = max(0.0, p.heart_rate - 0.008)

        # ── Apply step penalty ────────────────────────────────────────────────
        r += step_penalty

        # ── Update landmarks ──────────────────────────────────────────────────
        self._update_landmarks(action)

        return float(np.clip(r, -5.0, 15.0))

    def _apply_action(self, action: int):
        """Apply secondary state changes not in reward computation."""
        p = self._patient
        if action == 8 and p.recovery_position:
            # Shift landmarks to reflect recovery position
            p.landmarks = self._generate_recovery_pose()

    def _update_landmarks(self, action: int):
        """Update landmarks to reflect action effects on patient posture."""
        p = self._patient
        if action == 10 or action == 2:
            # Head tilts back — nose y decreases (head moves up in frame)
            p.landmarks[1] = max(0.1, p.landmarks[1] - 0.05 * p.head_position)
        elif action == 4:
            # Chest compressions — slight chest depression visible
            p.landmarks[33] = min(1.0, p.landmarks[33] + 0.02)  # left shoulder
            p.landmarks[36] = min(1.0, p.landmarks[36] + 0.02)  # right shoulder

    def _generate_recovery_pose(self) -> np.ndarray:
        """Lateral decubitus recovery position landmarks."""
        base = self._generate_collapse_pose()
        # Shift x-coordinates to represent lateral rotation
        for i in range(0, len(base), 3):
            base[i] = np.clip(base[i] + 0.15, 0.0, 1.0)
        return base