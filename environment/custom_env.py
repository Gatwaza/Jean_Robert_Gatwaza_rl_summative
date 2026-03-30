"""
CPR Position Assessment Environment — FIXED v3.1
=================================================
Bug fixes applied vs original v3:

  FIX 1 (CRITICAL — Protocol Deadlock):
      TILT_HEAD_BACK now advances stage 2→3 when it successfully opens
      the airway, exactly as OPEN_AIRWAY does.  Previously, using
      TILT_HEAD_BACK at stage 2 would open the airway but never set
      _protocol_stage = 3, making CHECK_BREATHING permanently
      out-of-order (-0.8) and OPEN_AIRWAY redundant (-0.3, already open).
      The protocol was deadlocked at stage 2 and MONITOR_PULSE (-0.2)
      became the rational minimum-loss action — causing DQN to collapse.

  FIX 2 (CRITICAL — ROSC Bonus Unreachable):
      Terminal ROSC bonus previously required
          heart_rate >= 0.9 AND consciousness_level >= 0.7
      The episode terminates the instant heart_rate >= 0.9, so
      consciousness_level had to already be >= 0.7 when the killing
      blow was landed — an extremely narrow path.  The bonus now fires
      on heart_rate >= 0.9 alone (+15.0, clipped to the existing ceiling).
      A smaller supplemental bonus (+5.0) is retained for full
      consciousness recovery as an incentive, but it is no longer a gate.

  FIX 3 (HIGH — Heart-Rate Decay Too Aggressive):
      Decay was 0.008 per non-intervention step.  In a 200-step episode
      where DQN/REINFORCE waste 100+ steps monitoring, that erases all
      HR gains (100 × 0.008 = 0.8).  Reduced to 0.004 so the training
      signal from compressions survives long enough to be exploited.

  FIX 4 (MEDIUM — MONITOR_PULSE Trap at Stage 2):
      At stage 2 (after airway work), MONITOR_PULSE gave -0.2 while
      every other available action gave -0.3 to -2.0, making it the
      Q-network's rational minimum-loss choice.  Penalty raised to -0.5
      at stage < 4 to match the scale of other out-of-order penalties.

  FIX 5 (LOW — Already-done guard missing on TILT_HEAD_BACK):
      Added already_done.add(10) when TILT_HEAD_BACK successfully opens
      the airway and advances the stage, mirroring OPEN_AIRWAY's guard.
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

    landmarks: np.ndarray = field(default_factory=lambda: np.zeros(51))
    heart_rate: float = 0.0
    chest_rise_rate: float = 0.0
    airway_open: bool = False
    head_position: float = 0.0
    hand_placement: float = 0.0
    compression_depth: float = 0.0
    recovery_position: bool = False
    consciousness_level: float = 0.0
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
        Gaussian noise added to landmark observations (default 0.02).
    difficulty : str
        'easy' | 'medium' | 'hard'
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

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._patient: Optional[PatientState] = None
        self._step_count: int = 0
        self._actions_done: set = set()
        self._protocol_stage: int = 0
        self._cumulative_reward: float = 0.0

        _diff_map = {"easy": 1.5, "medium": 1.0, "hard": 0.6}
        self._diff_scale = _diff_map.get(difficulty, 1.0)

        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0
        self._protocol_stage = 0
        self._cumulative_reward = 0.0
        self._actions_done = set()

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

        return self._get_obs(), self._get_info()

    def step(self, action: int):
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
        rng = self.np_random if hasattr(self, "np_random") else np.random
        base_landmarks = np.array([
            0.50, 0.82, 0.95,
            0.53, 0.80, 0.90,
            0.56, 0.79, 0.88,
            0.58, 0.78, 0.85,
            0.47, 0.80, 0.90,
            0.44, 0.79, 0.88,
            0.42, 0.78, 0.85,
            0.60, 0.81, 0.80,
            0.40, 0.81, 0.80,
            0.54, 0.84, 0.90,
            0.46, 0.84, 0.90,
            0.60, 0.72, 0.95,
            0.40, 0.72, 0.95,
            0.65, 0.60, 0.90,
            0.35, 0.60, 0.90,
            0.62, 0.50, 0.85,
            0.38, 0.50, 0.85,
        ], dtype=np.float32)
        noise = rng.normal(0, 0.015, size=base_landmarks.shape).astype(np.float32)
        return np.clip(base_landmarks + noise, 0.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        p = self._patient
        base = np.concatenate([
            p.landmarks,
            [p.heart_rate, p.chest_rise_rate],
        ]).astype(np.float32)
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
        if p.heart_rate >= 0.9:
            return True
        if p.time_without_action > 15:
            return True
        return False

    def _compute_reward(self, action: int) -> float:
        """
        Reward function v3.1 — protocol deadlock fixed, ROSC bonus accessible.
        """
        p = self._patient
        r = 0.0
        stage = self._protocol_stage
        scale = self._diff_scale

        # Penalize CPR actions after ROSC (patient already revived)
        if p.heart_rate >= 0.9 and action in [4, 5]:
            return -2.0

        # Per-step living penalty (grows linearly with steps)
        step_penalty = -0.02 * (self._step_count / self.max_steps)

        already_done = self._actions_done

        # ── Protocol-sequenced rewards ────────────────────────────────────────

        if action == 0:  # ASSESS_CONSCIOUSNESS
            if stage == 0:
                r = 3.0 * scale
                self._protocol_stage = 1
                already_done.add(0)
            elif action in already_done:
                r = -1.0
            else:
                r = -0.8

        elif action == 1:  # CALL_EMERGENCY
            if stage == 1:
                r = 3.0 * scale
                self._protocol_stage = 2
                already_done.add(1)
            elif stage == 0:
                r = -0.5
            elif action in already_done:
                r = -1.0
            else:
                r = -0.5

        elif action == 10:  # TILT_HEAD_BACK
            if stage >= 2:
                if p.head_position < 0.9:
                    r = 1.5 * scale
                    p.head_position = min(1.0, p.head_position + 0.4)
                    p.airway_open = p.head_position > 0.5
                    # ── FIX 1 ────────────────────────────────────────────────
                    # Advance protocol stage when TILT_HEAD_BACK opens the
                    # airway, exactly as OPEN_AIRWAY does.  Previously this was
                    # missing: airway_open became True but stage stayed at 2,
                    # making CHECK_BREATHING permanently out-of-order and
                    # collapsing DQN into MONITOR_PULSE loops.
                    if p.airway_open and stage == 2:
                        self._protocol_stage = 3
                        already_done.add(10)
                    # ── END FIX 1 ────────────────────────────────────────────
                else:
                    r = -0.3  # already tilted
            else:
                r = -0.8

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
                    r = -0.3
            else:
                r = -0.8

        elif action == 3:  # CHECK_BREATHING
            if stage == 3:
                r = 1.5 * scale
                self._protocol_stage = 4
                already_done.add(3)
            elif stage > 3:
                r = -0.3
            else:
                r = -0.8

        elif action == 9:  # REPOSITION_HANDS
            if stage >= 4:
                if p.hand_placement < 1.0:
                    p.hand_placement = min(1.0, p.hand_placement + 0.5)
                    r = 1.0 * scale
                else:
                    r = -0.2
            else:
                r = -0.5

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
                r = -2.0

        elif action == 5:  # DELIVER_RESCUE_BREATHS
            if stage >= 5 and p.airway_open:
                p.breaths_delivered += 2
                p.chest_rise_rate = min(1.0, p.chest_rise_rate + 0.2 * scale)
                p.heart_rate = min(1.0, p.heart_rate + 0.05 * scale)
                r = 2.0 * scale
            elif not p.airway_open:
                r = -2.0
            else:
                r = -1.0

        elif action == 6:  # DEFIBRILLATE
            if p.heart_rate >= 0.85:
                r = -1.0
            elif stage >= 4 and p.compressions_delivered >= 30:
                p.heart_rate = min(1.0, p.heart_rate + 0.3 * scale)
                r = 4.0 * scale
                already_done.add(6)
            else:
                r = -2.0

        elif action == 7:  # MONITOR_PULSE
            if stage >= 4:
                r = 0.5
            elif stage >= 2:
                # ── FIX 4 ────────────────────────────────────────────────────
                # Raised from -0.2 to -0.5.  At stage 2, the old -0.2 was the
                # "least bad" option when the protocol was deadlocked (all other
                # actions gave -0.3 to -2.0).  That made MONITOR_PULSE the
                # rational minimum-loss choice for DQN, causing action collapse.
                r = -0.5
                # ── END FIX 4 ────────────────────────────────────────────────
            else:
                r = -0.5

        elif action == 8:  # RECOVERY_POSITION
            if p.heart_rate >= 0.7:
                p.recovery_position = True
                p.consciousness_level = min(1.0, p.consciousness_level + 0.3)
                r = 5.0 * scale
            else:
                r = -1.0

        elif action == 11:  # WAIT_OBSERVE
            r = -0.5 * (1.0 + p.time_without_action * 0.15)

        # ── ROSC terminal bonus ───────────────────────────────────────────────
        # ── FIX 2 ────────────────────────────────────────────────────────────
        # Original: required heart_rate >= 0.9 AND consciousness_level >= 0.7.
        # The episode terminates the moment heart_rate >= 0.9, so consciousness
        # had to already be >= 0.7 on the terminal step — an extremely narrow
        # training signal that DQN and REINFORCE almost never saw.
        # Fix: fire the big bonus on heart_rate >= 0.9 alone (+15.0 hits the
        # existing clip ceiling).  A supplemental +5.0 is retained for full
        # consciousness recovery as an additional incentive.
        if p.heart_rate >= 0.9:
            r += 15.0  # ROSC achieved
            if p.consciousness_level >= 0.7:
                r += 5.0  # full recovery bonus (will be clipped, but shapes gradient)
        # ── END FIX 2 ────────────────────────────────────────────────────────

        # ── Continuous HR decay without active intervention ───────────────────
        # ── FIX 3 ────────────────────────────────────────────────────────────
        # Reduced from 0.008 to 0.004 per step.
        # Old rate: 100 passive steps × 0.008 = 0.8 HR loss (cancels all gains).
        # New rate: 100 passive steps × 0.004 = 0.4 HR loss (gains can survive).
        if action not in [4, 5, 6]:
            p.heart_rate = max(0.0, p.heart_rate - 0.004)
        # ── END FIX 3 ────────────────────────────────────────────────────────

        r += step_penalty
        self._update_landmarks(action)

        return float(np.clip(r, -5.0, 15.0))

    def _apply_action(self, action: int):
        p = self._patient
        if action == 8 and p.recovery_position:
            p.landmarks = self._generate_recovery_pose()

    def _update_landmarks(self, action: int):
        p = self._patient
        if action == 10 or action == 2:
            p.landmarks[1] = max(0.1, p.landmarks[1] - 0.05 * p.head_position)
        elif action == 4:
            p.landmarks[33] = min(1.0, p.landmarks[33] + 0.02)
            p.landmarks[36] = min(1.0, p.landmarks[36] + 0.02)

    def _generate_recovery_pose(self) -> np.ndarray:
        base = self._generate_collapse_pose()
        for i in range(0, len(base), 3):
            base[i] = np.clip(base[i] + 0.15, 0.0, 1.0)
        return base