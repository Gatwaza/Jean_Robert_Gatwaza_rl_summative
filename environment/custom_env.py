"""
CPR Position Assessment Environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Deque

# Patient State

@dataclass
class PatientState:
    landmarks:            np.ndarray = field(default_factory=lambda: np.zeros(51))
    heart_rate:           float = 0.0
    chest_rise_rate:      float = 0.0
    airway_open:          bool  = False
    head_position:        float = 0.0
    hand_placement:       float = 0.0
    compression_depth:    float = 0.0
    recovery_position:    bool  = False
    consciousness_level:  float = 0.0
    compressions_delivered: int = 0
    breaths_delivered:    int   = 0
    time_without_action:  int   = 0


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
OBS_DIM   = 56   # 51 landmarks + 2 vitals + 1 stage norm + 2 temporal


# Environment

class CPREnv(gym.Env):
    """
    Gymnasium CPR environment — exploit-proof reward shaping.

    Parameters
    ----------
    max_steps   : max steps per episode        (default 200)
    noise_std   : landmark jitter std          (default 0.02)
    difficulty  : 'easy' | 'medium' | 'hard'
    render_mode : 'human' | 'rgb_array' | None
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Consecutive-use caps: key = action_id, value = max consecutive uses allowed
    CONSECUTIVE_CAPS: Dict[int, int] = {
        7:  3,   # MONITOR_PULSE  — 3 max, then penalise (primary exploit)
        11: 2,   # WAIT_OBSERVE   — 2 max
        10: 2,   # TILT_HEAD_BACK — only useful once, diminishing returns fast
        3:  2,   # CHECK_BREATHING
    }

    def __init__(
        self,
        max_steps:   int   = 200,
        noise_std:   float = 0.02,
        difficulty:  str   = "medium",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.max_steps   = max_steps
        self.noise_std   = noise_std
        self.difficulty  = difficulty
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._set_difficulty_params(difficulty)
        self._reset_state()

        self._renderer = None
        self._episode  = 0

    # Difficulty

    def set_difficulty(self, difficulty: str):
        """Hot-swap difficulty without resetting. Use for curriculum schedules."""
        self.difficulty = difficulty
        self._set_difficulty_params(difficulty)

    def _set_difficulty_params(self, difficulty: str):
        d = difficulty.lower()
        if d == "easy":
            self._diff_scale       = 1.5
            self._hr_decay         = 0.004
            self._rosc_threshold   = 0.88
            self._comp_depth_base  = 0.55
            self._sustain_required = 1
            self._comp_hr_gain     = 0.26
            self._breath_hr_gain   = 0.04
        elif d == "hard":
            self._diff_scale       = 0.70
            self._hr_decay         = 0.012
            self._rosc_threshold   = 0.92
            self._comp_depth_base  = 0.38
            self._sustain_required = 3
            self._comp_hr_gain     = 0.18
            self._breath_hr_gain   = 0.02
        else:  # medium
            self._diff_scale       = 1.0
            self._hr_decay         = 0.008
            self._rosc_threshold   = 0.90
            self._comp_depth_base  = 0.45
            self._sustain_required = 2
            self._comp_hr_gain     = 0.22   # ↑ from 0.18 — primary driver
            self._breath_hr_gain   = 0.03   # ↓ from 0.08 — support role

    # Episode-level state

    def _reset_state(self):
        self._step_count        = 0
        self._protocol_stage    = 0
        self._cumulative_reward = 0.0
        self._actions_done: set = set()
        self._action_history    = [-1, -1, -1]
        self._repeat_counts:    Dict[int, int] = {}
        self._last_action       = -1
        self._consecutive_count = 0
        self._recent_actions:   deque = deque(maxlen=10)
        self._last_comp_step:   int   = 0   # step-count rhythm (replaces wall-clock)
        self._comp_intervals:   list  = []
        self._comp_streak       = 0
        self._rosc_steps        = 0

    # Gymnasium API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode += 1
        self._reset_state()
        self._patient = PatientState(
            landmarks = self._generate_collapse_pose()
        )
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step_count += 1

        # Consecutive tracking
        if action == self._last_action:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
        self._last_action = action
        self._recent_actions.append(action)

        reward = self._compute_reward(action)
        self._apply_action(action)
        self._action_history = [action] + self._action_history[:2]
        self._patient.time_without_action = (
            0 if action != 11 else self._patient.time_without_action + 1
        )
        self._cumulative_reward += reward

        terminated = self._is_terminal()
        truncated  = self._step_count >= self.max_steps
        info = self._get_info()
        info["action_name"] = ACTION_NAMES[action]
        info["reward"]      = reward
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            from rendering import CPRRenderer
            self._renderer = CPRRenderer()
        self._renderer.render(
            self._patient, self._step_count, self._cumulative_reward,
            last_action=self._action_history[0] if self._action_history else -1,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # Reward v4

    def _compute_reward(self, action: int) -> float:
        p, stage, scale, done = (
            self._patient, self._protocol_stage, self._diff_scale, self._actions_done
        )
        r = -0.01 * (self._step_count / self.max_steps)   # living cost

        # ── CONSECUTIVE-USE EXPLOIT BLOCK ─────────────────────────────────────
        cap = self.CONSECUTIVE_CAPS.get(action, 0)
        if cap > 0 and self._consecutive_count > cap:
            over = self._consecutive_count - cap
            penalty = -0.6 * min(over, 6)
            # For MONITOR_PULSE specifically — hard block after cap
            if action == 7:
                r += penalty + self._diversity_bonus()
                return float(np.clip(r, -6.0, 22.0))
            r += penalty

        # Repeat penalty
        self._repeat_counts[action] = self._repeat_counts.get(action, 0) + 1
        n_reps = self._repeat_counts[action]
        if action in done and n_reps > 1:
            r -= 0.35 * min(n_reps - 1, 5)

        # Protocol rewards
        if action == 0:   # ASSESS_CONSCIOUSNESS
            if stage == 0:
                r += 3.0 * scale; self._protocol_stage = 1; done.add(0)
            elif 0 in done: r -= 0.8
            else:           r -= 0.5

        elif action == 1:   # CALL_EMERGENCY
            if stage == 1:
                r += 3.0 * scale; self._protocol_stage = 2; done.add(1)
            elif stage == 0: r -= 0.8
            elif 1 in done:  r -= 0.6
            else:            r -= 0.4

        elif action == 10:  # TILT_HEAD_BACK
            if stage >= 2:
                if p.head_position < 0.9:
                    r += 1.5 * scale * (1 - p.head_position)
                    p.head_position = min(1.0, p.head_position + 0.4)
                    p.airway_open   = p.head_position > 0.5
                else:
                    r -= 0.3
            else:
                r -= 0.8

        elif action == 2:   # OPEN_AIRWAY
            if stage >= 2:
                if not p.airway_open:
                    r += 1.5 * scale
                    p.head_position = min(1.0, p.head_position + 0.5)
                    p.airway_open   = True
                    if stage == 2:
                        self._protocol_stage = 3; done.add(2)
                else:
                    r -= 0.3
            else:
                r -= 0.8

        elif action == 3:   # CHECK_BREATHING
            if stage == 3:
                r += 1.5 * scale; self._protocol_stage = 4; done.add(3)
            elif stage > 3: r -= 0.25
            else:           r -= 0.8

        elif action == 9:   # REPOSITION_HANDS
            if stage >= 4:
                if p.hand_placement < 1.0:
                    r += 1.0 * scale * (1 - p.hand_placement)
                    p.hand_placement = min(1.0, p.hand_placement + 0.5)
                else:
                    r -= 0.2
            else:
                r -= 0.5

        elif action == 4:   # BEGIN_CHEST_COMPRESSIONS — PRIMARY HR DRIVER
            if stage >= 4:
                depth = min(1.0, self._comp_depth_base + 0.5 * p.hand_placement)
                p.compression_depth      = depth
                p.compressions_delivered += 30

                # Streak bonus: reward continued compressions (clinical 30:2 rhythm)
                self._comp_streak += 1
                streak_mult = min(1.0 + 0.08 * (self._comp_streak - 1), 1.5)
                hr_gain = self._comp_hr_gain * depth * scale * streak_mult
                p.heart_rate = min(1.0, p.heart_rate + hr_gain)
                r += 3.5 * depth * scale

                # Compression rhythm bonus: reward consistent step-spacing.
                # In a 200-step episode representing ~2 min of CPR, each step
                # maps to ≈ 0.6 s.  30 compressions at 100 bpm → ~18 s → ≈ 30
                # steps.  One action per call means a "good rhythm" is 1 step
                # between consecutive BEGIN_CHEST_COMPRESSIONS calls.
                # We score by step-gap to the previous compression:
                #   gap == 1  → ideal (100-120 bpm equivalent)  +1.0
                #   gap == 2  → acceptable                       +0.4
                #   other     → too fast / too slow              −0.5
                if self._last_comp_step > 0:
                    step_gap = self._step_count - self._last_comp_step
                    self._comp_intervals.append(step_gap)
                    if len(self._comp_intervals) > 5:
                        self._comp_intervals.pop(0)
                    avg_gap = float(np.mean(self._comp_intervals))
                    if   avg_gap <= 1.5: r += 1.0 * scale
                    elif avg_gap <= 2.5: r += 0.4 * scale
                    else:                r -= 0.5
                self._last_comp_step = self._step_count

                if stage == 4:
                    self._protocol_stage = 5
            else:
                self._comp_streak = 0
                r -= 2.0

        elif action == 5:   # DELIVER_RESCUE_BREATHS — SUPPORT ONLY
            self._comp_streak = 0   # breaks compression streak intentionally
            if stage >= 5 and p.airway_open:
                p.breaths_delivered  += 2
                p.chest_rise_rate     = min(1.0, p.chest_rise_rate + 0.25 * scale)
                # Breath HR gain intentionally small — policy must use compressions
                p.heart_rate          = min(1.0, p.heart_rate + self._breath_hr_gain * scale)
                r += 2.5 * scale
            elif stage >= 5:
                r -= 2.0   # airway closed → dangerous
            else:
                r -= 1.0

        elif action == 6:   # DEFIBRILLATE
            self._comp_streak = 0
            if p.heart_rate >= 0.85:
                r -= 1.0
            elif stage >= 4 and p.compressions_delivered >= 30:
                p.heart_rate = min(1.0, p.heart_rate + 0.35 * scale)
                r += 4.5 * scale; done.add(6)
            elif stage >= 4:
                p.heart_rate = min(1.0, p.heart_rate + 0.20 * scale)
                r += 2.0 * scale
            else:
                r -= 2.0

        elif action == 7:   # MONITOR_PULSE (within cap — handled above for over-cap)
            r += 0.5 if stage >= 4 else (-0.2 if stage >= 2 else -0.5)

        elif action == 8:   # RECOVERY_POSITION
            if 8 in done:
                r -= 1.5
            elif p.heart_rate >= 0.7:
                p.recovery_position   = True
                p.consciousness_level = min(1.0, p.consciousness_level + 0.3)
                r += 5.0 * scale; done.add(8)
            else:
                r -= 1.0

        elif action == 11:  # WAIT_OBSERVE
            r -= 0.5 * (1.0 + p.time_without_action * 0.15)

        # ROSC terminal bonus
        # Reduced from 20.0 → 8.0 and clip tightened to [-3, 10] to keep the
        # Q-value range compact and prevent DQN overestimation of rare bonuses.
        if p.heart_rate >= self._rosc_threshold and p.consciousness_level >= 0.7:
            r += 8.0

        # HR decay when not actively treating
        if action not in (4, 5, 6):
            p.heart_rate = max(0.0, p.heart_rate - self._hr_decay)
            if action != 4:
                self._comp_streak = max(0, self._comp_streak - 1)

        # ── Diversity bonus (collapse prevention) ──────────────────────────────
        r += self._diversity_bonus()

        self._update_landmarks(action)
        return float(np.clip(r, -3.0, 10.0))

    def _diversity_bonus(self) -> float:
        """Reward for using varied actions in the last 10 steps."""
        if len(self._recent_actions) < 3:
            return 0.0
        unique = len(set(self._recent_actions))
        return 0.05 * max(0, unique - 1)   # 0 for 1 unique → 0.25 for 6+ unique

    # Terminal

    def _is_terminal(self) -> bool:
        p = self._patient
        if p.heart_rate >= self._rosc_threshold:
            self._rosc_steps += 1
            if self._rosc_steps >= self._sustain_required:
                p.consciousness_level = min(1.0, p.consciousness_level + 0.4)
                return True
        else:
            self._rosc_steps = 0
        if p.time_without_action > 12:     return True
        if self._cumulative_reward < -120: return True
        return False

    # Observation

    def _get_obs(self) -> np.ndarray:
        p     = self._patient
        stage = self._protocol_stage / 5.0
        hist  = [(a + 1) / N_ACTIONS if a >= 0 else 0.0
                 for a in self._action_history[:2]]
        base = np.concatenate([
            p.landmarks,
            [p.heart_rate, p.chest_rise_rate],
            [stage],
            hist,
        ]).astype(np.float32)
        noise = self.np_random.normal(0, self.noise_std, size=base.shape).astype(np.float32)
        return np.clip(base + noise, 0.0, 1.0)

    def _get_info(self) -> Dict[str, Any]:
        p = self._patient
        return {
            "step":              self._step_count,
            "protocol_stage":    self._protocol_stage,
            "heart_rate":        p.heart_rate,
            "compressions":      p.compressions_delivered,
            "breaths":           p.breaths_delivered,
            "airway_open":       p.airway_open,
            "cumulative_reward": self._cumulative_reward,
            "difficulty":        self.difficulty,
            "comp_streak":       self._comp_streak,
        }

    # Action effects

    def _apply_action(self, action: int):
        p = self._patient
        if action == 8 and p.recovery_position:
            p.landmarks = self._generate_recovery_pose()

    def _update_landmarks(self, action: int):
        p = self._patient
        if action in (2, 10):
            p.landmarks[1] = max(0.1, p.landmarks[1] - 0.05 * p.head_position)
        elif action == 4:
            p.landmarks[33] = min(1.0, p.landmarks[33] + 0.02)
            p.landmarks[36] = min(1.0, p.landmarks[36] + 0.02)

    # Pose generators

    def _generate_collapse_pose(self) -> np.ndarray:
        rng = self.np_random if hasattr(self, "np_random") else np.random
        base = np.array([
            0.50, 0.82, 0.95,
            0.53, 0.80, 0.90,  0.56, 0.79, 0.88,  0.58, 0.78, 0.85,
            0.47, 0.80, 0.90,  0.44, 0.79, 0.88,  0.42, 0.78, 0.85,
            0.60, 0.81, 0.80,  0.40, 0.81, 0.80,
            0.54, 0.84, 0.90,  0.46, 0.84, 0.90,
            0.60, 0.72, 0.95,  0.40, 0.72, 0.95,
            0.65, 0.60, 0.90,  0.35, 0.60, 0.90,
            0.62, 0.50, 0.85,  0.38, 0.50, 0.85,
        ], dtype=np.float32)
        noise = rng.normal(0, 0.015, size=base.shape).astype(np.float32)
        return np.clip(base + noise, 0.0, 1.0)

    def _generate_recovery_pose(self) -> np.ndarray:
        base = self._generate_collapse_pose()
        for i in range(0, len(base), 3):
            base[i] = np.clip(base[i] + 0.15, 0.0, 1.0)
        return base