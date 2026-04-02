"""
CPR Simulation Renderer  (v4 — Professional Edition)
======================================================
Cinematic 2D visualisation: scrolling ECG heartbeat, animated compression
depth meter, phase banner transitions, ROSC particle burst, action history
timeline, and a clean clinical dark-theme dashboard.

Layout (1280 × 720):
┌──────────────────────────────┬──────────┐
│  PHASE BANNER (full width)   │          │
├────────────────┬─────────────┤ VITALS   │
│                │ ECG scroll  │  PANEL   │
│  Patient scene │─────────────│          │
│  + skeleton    │ Action log  │          │
│                │─────────────│          │
├────────────────┴─────────────┤          │
│  Reward curve (full sim-w)   │          │
└──────────────────────────────┴──────────┘
"""

import numpy as np
import math
import sys
import colorsys
from typing import Optional, Tuple, List

try:
    import pygame
    import pygame.gfxdraw
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ── MediaPipe skeleton connections ────────────────────────────────────────────
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (7, 9), (8, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":           (12, 14, 22),
    "panel_bg":     (18, 21, 34),
    "panel_border": (32, 38, 65),
    "ground":       (30, 36, 52),
    "ground_mat":   (22, 44, 70),
    "patient_skin": (210, 168, 128),
    "shirt":        (48, 110, 190),
    "trouser":      (50, 50, 65),
    "skeleton":     (60, 220, 170),
    "joint":        (255, 220, 60),
    "joint_lo":     (180, 140, 40),
    "text_hi":      (230, 235, 245),
    "text_mid":     (150, 158, 185),
    "text_lo":      (80, 90, 115),
    "bar_bg":       (28, 33, 52),
    "green":        (50, 210, 110),
    "amber":        (235, 168, 35),
    "red":          (215, 65, 65),
    "blue":         (55, 140, 240),
    "ecg_line":     (55, 220, 130),
    "ecg_glow":     (20, 80, 50),
    "rosc_gold":    (255, 210, 50),
    "reward_pos":   (60, 210, 110),
    "reward_neg":   (215, 65, 65),
    "comp_fill":    (55, 200, 240),
    "separator":    (35, 42, 68),

    # Phase colours
    "phase_random":   (70, 75, 115),
    "phase_training": (30, 65, 130),
    "phase_demo":     (20, 100, 55),
}

ACTION_SHORT = [
    "ASSESS",  "CALL 911",  "AIRWAY",    "CHK BREATH",
    "COMPRESS","RSC BREATH","DEFIB",     "MON PULSE",
    "RECOVERY","REPOSITION","TILT HEAD", "WAIT",
]

ACTION_ICONS = ["👁", "📞", "💨", "👂", "👐", "💋", "⚡", "💓", "↩", "✋", "🤕", "⏸"]

W, H         = 1280, 720
PANEL_W      = 290   # right vitals panel
SIM_W        = W - PANEL_W
BANNER_H     = 42
ECG_H        = 110
ACTION_LOG_H = 85
SCENE_H      = H - BANNER_H - ECG_H - ACTION_LOG_H - 15  # scene area height
REWARD_H     = 0   # embedded inside scene bottom stripe


# ══════════════════════════════════════════════════════════════════════════════
class CPRRenderer:
    """
    High-fidelity Pygame renderer for the CPR RL environment.

    Designed for both headless screenshot capture (rgb_array mode)
    and interactive display (human mode).
    """

    def __init__(self):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not installed.  pip install pygame")

        pygame.init()
        pygame.display.set_caption("CPR  ·  Reinforcement Learning  ·  v4")
        self.screen  = pygame.display.set_mode((W, H))
        self.surface = pygame.Surface((W, H))
        self.clock   = pygame.time.Clock()

        # Fonts — use system fallbacks gracefully
        def _font(size, bold=False):
            for name in ("DejaVu Sans", "Helvetica Neue", "Arial", ""):
                try:
                    return pygame.font.SysFont(name, size, bold=bold)
                except Exception:
                    pass
            return pygame.font.Font(None, size)

        self._f_xl   = _font(26, True)
        self._f_lg   = _font(20, True)
        self._f_md   = _font(15)
        self._f_sm   = _font(12)
        self._f_xs   = _font(10)

        # State buffers
        self._reward_history:  List[float] = []
        self._action_history:  List[Tuple[int, float]] = []   # (action, reward)
        self._ecg_buffer:      List[float] = []               # raw y values
        self._frame            = 0
        self._rosc_particles:  List[dict]  = []
        self._phase            = "random"
        self._phase_banner_t   = 0.0     # 0→1 reveal animation
        self._banner_color     = C["phase_random"]
        self._prev_hr          = 0.0

        # Pre-build ECG waveform template (one beat at 120bpm)
        self._ecg_template     = _build_ecg_template()
        self._ecg_pos          = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(self, patient, step: int, cumulative_reward: float,
               phase: str = "random", algorithm: str = "",
               experiment: int = 0, last_action: int = -1,
               last_reward: float = 0.0):

        self._frame += 1
        dt = 1.0 / max(self.clock.get_fps(), 1.0)

        # ── Update animated state ─────────────────────────────────────────────
        self._update_ecg(patient.heart_rate, dt)
        self._update_reward(cumulative_reward)
        self._update_action(last_action, last_reward)
        self._update_phase(phase)
        self._update_particles(dt)

        # ── Draw all layers ───────────────────────────────────────────────────
        self.surface.fill(C["bg"])

        self._draw_phase_banner(phase, algorithm, experiment)
        self._draw_scene(patient)
        self._draw_ecg_panel(patient)
        self._draw_action_log()
        self._draw_vitals_panel(patient, step, cumulative_reward, algorithm)
        self._draw_reward_curve()
        self._draw_rosc_particles()
        self._draw_compression_overlay(patient)

        # ROSC overlay
        if patient.heart_rate >= 0.9:
            self._draw_rosc_glory()

        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(30)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close(); sys.exit()

    def close(self):
        pygame.quit()

    def get_rgb_array(self) -> np.ndarray:
        return pygame.surfarray.array3d(self.surface).transpose(1, 0, 2)

    # ── Phase banner ───────────────────────────────────────────────────────────

    def _update_phase(self, phase: str):
        if phase != self._phase:
            self._phase = phase
            self._phase_banner_t = 0.0
            target = {
                "random":   C["phase_random"],
                "training": C["phase_training"],
                "demo":     C["phase_demo"],
            }.get(phase, C["phase_random"])
            self._banner_color = _lerp_color(self._banner_color, target, 0.15)
        else:
            self._banner_color = _lerp_color(
                self._banner_color,
                {
                    "random":   C["phase_random"],
                    "training": C["phase_training"],
                    "demo":     C["phase_demo"],
                }.get(phase, C["phase_random"]),
                0.04,
            )
        self._phase_banner_t = min(1.0, self._phase_banner_t + 0.06)

    def _draw_phase_banner(self, phase: str, algo: str, exp: int):
        # Animated slide-in from top
        slide = _ease_out(self._phase_banner_t)
        y_off = int((1 - slide) * -(BANNER_H + 4))

        banner_rect = pygame.Rect(0, y_off, W, BANNER_H)
        pygame.draw.rect(self.surface, self._banner_color, banner_rect)

        # Bottom glow stripe
        _draw_hline_glow(self.surface, BANNER_H + y_off - 1, W,
                         _lighten(self._banner_color, 0.4), width=2)

        phase_labels = {
            "random":   "◈  EXPLORING  ·  Random Agent",
            "training": f"◉  TRAINING   ·  {algo}   Experiment {exp}/10",
            "demo":     f"★  DEMO       ·  {algo}   Best Model",
        }
        label = phase_labels.get(phase, phase.upper())
        txt = self._f_lg.render(label, True, C["text_hi"])
        self.surface.blit(txt, (18, y_off + (BANNER_H - txt.get_height()) // 2))

        # Right-side pulse indicator
        pulse_col = C["green"] if self._frame % 30 < 15 else _darken(C["green"], 0.5)
        pygame.draw.circle(self.surface, pulse_col,
                           (W - PANEL_W - 20, y_off + BANNER_H // 2), 6)

    # ── Main scene ─────────────────────────────────────────────────────────────

    def _draw_scene(self, patient):
        y0 = BANNER_H
        scene_rect = pygame.Rect(0, y0, SIM_W, SCENE_H)
        pygame.draw.rect(self.surface, C["panel_bg"], scene_rect)

        cx = SIM_W // 2
        cy = y0 + SCENE_H // 2 + 20

        # Floor
        _draw_rounded_rect(self.surface, C["ground"],
                           pygame.Rect(cx - 260, cy - 20, 520, 180), 12)
        # CPR mat
        _draw_rounded_rect(self.surface, C["ground_mat"],
                           pygame.Rect(cx - 165, cy - 10, 330, 160), 8)
        mat_lbl = self._f_xs.render("CPR TRAINING MAT", True, C["text_lo"])
        self.surface.blit(mat_lbl, (cx - mat_lbl.get_width() // 2, cy + 150))

        # Patient body
        self._draw_patient_body(patient, cx, cy)

        # Skeleton overlay
        self._draw_skeleton(patient, cx, cy)

        # Corner label
        lbl = self._f_sm.render("SCENE VIEW", True, C["text_lo"])
        self.surface.blit(lbl, (8, y0 + 8))

        # Separator
        pygame.draw.line(self.surface, C["separator"],
                         (0, y0 + SCENE_H), (SIM_W, y0 + SCENE_H), 1)

    def _draw_patient_body(self, patient, cx, cy):
        skin = C["patient_skin"]
        hr   = patient.heart_rate

        # Legs (lying down = horizontal bars)
        for dx, dz in [(-20, 0), (20, 45)]:
            pygame.draw.rect(self.surface, C["trouser"],
                             pygame.Rect(cx - 35 + dx, cy + 85, 28, 75), border_radius=6)
            pygame.draw.rect(self.surface, skin,
                             pygame.Rect(cx - 35 + dx, cy + 145, 28, 30), border_radius=4)

        # Torso
        _draw_rounded_rect(self.surface, C["shirt"],
                           pygame.Rect(cx - 42, cy + 8, 84, 90), 10)

        # Compression wave on chest
        if patient.compression_depth > 0.2:
            d = int(patient.compression_depth * 18)
            col = _lerp_color(C["amber"], C["green"],
                              min(1.0, patient.compression_depth * 1.5))
            _draw_rounded_rect(self.surface, col,
                               pygame.Rect(cx - 22, cy + 28, 44, d), 4)
            # Depth label
            dep_lbl = self._f_xs.render(f"{patient.compression_depth*6:.1f}cm", True, col)
            self.surface.blit(dep_lbl, (cx + 26, cy + 28))

        # Head tilt animation
        head_y = cy - 52 + int(-22 * patient.head_position)
        pygame.draw.ellipse(self.surface, skin,
                            (cx - 24, head_y, 48, 54))
        # Eyes — closed (unconscious)
        pygame.draw.line(self.surface, _darken(skin, 0.4),
                         (cx - 12, head_y + 22), (cx - 4,  head_y + 22), 2)
        pygame.draw.line(self.surface, _darken(skin, 0.4),
                         (cx + 4,  head_y + 22), (cx + 12, head_y + 22), 2)

        # Airway indicator
        if patient.airway_open:
            pygame.draw.arc(self.surface, C["ecg_line"],
                            (cx - 18, head_y + 38, 36, 16), 0, math.pi, 2)

        # Arms
        for side, sign in [(-1, -1), (1, 1)]:
            ax = cx + sign * 50
            pygame.draw.rect(self.surface, skin,
                             pygame.Rect(ax - 9, cy + 18, 18, 50), border_radius=5)
            pygame.draw.rect(self.surface, skin,
                             pygame.Rect(ax - 8, cy + 62, 16, 28), border_radius=4)

        # Recovery position indicator
        if patient.recovery_position:
            rlbl = self._f_md.render("↩  RECOVERY POSITION", True, C["green"])
            self.surface.blit(rlbl, (cx - rlbl.get_width() // 2, BANNER_H + 14))

        # HR tint overlay — subtle pulse on skin
        if hr > 0.3 and self._frame % 24 < 4:
            pulse_surf = pygame.Surface((48, 54), pygame.SRCALPHA)
            pulse_surf.fill((*C["green"], int(30 * hr)))
            self.surface.blit(pulse_surf, (cx - 24, head_y))

    def _draw_skeleton(self, patient, cx, cy):
        lm = patient.landmarks
        sx = SIM_W * 0.50
        sy = SCENE_H * 0.52

        def pt(i):
            x = lm[i*3]   * sx + cx - sx//2
            y = lm[i*3+1] * sy + BANNER_H + 30
            v = lm[i*3+2]
            return int(x), int(y), v

        pts = [pt(i) for i in range(17)]

        # Connections
        for a, b in SKELETON_CONNECTIONS:
            if a >= len(pts) or b >= len(pts): continue
            ax, ay, av = pts[a]
            bx, by, bv = pts[b]
            vis = min(av, bv)
            if vis < 0.2: continue
            alpha = int(vis * 200)
            col = _alpha_blend(C["skeleton"], C["bg"], vis * 0.85)
            pygame.draw.line(self.surface, col, (ax, ay), (bx, by), 2)

        # Joints
        for x, y, v in pts:
            if v < 0.25: continue
            r = max(3, int(v * 5))
            col = _lerp_color(C["joint_lo"], C["joint"], v)
            pygame.gfxdraw.filled_circle(self.surface, x, y, r, (*col, int(v * 220)))
            pygame.gfxdraw.aacircle(self.surface, x, y, r, (*col, 180))

    # ── ECG Panel ──────────────────────────────────────────────────────────────

    def _update_ecg(self, heart_rate: float, dt: float):
        # Speed proportional to HR: faster at higher HR
        bpm = max(10.0, heart_rate * 80 + 20)
        beats_per_sec = bpm / 60.0
        samples_per_frame = beats_per_sec * len(self._ecg_template) / 30.0

        self._ecg_pos = (self._ecg_pos + samples_per_frame) % len(self._ecg_template)

        # Emit next sample
        idx = int(self._ecg_pos)
        raw = self._ecg_template[idx] * heart_rate
        self._ecg_buffer.append(raw)
        if len(self._ecg_buffer) > SIM_W - 10:
            self._ecg_buffer.pop(0)

    def _draw_ecg_panel(self, patient):
        y0 = BANNER_H + SCENE_H
        rect = pygame.Rect(0, y0, SIM_W, ECG_H)
        pygame.draw.rect(self.surface, C["panel_bg"], rect)

        # Title
        lbl = self._f_sm.render("ECG  ·  CARDIAC MONITOR", True, C["text_lo"])
        self.surface.blit(lbl, (8, y0 + 5))

        hr_pct = int(patient.heart_rate * 100)
        hr_col = C["green"] if hr_pct >= 60 else C["amber"] if hr_pct >= 30 else C["red"]
        hr_lbl = self._f_lg.render(f"{hr_pct}%", True, hr_col)
        self.surface.blit(hr_lbl, (SIM_W - 60, y0 + 6))

        # Glow background strip
        glow_surf = pygame.Surface((SIM_W, ECG_H - 22), pygame.SRCALPHA)
        glow_surf.fill((0, 0, 0, 0))
        pygame.draw.rect(glow_surf, (*C["ecg_glow"], 60),
                         pygame.Rect(0, 0, SIM_W, ECG_H - 22))
        self.surface.blit(glow_surf, (0, y0 + 20))

        # Grid lines
        ecg_y0 = y0 + 22
        ecg_h  = ECG_H - 28
        mid_y  = ecg_y0 + ecg_h // 2
        for gx in range(0, SIM_W, 60):
            pygame.draw.line(self.surface, C["panel_border"],
                             (gx, ecg_y0), (gx, ecg_y0 + ecg_h), 1)
        pygame.draw.line(self.surface, C["separator"],
                         (0, mid_y), (SIM_W, mid_y), 1)

        # ECG waveform
        buf = self._ecg_buffer
        if len(buf) < 2:
            return
        w_each = (SIM_W - 10) / max(len(buf), 1)
        pts = []
        for i, v in enumerate(buf):
            x = int(i * w_each) + 5
            y = int(mid_y - v * (ecg_h * 0.42))
            pts.append((x, y))

        if len(pts) > 1:
            # Glow pass (thick, transparent)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    glow_pts = [(p[0]+dx, p[1]+dy) for p in pts]
                    pygame.draw.lines(self.surface,
                                      (*C["ecg_glow"], 40), False, glow_pts, 3)
            pygame.draw.lines(self.surface, C["ecg_line"], False, pts, 2)

        # Separator below
        pygame.draw.line(self.surface, C["separator"],
                         (0, y0 + ECG_H - 1), (SIM_W, y0 + ECG_H - 1), 1)

    # ── Action History Log ─────────────────────────────────────────────────────

    def _update_action(self, action: int, reward: float):
        if action < 0: return
        self._action_history.append((action, reward))
        if len(self._action_history) > 12:
            self._action_history.pop(0)

    def _draw_action_log(self):
        y0 = BANNER_H + SCENE_H + ECG_H
        rect = pygame.Rect(0, y0, SIM_W, ACTION_LOG_H)
        pygame.draw.rect(self.surface, C["bg"], rect)

        lbl = self._f_sm.render("ACTION HISTORY", True, C["text_lo"])
        self.surface.blit(lbl, (8, y0 + 5))

        if not self._action_history:
            return

        slot_w = min(90, (SIM_W - 10) // max(len(self._action_history), 1))
        for i, (a, r) in enumerate(reversed(self._action_history[-10:])):
            x = SIM_W - 10 - (i + 1) * (slot_w + 4)
            if x < 0: break
            alpha = max(40, 255 - i * 22)
            col = C["green"] if r >= 0 else C["red"]
            box_col = _darken(col, 0.72)

            _draw_rounded_rect(self.surface, box_col,
                               pygame.Rect(x, y0 + 20, slot_w, ACTION_LOG_H - 28), 5)
            name = ACTION_SHORT[a] if a < len(ACTION_SHORT) else f"ACT{a}"
            t = self._f_xs.render(name, True, _lighten(col, 0.5))
            self.surface.blit(t, (x + (slot_w - t.get_width()) // 2, y0 + 26))
            rsign = f"+{r:.1f}" if r >= 0 else f"{r:.1f}"
            rt = self._f_xs.render(rsign, True, col)
            self.surface.blit(rt, (x + (slot_w - rt.get_width()) // 2, y0 + 43))

    # ── Vitals Panel ───────────────────────────────────────────────────────────

    def _draw_vitals_panel(self, patient, step: int, cum_rew: float, algo: str):
        px = SIM_W
        panel_rect = pygame.Rect(px, 0, PANEL_W, H)
        pygame.draw.rect(self.surface, C["panel_bg"], panel_rect)
        pygame.draw.line(self.surface, C["panel_border"],
                         (px, 0), (px, H), 2)

        y = BANNER_H + 8

        # Title
        title = self._f_lg.render("VITAL MONITOR", True, C["blue"])
        self.surface.blit(title, (px + (PANEL_W - title.get_width()) // 2, y)); y += 32

        # Step / Algorithm
        self._kv(px, y, "STEP",  str(step)); y += 20
        self._kv(px, y, "ALGO",  algo or "—"); y += 20
        self._kv(px, y, "SCORE", f"{cum_rew:.1f}"); y += 28

        pygame.draw.line(self.surface, C["separator"],
                         (px + 8, y), (px + PANEL_W - 8, y), 1); y += 10

        # Vital bars
        self._vbar(px, y, "HEART RATE",    patient.heart_rate,          C["red"]);   y += 50
        self._vbar(px, y, "CHEST RISE",    patient.chest_rise_rate,     C["blue"]);  y += 50
        self._vbar(px, y, "HAND PLACEMENT",patient.hand_placement,      C["amber"]); y += 50
        self._vbar(px, y, "CONSCIOUSNESS", patient.consciousness_level, C["green"]); y += 50

        pygame.draw.line(self.surface, C["separator"],
                         (px + 8, y), (px + PANEL_W - 8, y), 1); y += 10

        # Airway
        aw_col = C["green"] if patient.airway_open else C["red"]
        aw_txt = "AIRWAY:  OPEN ✓" if patient.airway_open else "AIRWAY:  CLOSED ✗"
        t = self._f_md.render(aw_txt, True, aw_col)
        self.surface.blit(t, (px + 10, y)); y += 24

        # Compressions / Breaths
        self._kv(px, y, "COMPRESSIONS", str(patient.compressions_delivered)); y += 20
        self._kv(px, y, "BREATHS",      str(patient.breaths_delivered));      y += 20
        self._kv(px, y, "RECOVERY",
                 "YES ✓" if patient.recovery_position else "NO"); y += 28

        pygame.draw.line(self.surface, C["separator"],
                         (px + 8, y), (px + PANEL_W - 8, y), 1); y += 12

        # ROSC indicator (big, flashing when achieved)
        if patient.heart_rate >= 0.9:
            flash = self._frame % 20 < 10
            col = C["rosc_gold"] if flash else _darken(C["rosc_gold"], 0.6)
            rosc_bg = pygame.Surface((PANEL_W - 20, 40), pygame.SRCALPHA)
            rosc_bg.fill((*_darken(C["rosc_gold"], 0.85), 180))
            self.surface.blit(rosc_bg, (px + 10, y))
            rlbl = self._f_lg.render("★  ROSC  ★", True, col)
            self.surface.blit(rlbl,
                              (px + (PANEL_W - rlbl.get_width()) // 2, y + 8))
        else:
            # Compression rate hint
            hint = self._f_xs.render("TARGET: 100–120 bpm · 5–6 cm depth", True, C["text_lo"])
            self.surface.blit(hint, (px + 4, y))

    def _vbar(self, px: int, y: int, label: str, value: float, color: tuple):
        bw = PANEL_W - 22
        lbl = self._f_xs.render(label, True, C["text_lo"])
        self.surface.blit(lbl, (px + 11, y))

        bg = pygame.Rect(px + 11, y + 15, bw, 13)
        pygame.draw.rect(self.surface, C["bar_bg"], bg, border_radius=5)

        fill_w = max(0, int(value * bw))
        if fill_w > 0:
            col = color if value >= 0.4 else C["red"]
            fill = pygame.Rect(px + 11, y + 15, fill_w, 13)
            pygame.draw.rect(self.surface, col, fill, border_radius=5)

        val_t = self._f_xs.render(f"{value:.0%}", True, C["text_mid"])
        self.surface.blit(val_t, (px + 11 + bw - val_t.get_width() - 2, y))

    def _kv(self, px: int, y: int, key: str, val: str):
        k = self._f_xs.render(key + ": ", True, C["text_lo"])
        v = self._f_sm.render(val, True, C["text_hi"])
        self.surface.blit(k, (px + 11, y))
        self.surface.blit(v, (px + 11 + k.get_width(), y))

    # ── Reward curve ───────────────────────────────────────────────────────────

    def _update_reward(self, cum: float):
        self._reward_history.append(cum)
        if len(self._reward_history) > 300:
            self._reward_history.pop(0)

    def _draw_reward_curve(self):
        """Embed a sparkline reward curve at the bottom of the scene area."""
        buf = self._reward_history
        if len(buf) < 2: return

        gx = 8
        gy = BANNER_H + SCENE_H - 48
        gw = SIM_W - 16
        gh = 40

        bg = pygame.Surface((gw, gh), pygame.SRCALPHA)
        bg.fill((10, 12, 20, 160))
        self.surface.blit(bg, (gx, gy))

        rmin, rmax = min(buf), max(buf) + 1e-9
        rng = rmax - rmin

        pts = []
        for i, r in enumerate(buf):
            x = gx + int(i / max(len(buf) - 1, 1) * (gw - 2)) + 1
            y = gy + gh - 4 - int((r - rmin) / rng * (gh - 8))
            pts.append((x, y))

        if len(pts) > 1:
            # Gradient fill
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i+1]
                t = i / max(len(pts) - 1, 1)
                col = _lerp_color(C["reward_neg"], C["reward_pos"], t)
                pygame.draw.line(self.surface, col, p1, p2, 2)

        lbl = self._f_xs.render("CUMULATIVE REWARD", True, C["text_lo"])
        self.surface.blit(lbl, (gx + 4, gy + 2))

    # ── Compression depth overlay ──────────────────────────────────────────────

    def _draw_compression_overlay(self, patient):
        if patient.compression_depth < 0.05: return

        bx, by = SIM_W - 28, BANNER_H + 20
        bh = SCENE_H - 50

        # Background bar
        pygame.draw.rect(self.surface, C["bar_bg"],
                         pygame.Rect(bx, by, 18, bh), border_radius=5)

        fill_h = int(patient.compression_depth * bh)
        col = _lerp_color(C["red"], C["green"],
                          min(1.0, patient.compression_depth * 1.5))
        if fill_h > 0:
            pygame.draw.rect(self.surface, col,
                             pygame.Rect(bx, by + bh - fill_h, 18, fill_h),
                             border_radius=5)

        # Target zone marker (5–6 cm = ~0.7–0.9 normalised)
        for v in [0.7, 0.9]:
            vy = by + bh - int(v * bh)
            pygame.draw.line(self.surface, C["amber"], (bx - 2, vy), (bx + 20, vy), 1)

        d_lbl = self._f_xs.render("DEP", True, C["text_lo"])
        self.surface.blit(d_lbl, (bx + 1, by - 12))

    # ── ROSC glory overlay ────────────────────────────────────────────────────

    def _draw_rosc_glory(self):
        t = (self._frame % 60) / 60.0
        alpha = int(80 + 40 * math.sin(t * math.pi * 2))
        overlay = pygame.Surface((SIM_W, BANNER_H), pygame.SRCALPHA)
        overlay.fill((*C["rosc_gold"], alpha // 4))
        self.surface.blit(overlay, (0, BANNER_H))

        msg = self._f_xl.render("★  RETURN OF SPONTANEOUS CIRCULATION  ★", True, C["rosc_gold"])
        x = (SIM_W - msg.get_width()) // 2
        y = BANNER_H + (SCENE_H - msg.get_height()) // 2
        # Shadow
        shadow = self._f_xl.render("★  RETURN OF SPONTANEOUS CIRCULATION  ★",
                                    True, (0, 0, 0))
        self.surface.blit(shadow, (x + 2, y + 2))
        self.surface.blit(msg, (x, y))

        # Spawn particles
        if self._frame % 3 == 0:
            import random
            for _ in range(6):
                self._rosc_particles.append({
                    "x": random.uniform(50, SIM_W - 50),
                    "y": BANNER_H + SCENE_H // 2,
                    "vx": random.uniform(-2, 2),
                    "vy": random.uniform(-8, -2),
                    "life": 1.0,
                    "size": random.uniform(3, 8),
                    "col": random.choice([C["rosc_gold"], C["green"], C["text_hi"]]),
                })

    def _update_particles(self, dt: float):
        alive = []
        for p in self._rosc_particles:
            p["x"]    += p["vx"]
            p["y"]    += p["vy"]
            p["vy"]   += 0.18   # gravity
            p["life"] -= dt * 1.2
            if p["life"] > 0:
                alive.append(p)
        self._rosc_particles = alive

    def _draw_rosc_particles(self):
        for p in self._rosc_particles:
            col = (*p["col"], int(p["life"] * 220))
            r   = max(1, int(p["size"] * p["life"]))
            pygame.gfxdraw.filled_circle(self.surface,
                                         int(p["x"]), int(p["y"]), r, col)


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_ecg_template(n: int = 120) -> np.ndarray:
    """Synthetic PQRST waveform, one cardiac cycle."""
    t = np.linspace(0, 1, n)
    ecg = np.zeros(n)
    # P wave
    ecg += 0.25 * np.exp(-((t - 0.18) ** 2) / 0.001)
    # Q dip
    ecg -= 0.08 * np.exp(-((t - 0.30) ** 2) / 0.0004)
    # R spike
    ecg += 1.00 * np.exp(-((t - 0.35) ** 2) / 0.0002)
    # S dip
    ecg -= 0.25 * np.exp(-((t - 0.40) ** 2) / 0.0003)
    # T wave
    ecg += 0.35 * np.exp(-((t - 0.58) ** 2) / 0.003)
    return ecg.astype(np.float32)


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _lerp_color(a: tuple, b: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _lighten(col: tuple, amount: float) -> tuple:
    return tuple(min(255, int(c + (255 - c) * amount)) for c in col[:3])


def _darken(col: tuple, amount: float) -> tuple:
    return tuple(max(0, int(c * (1 - amount))) for c in col[:3])


def _alpha_blend(fg: tuple, bg: tuple, alpha: float) -> tuple:
    return _lerp_color(bg, fg, alpha)


def _draw_rounded_rect(surf, color, rect, radius):
    pygame.draw.rect(surf, color, rect, border_radius=radius)


def _draw_hline_glow(surf, y: int, width: int, color: tuple, width_px: int = 2):
    pygame.draw.line(surf, color, (0, y), (width, y), width_px)