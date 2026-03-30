"""
CPR Simulation Renderer
========================
Advanced 2D visualisation using Pygame with OpenGL-style rendering.
Displays the patient, landmark skeleton, vital signs dashboard,
action history, and reward tracker in real time.
"""

import numpy as np
import math
import sys
from typing import Optional

try:
    import pygame
    import pygame.gfxdraw
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# MediaPipe 17-keypoint connection pairs (indices into the 17-point list)
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),        # nose → left eye chain
    (0, 4), (4, 5), (5, 6),        # nose → right eye chain
    (7, 9), (8, 10),               # ear–mouth
    (11, 12),                      # shoulder–shoulder
    (11, 13), (13, 15),            # left arm
    (12, 14), (14, 16),            # right arm
]

COLORS = {
    "bg":           (18, 18, 28),
    "ground":       (42, 48, 60),
    "patient":      (220, 180, 140),
    "skeleton":     (80, 200, 160),
    "joint":        (255, 220, 80),
    "vitals_bg":    (28, 28, 45),
    "vitals_border":(60, 120, 200),
    "heart":        (220, 60, 80),
    "breath":       (80, 180, 220),
    "text":         (230, 230, 240),
    "dim_text":     (120, 130, 145),
    "reward_pos":   (80, 220, 120),
    "reward_neg":   (220, 80, 80),
    "action_bg":    (32, 36, 52),
    "bar_green":    (60, 200, 100),
    "bar_amber":    (240, 170, 40),
    "bar_red":      (220, 70, 70),
    "grid":         (35, 38, 55),
}

ACTION_NAMES_SHORT = [
    "ASSESS",  "CALL 911",  "OPEN AIRWAY",   "CHK BREATH",
    "COMPRESS", "RESCUE BREATH", "DEFIBRILLATE", "MON PULSE",
    "RECOVERY POS", "REPOSITION", "TILT HEAD", "WAIT",
]

W, H = 1280, 720
PANEL_W = 320


class CPRRenderer:
    """
    Renders the CPR environment using Pygame.

    The display is divided into:
    ┌──────────────────────┬──────────┐
    │   Patient Sim (3D-ish)│  Vitals  │
    │   with skeleton       │  Panel   │
    │                       │          │
    └───────────────────────┴──────────┘
    """

    def __init__(self):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not installed. Run: pip install pygame")

        pygame.init()
        pygame.display.set_caption("CPR Position Assessment — RL Environment")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self._font_lg = pygame.font.SysFont("DejaVuSans", 20, bold=True)
        self._font_md = pygame.font.SysFont("DejaVuSans", 15)
        self._font_sm = pygame.font.SysFont("DejaVuSans", 12)
        self._reward_history = []
        self._action_history = []
        self._frame = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, patient, step: int, cumulative_reward: float):
        self._frame += 1
        self.screen.fill(COLORS["bg"])

        self._draw_grid()
        self._draw_scene(patient)
        self._draw_vitals_panel(patient, step, cumulative_reward)
        self._draw_reward_curve(cumulative_reward)
        self._draw_action_log(patient)
        self._draw_header(step)

        pygame.display.flip()
        self.clock.tick(30)

        # Handle quit events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()
                sys.exit()

    def close(self):
        pygame.quit()

    # ------------------------------------------------------------------
    # Scene drawing
    # ------------------------------------------------------------------

    def _draw_grid(self):
        sim_w = W - PANEL_W
        for x in range(0, sim_w, 60):
            pygame.draw.line(self.screen, COLORS["grid"], (x, 0), (x, H), 1)
        for y in range(0, H, 60):
            pygame.draw.line(self.screen, COLORS["grid"], (0, y), (sim_w, y), 1)

    def _draw_scene(self, patient):
        sim_w = W - PANEL_W
        cx, cy = sim_w // 2, H // 2 + 40

        # Ground mat
        mat_rect = pygame.Rect(cx - 200, cy - 30, 400, 200)
        pygame.draw.rect(self.screen, COLORS["ground"], mat_rect, border_radius=10)
        pygame.draw.rect(self.screen, (55, 65, 80), mat_rect, 2, border_radius=10)

        # Label
        lbl = self._font_sm.render("FLOOR MAT", True, COLORS["dim_text"])
        self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy + 155))

        # Draw patient body
        self._draw_patient_body(patient, cx, cy)

        # Draw skeleton
        self._draw_skeleton(patient, cx, cy)

        # Recovery position indicator
        if patient.recovery_position:
            txt = self._font_md.render("↩ RECOVERY POSITION", True, COLORS["reward_pos"])
            self.screen.blit(txt, (cx - txt.get_width() // 2, cy - 80))

    def _draw_patient_body(self, patient, cx, cy):
        """Draw a simplified 2D body representation."""
        # Torso
        torso = pygame.Rect(cx - 35, cy + 10, 70, 110)
        pygame.draw.rect(self.screen, COLORS["patient"], torso, border_radius=8)

        # Head
        head_y_offset = int(-20 * patient.head_position)  # tilt shows visually
        pygame.draw.ellipse(self.screen, COLORS["patient"],
                            (cx - 22, cy - 50 + head_y_offset, 44, 50))

        # Arms
        pygame.draw.rect(self.screen, COLORS["patient"], (cx - 70, cy + 15, 35, 20), border_radius=5)
        pygame.draw.rect(self.screen, COLORS["patient"], (cx + 35, cy + 15, 35, 20), border_radius=5)

        # Legs
        pygame.draw.rect(self.screen, COLORS["patient"], (cx - 30, cy + 120, 25, 80), border_radius=5)
        pygame.draw.rect(self.screen, COLORS["patient"], (cx + 5, cy + 120, 25, 80), border_radius=5)

        # Chest compression visual
        if patient.compression_depth > 0.3:
            depth_px = int(patient.compression_depth * 12)
            comp_color = COLORS["bar_amber"] if patient.compression_depth < 0.7 else COLORS["bar_green"]
            pygame.draw.rect(self.screen, comp_color,
                             (cx - 20, cy + 30, 40, depth_px), border_radius=3)
            txt = self._font_sm.render("COMPRESS", True, comp_color)
            self.screen.blit(txt, (cx - txt.get_width() // 2, cy + 45 + depth_px))

        # Airway open indicator
        if patient.airway_open:
            pygame.draw.arc(self.screen, COLORS["breath"],
                            (cx - 15, cy - 45 + int(-20 * patient.head_position), 30, 20),
                            0, math.pi, 2)

    def _draw_skeleton(self, patient, cx, cy):
        """Map 17 normalised landmarks onto screen space."""
        lm = patient.landmarks
        sim_w = W - PANEL_W
        scale_x = sim_w * 0.55
        scale_y = H * 0.55
        offset_x = cx - scale_x // 2
        offset_y = cy - scale_y // 2

        pts = []
        for i in range(17):
            x = lm[i * 3] * scale_x + offset_x
            y = lm[i * 3 + 1] * scale_y + offset_y
            v = lm[i * 3 + 2]
            pts.append((int(x), int(y), v))

        # Draw connections
        for (a, b) in SKELETON_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                va, vb = pts[a][2], pts[b][2]
                alpha = int(min(va, vb) * 200)
                col = (*COLORS["skeleton"][:3],)
                pygame.draw.line(self.screen, col,
                                 pts[a][:2], pts[b][:2], 2)

        # Draw joints
        for (x, y, v) in pts:
            if v > 0.3:
                r = max(3, int(v * 6))
                pygame.gfxdraw.filled_circle(self.screen, x, y, r, (*COLORS["joint"], int(v * 230)))

    # ------------------------------------------------------------------
    # Vitals Panel
    # ------------------------------------------------------------------

    def _draw_vitals_panel(self, patient, step: int, cumulative_reward: float):
        px = W - PANEL_W
        panel_rect = pygame.Rect(px, 0, PANEL_W, H)
        pygame.draw.rect(self.screen, COLORS["vitals_bg"], panel_rect)
        pygame.draw.line(self.screen, COLORS["vitals_border"], (px, 0), (px, H), 2)

        y = 15
        # Title
        title = self._font_lg.render("VITAL MONITOR", True, COLORS["vitals_border"])
        self.screen.blit(title, (px + (PANEL_W - title.get_width()) // 2, y)); y += 35

        # Step / stage
        self._draw_kv(px, y, "STEP", str(step)); y += 22
        self._draw_kv(px, y, "REWARD", f"{cumulative_reward:.1f}"); y += 30

        # HR bar
        self._draw_bar(px, y, "HEART RATE", patient.heart_rate, COLORS["heart"]); y += 45
        # Chest rise
        self._draw_bar(px, y, "CHEST RISE", patient.chest_rise_rate, COLORS["breath"]); y += 45
        # Airway
        aw_col = COLORS["bar_green"] if patient.airway_open else COLORS["bar_red"]
        aw_txt = self._font_md.render(f"AIRWAY: {'OPEN ✓' if patient.airway_open else 'CLOSED ✗'}", True, aw_col)
        self.screen.blit(aw_txt, (px + 15, y)); y += 28
        # Hand placement
        self._draw_bar(px, y, "HAND PLACEMENT", patient.hand_placement, COLORS["bar_amber"]); y += 45
        # Compressions
        comp_txt = self._font_md.render(f"COMPRESSIONS: {patient.compressions_delivered}", True, COLORS["text"])
        self.screen.blit(comp_txt, (px + 15, y)); y += 22
        # Breaths
        br_txt = self._font_md.render(f"BREATHS: {patient.breaths_delivered}", True, COLORS["text"])
        self.screen.blit(br_txt, (px + 15, y)); y += 22
        # Consciousness
        self._draw_bar(px, y, "CONSCIOUSNESS", patient.consciousness_level, COLORS["bar_green"]); y += 45

        # Recovery
        rc_col = COLORS["bar_green"] if patient.recovery_position else COLORS["dim_text"]
        rc_txt = self._font_md.render(f"RECOVERY POS: {'YES' if patient.recovery_position else 'NO'}", True, rc_col)
        self.screen.blit(rc_txt, (px + 15, y)); y += 35

        # ROSC indicator
        if patient.heart_rate >= 0.9:
            rosc = self._font_lg.render("★ ROSC ★", True, COLORS["reward_pos"])
            self.screen.blit(rosc, (px + (PANEL_W - rosc.get_width()) // 2, y))

    def _draw_bar(self, px, y, label, value, color):
        bar_w = PANEL_W - 30
        lbl = self._font_sm.render(label, True, COLORS["dim_text"])
        self.screen.blit(lbl, (px + 15, y))
        bg_rect = pygame.Rect(px + 15, y + 16, bar_w, 12)
        pygame.draw.rect(self.screen, COLORS["grid"], bg_rect, border_radius=4)
        fill_w = int(value * bar_w)
        if fill_w > 0:
            fill_rect = pygame.Rect(px + 15, y + 16, fill_w, 12)
            pygame.draw.rect(self.screen, color, fill_rect, border_radius=4)
        val_txt = self._font_sm.render(f"{value:.0%}", True, COLORS["text"])
        self.screen.blit(val_txt, (px + 15 + bar_w - val_txt.get_width(), y))

    def _draw_kv(self, px, y, key, val):
        k = self._font_sm.render(key + ": ", True, COLORS["dim_text"])
        v = self._font_md.render(val, True, COLORS["text"])
        self.screen.blit(k, (px + 15, y))
        self.screen.blit(v, (px + 15 + k.get_width(), y))

    # ------------------------------------------------------------------
    # Reward curve
    # ------------------------------------------------------------------

    def _draw_reward_curve(self, cumulative_reward: float):
        self._reward_history.append(cumulative_reward)
        if len(self._reward_history) > 200:
            self._reward_history.pop(0)

        if len(self._reward_history) < 2:
            return

        sim_w = W - PANEL_W
        graph_h = 80
        graph_y = H - graph_h - 10
        graph_x = 10

        pygame.draw.rect(self.screen, COLORS["action_bg"],
                         (graph_x, graph_y, sim_w - 20, graph_h), border_radius=5)
        pygame.draw.rect(self.screen, COLORS["grid"],
                         (graph_x, graph_y, sim_w - 20, graph_h), 1, border_radius=5)

        rmin = min(self._reward_history)
        rmax = max(self._reward_history) + 1e-9
        rng = rmax - rmin

        pts = []
        for i, r in enumerate(self._reward_history):
            x = graph_x + int(i / len(self._reward_history) * (sim_w - 40)) + 10
            y = graph_y + graph_h - 10 - int((r - rmin) / rng * (graph_h - 20))
            pts.append((x, y))

        if len(pts) > 1:
            pygame.draw.lines(self.screen, COLORS["reward_pos"], False, pts, 2)

        lbl = self._font_sm.render("Cumulative Reward", True, COLORS["dim_text"])
        self.screen.blit(lbl, (graph_x + 8, graph_y + 4))

    # ------------------------------------------------------------------
    # Action log
    # ------------------------------------------------------------------

    def _draw_action_log(self, patient):
        pass  # Could add last N actions panel here

    def _draw_header(self, step: int):
        sim_w = W - PANEL_W
        title = self._font_lg.render(
            "CPR Position Assessment — RL Simulation", True, COLORS["vitals_border"]
        )
        self.screen.blit(title, ((sim_w - title.get_width()) // 2, 8))
        sub = self._font_sm.render(
            "Algorithms: DQN · REINFORCE · PPO  |  Observation: MediaPipe Pose Landmarks",
            True, COLORS["dim_text"],
        )
        self.screen.blit(sub, ((sim_w - sub.get_width()) // 2, 32))
