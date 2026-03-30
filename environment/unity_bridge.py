"""
Unity WebSocket Bridge
=======================
Python asyncio WebSocket server.
Unity connects as a client and receives real-time JSON state packets.

Architecture:
    Python RL Engine  →  UnityBridge (this file)  →  Unity 3D
                              ↑ port 8765

Message format sent to Unity:
{
  "type": "state" | "phase_change" | "episode_end" | "ping",
  "phase": "random" | "training" | "demo",
  "algorithm": "DQN" | "REINFORCE" | "PPO" | "RANDOM",
  "episode": 3,
  "step": 42,
  "action": 4,
  "action_name": "BEGIN_CHEST_COMPRESSIONS",
  "reward": 2.85,
  "cumulative_reward": 14.35,
  "is_correct": true,
  "feedback": "Good! Correct sequence.",
  "protocol_stage": 4,
  "landmarks": [x0,y0,v0, x1,y1,v1, ...],   // 51 floats
  "vitals": {
    "heart_rate": 0.45,
    "chest_rise": 0.20,
    "airway_open": true,
    "compressions": 30,
    "hand_placement": 0.80,
    "consciousness": 0.10
  },
  "rosc": false,
  "mean_reward": 12.4,    // rolling average (training phase)
  "experiment": 3         // hyperparameter experiment number
}

Usage:
    bridge = UnityBridge()
    bridge.start()                  # non-blocking background thread
    bridge.send_state(state_dict)   # thread-safe
    bridge.stop()
"""

import asyncio
import json
import threading
import logging
import time
from typing import Dict, Any, Optional, Set

log = logging.getLogger("UnityBridge")

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    log.warning("websockets not installed. Run: pip install websockets")


# ── Feedback templates per action / correctness ──────────────────────────────
FEEDBACK_MAP = {
    (0, True):  "✓ Scene safety assessed. Good start!",
    (0, False): "⚠ Assess consciousness before anything else.",
    (1, True):  "✓ Emergency services alerted. Critical step done.",
    (1, False): "⚠ Call 911 early — do not skip this step.",
    (2, True):  "✓ Airway opened with head-tilt chin-lift.",
    (2, False): "⚠ Open the airway before rescue breaths.",
    (3, True):  "✓ Breathing checked — look, listen, feel.",
    (3, False): "⚠ Check breathing before compressions.",
    (4, True):  "✓ Compressions! Push hard and fast — 5–6 cm depth.",
    (4, False): "⚠ Correct hand placement needed before compressions.",
    (5, True):  "✓ Rescue breaths delivered. Chest should rise.",
    (5, False): "⚠ Airway must be open for rescue breaths to work!",
    (6, True):  "✓ Defibrillation applied. Stand clear!",
    (6, False): "⚠ Need 30 compressions before defibrillation.",
    (7, True):  "✓ Pulse monitored — checking circulation.",
    (8, True):  "✓ Recovery position — patient is breathing again!",
    (8, False): "⚠ Recovery position only after ROSC (pulse restored).",
    (9, True):  "✓ Hand placement corrected — lower third of sternum.",
    (10, True): "✓ Head tilted back — airway extended.",
    (11, False):"⚠ Act quickly — every second counts in cardiac arrest!",
}


class UnityBridge:
    """
    Thread-safe WebSocket server.
    Broadcasts state to all connected Unity clients.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._queue: asyncio.Queue = None  # initialised in thread

        # Stats
        self.connected_clients = 0
        self.messages_sent = 0

    # ── Public API (call from any thread) ────────────────────────────────────

    def start(self):
        """Start the WebSocket server in a background thread."""
        if not WS_AVAILABLE:
            log.warning("WebSocket server not started — websockets package missing.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="UnityBridgeThread"
        )
        self._thread.start()
        time.sleep(0.5)  # give event loop time to initialise
        log.info(f"Unity WebSocket bridge started on ws://{self.host}:{self.port}")

    def stop(self):
        """Gracefully stop the server."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def send_state(self, state: Dict[str, Any]):
        """
        Thread-safe state broadcast to all Unity clients.
        Call this from the RL training/inference loop.
        """
        if not WS_AVAILABLE or not self._running:
            return

        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, json.dumps(state, default=_json_default)
            )
            self.messages_sent += 1

    def is_connected(self) -> bool:
        return self.connected_clients > 0

    # ── Internal event loop ──────────────────────────────────────────────────

    def _run_event_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(
            self._handle_client,
            self.host, self.port,
            ping_interval=20,
            ping_timeout=10,
        ) as server:
            self._server = server
            # Broadcast dispatcher
            asyncio.ensure_future(self._dispatcher())
            await asyncio.Future()  # run forever

    async def _handle_client(self, ws):
        self._clients.add(ws)
        self.connected_clients = len(self._clients)
        log.info(f"Unity client connected from {ws.remote_address}. "
                 f"Total clients: {self.connected_clients}")
        try:
            # Send welcome ping
            await ws.send(json.dumps({"type": "ping", "message": "CPR-RL Python Bridge ready"}))
            async for message in ws:
                # Handle incoming messages from Unity (e.g., user input)
                try:
                    data = json.loads(message)
                    log.debug(f"Unity → Python: {data}")
                except Exception:
                    pass
        except Exception as e:
            log.debug(f"Client disconnected: {e}")
        finally:
            self._clients.discard(ws)
            self.connected_clients = len(self._clients)

    async def _dispatcher(self):
        """Pull messages from queue and broadcast to all clients."""
        while True:
            try:
                message = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                if self._clients:
                    await asyncio.gather(
                        *[self._safe_send(ws, message) for ws in self._clients],
                        return_exceptions=True,
                    )
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                log.error(f"Dispatcher error: {e}")

    @staticmethod
    async def _safe_send(ws, message: str):
        try:
            await ws.send(message)
        except Exception:
            pass


# ── State builder helper ─────────────────────────────────────────────────────

def build_state_packet(
    phase: str,
    algorithm: str,
    episode: int,
    step: int,
    action: int,
    action_name: str,
    reward: float,
    cumulative_reward: float,
    landmarks: "np.ndarray",
    patient,
    protocol_stage: int,
    experiment: int = 0,
    mean_reward: float = 0.0,
) -> Dict[str, Any]:
    """
    Build the full JSON state packet for Unity.
    """
    # Determine if the action was 'correct' based on reward positivity
    is_correct = reward > 0.0

    # Get contextual feedback for layperson UI
    feedback_key = (action, is_correct)
    feedback = FEEDBACK_MAP.get(feedback_key, "Keep following the CPR protocol.")

    # ROSC detection
    rosc = getattr(patient, "heart_rate", 0.0) >= 0.9

    return {
        "type": "state",
        "phase": phase,
        "algorithm": algorithm,
        "episode": episode,
        "step": step,
        "action": action,
        "action_name": action_name,
        "reward": round(float(reward), 3),
        "cumulative_reward": round(float(cumulative_reward), 3),
        "is_correct": is_correct,
        "feedback": feedback,
        "protocol_stage": protocol_stage,
        "landmarks": [round(float(v), 4) for v in landmarks],
        "vitals": {
            "heart_rate": round(float(getattr(patient, "heart_rate", 0.0)), 3),
            "chest_rise": round(float(getattr(patient, "chest_rise_rate", 0.0)), 3),
            "airway_open": bool(getattr(patient, "airway_open", False)),
            "compressions": int(getattr(patient, "compressions_delivered", 0)),
            "hand_placement": round(float(getattr(patient, "hand_placement", 0.0)), 3),
            "consciousness": round(float(getattr(patient, "consciousness_level", 0.0)), 3),
            "recovery_position": bool(getattr(patient, "recovery_position", False)),
        },
        "rosc": rosc,
        "mean_reward": round(float(mean_reward), 3),
        "experiment": experiment,
    }


def build_phase_change_packet(phase: str, algorithm: str, experiment: int = 0) -> Dict:
    return {
        "type": "phase_change",
        "phase": phase,
        "algorithm": algorithm,
        "experiment": experiment,
        "message": {
            "random":   "Random Agent — Exploring environment freely",
            "training": f"Training {algorithm} — Experiment {experiment}/10",
            "demo":     f"Best {algorithm} model in action",
        }.get(phase, phase),
    }


def build_episode_end_packet(
    episode: int, total_reward: float, steps: int, rosc: bool, algorithm: str
) -> Dict:
    return {
        "type": "episode_end",
        "episode": episode,
        "total_reward": round(total_reward, 3),
        "steps": steps,
        "rosc": rosc,
        "algorithm": algorithm,
        "outcome": "ROSC — Patient Revived ✓" if rosc else "Episode Ended",
    }


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not serializable: {type(obj)}")