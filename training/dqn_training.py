"""
DQN Training Script — CPR Position Assessment RL  (v2 — Anti-Collapse Edition)
================================================================================

Changes from v1 (informed by live run logs):
  • exploration_fraction raised across all experiments (was 0.30 → now 0.40-0.60)
    — DQN's ε-greedy needs to keep exploring longer to avoid action collapse.
  • total_timesteps → 750_000 (was 500_000) for more stable Q-value estimates
  • CollapseDetectorCallback from pg_training shared here too
  • OBS_DIM updated to 56 (custom_env v4 adds 5 extra features)
  • Curriculum difficulty: easy for exps 1-3, medium 4-7, hard 8-10
"""

import os
import sys
import json
import time
import numpy as np
import logging
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import CPREnv, ACTION_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DQN_Training")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _safe_load_results(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            content = f.read().strip()
        if not content:
            return []
        return json.loads(content)
    except Exception as e:
        log.warning(f"Results corrupt ({e}), resetting.")
        import shutil
        shutil.copy(path, path + ".bak")
        _atomic_save(path, [])
        return []


def _atomic_save(path, data):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=lambda o:
                float(o) if hasattr(o, "__float__") else
                int(o) if hasattr(o, "__int__") else str(o))
        os.replace(tmp, path)
    except Exception as e:
        log.error(f"Save failed {path}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
DQN_DIFFICULTY = {
    **{i: "easy"   for i in range(1, 4)},
    **{i: "medium" for i in range(4, 8)},
    **{i: "hard"   for i in range(8, 11)},
}


# ---------------------------------------------------------------------------
# Hyperparameter Grid (10 experiments) — higher exploration
# ---------------------------------------------------------------------------
DQN_EXPERIMENTS = [
    # Exp 1 — Baseline, easy, long exploration
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64, buffer_size=50_000,
         exploration_fraction=0.45, exploration_final_eps=0.05,
         learning_starts=2000, train_freq=4, target_update_interval=500,
         net_arch=[128, 128], total_timesteps=750_000),

    # Exp 2 — Higher LR, easy
    dict(learning_rate=1e-3, gamma=0.99, batch_size=64, buffer_size=50_000,
         exploration_fraction=0.50, exploration_final_eps=0.05,
         learning_starts=2000, train_freq=4, target_update_interval=500,
         net_arch=[128, 128], total_timesteps=750_000),

    # Exp 3 — Wide exploration, larger net, easy
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64, buffer_size=50_000,
         exploration_fraction=0.55, exploration_final_eps=0.08,
         learning_starts=2000, train_freq=4, target_update_interval=500,
         net_arch=[256, 256], total_timesteps=750_000),

    # Exp 4 — Medium, lower gamma, deep exploration
    dict(learning_rate=5e-4, gamma=0.95, batch_size=64, buffer_size=50_000,
         exploration_fraction=0.45, exploration_final_eps=0.05,
         learning_starts=2000, train_freq=4, target_update_interval=500,
         net_arch=[128, 128], total_timesteps=750_000),

    # Exp 5 — Larger buffer for diversity, medium
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64, buffer_size=100_000,
         exploration_fraction=0.45, exploration_final_eps=0.05,
         learning_starts=2000, train_freq=4, target_update_interval=500,
         net_arch=[128, 128], total_timesteps=750_000),

    # Exp 6 — More frequent updates, medium
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64, buffer_size=50_000,
         exploration_fraction=0.45, exploration_final_eps=0.05,
         learning_starts=2000, train_freq=1, target_update_interval=250,
         net_arch=[128, 128], total_timesteps=750_000),

    # Exp 7 — Slow target update, medium
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64, buffer_size=50_000,
         exploration_fraction=0.45, exploration_final_eps=0.05,
         learning_starts=2000, train_freq=4, target_update_interval=1000,
         net_arch=[128, 128], total_timesteps=750_000),

    # Exp 8 — Hard curriculum, three-layer net
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64, buffer_size=50_000,
         exploration_fraction=0.50, exploration_final_eps=0.08,
         learning_starts=2000, train_freq=4, target_update_interval=500,
         net_arch=[128, 128, 64], total_timesteps=750_000),

    # Exp 9 — Hard, sustained high epsilon floor
    dict(learning_rate=3e-4, gamma=0.99, batch_size=64, buffer_size=50_000,
         exploration_fraction=0.60, exploration_final_eps=0.10,
         learning_starts=2000, train_freq=4, target_update_interval=500,
         net_arch=[128, 128], total_timesteps=750_000),

    # Exp 10 — Hard, large batch, stable gradient
    dict(learning_rate=5e-4, gamma=0.99, batch_size=128, buffer_size=100_000,
         exploration_fraction=0.50, exploration_final_eps=0.08,
         learning_starts=4000, train_freq=4, target_update_interval=500,
         net_arch=[256, 128], total_timesteps=750_000),
]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: list = []
        self._current: dict = {}

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones",   [])
        for idx, (r, d) in enumerate(zip(rewards, dones)):
            self._current[idx] = self._current.get(idx, 0.0) + float(r)
            if d:
                self.episode_rewards.append(self._current[idx])
                self._current[idx] = 0.0
        return True


class CollapseDetectorCallback(BaseCallback):
    """Logs RED warning when any single action dominates > 50% of a window."""

    def __init__(self, window: int = 2048):
        super().__init__()
        self.window           = window
        self._buf:            deque = deque(maxlen=window)
        self.collapse_events: list  = []

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", [])
        for a in (actions if hasattr(actions, "__iter__") else [actions]):
            self._buf.append(int(a))

        if len(self._buf) == self.window and self.num_timesteps % self.window == 0:
            counts   = np.bincount(list(self._buf), minlength=12)
            top_idx  = int(np.argmax(counts))
            top_frac = float(counts[top_idx] / counts.sum())
            unique   = int(np.sum(counts > 0))

            if top_frac > 0.50:
                aname = ACTION_NAMES[top_idx] if top_idx < len(ACTION_NAMES) else str(top_idx)
                log.warning(
                    f"\033[91m[DQN COLLAPSE] t={self.num_timesteps:,} | "
                    f"{aname} = {top_frac:.0%} | unique={unique}\033[0m"
                )
                self.collapse_events.append({
                    "timestep": self.num_timesteps,
                    "action":   top_idx,
                    "fraction": round(top_frac, 3),
                })
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_env(difficulty: str = "medium"):
    def _inner():
        env = CPREnv(max_steps=200, difficulty=difficulty)
        return Monitor(env)
    return _inner


def train_dqn():
    results_path = "results/dqn_results.json"
    models_dir   = "models/dqn"
    logs_dir     = "logs/dqn"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    all_results = _safe_load_results(results_path)
    start_exp   = len(all_results)
    log.info(f"DQN: resuming from experiment {start_exp + 1}/10")

    for exp_idx, params in enumerate(DQN_EXPERIMENTS):
        exp_num    = exp_idx + 1
        difficulty = DQN_DIFFICULTY.get(exp_num, "medium")

        if exp_idx < start_exp:
            log.info(f"Skipping DQN exp {exp_num}")
            continue

        log.info(f"\n{'='*64}\nDQN Experiment {exp_num}/10  [difficulty={difficulty}]")
        log.info(f"Params: {params}\n{'='*64}")

        p        = params.copy()
        total_ts = p.pop("total_timesteps")
        net_arch = p.pop("net_arch")
        policy_kwargs = dict(net_arch=net_arch)

        try:
            vec_env     = make_vec_env(make_env(difficulty), n_envs=4)
            reward_cb   = RewardLoggerCallback()
            collapse_cb = CollapseDetectorCallback()
            ckpt_cb     = CheckpointCallback(
                save_freq=25_000,
                save_path=f"{models_dir}/exp_{exp_num}",
                name_prefix="dqn_cpr",
            )

            model = DQN(
                "MlpPolicy", vec_env,
                verbose=0,
                tensorboard_log=logs_dir,
                policy_kwargs=policy_kwargs,
                **p,
            )

            t0 = time.time()
            model.learn(
                total_timesteps=total_ts,
                callback=[reward_cb, collapse_cb, ckpt_cb],
                tb_log_name=f"dqn_exp_{exp_num}",
            )
            elapsed = time.time() - t0

            model_path = f"{models_dir}/dqn_exp_{exp_num}_final"
            model.save(model_path)

            ep_rews = reward_cb.episode_rewards
            result = {
                "exp":                exp_num,
                "difficulty":         difficulty,
                "learning_rate":      params["learning_rate"],
                "gamma":              params["gamma"],
                "buffer_size":        params["buffer_size"],
                "batch_size":         params["batch_size"],
                "exploration_fraction": params["exploration_fraction"],
                "net_arch":           str(net_arch),
                "mean_reward_last50": round(float(np.mean(ep_rews[-50:])) if ep_rews else 0.0, 3),
                "max_reward":         round(float(np.max(ep_rews))         if ep_rews else 0.0, 3),
                "total_episodes":     len(ep_rews),
                "train_time_s":       round(elapsed, 1),
                "reward_curve":       ep_rews[-200:],
                "collapse_events":    collapse_cb.collapse_events,
            }

            all_results.append(result)
            _atomic_save(results_path, all_results)

            n_col = len(collapse_cb.collapse_events)
            log.info(
                f"  ✓ Mean={result['mean_reward_last50']:.2f}  "
                f"Max={result['max_reward']:.2f}  "
                f"CollapseEvents={n_col}  Saved→{model_path}"
            )

        except Exception as e:
            log.error(f"  ✗ DQN Exp {exp_num} failed: {e}", exc_info=True)
            all_results.append({"exp": exp_num, "error": str(e)})
            _atomic_save(results_path, all_results)
        finally:
            try: vec_env.close()
            except Exception: pass

    log.info("\n" + "=" * 64)
    log.info("DQN Training Complete")
    log.info(f"Results → {results_path}")
    log.info(f"\n{'Exp':<5} {'LR':<10} {'Explor':<8} {'Mean':<12} {'Max':<10} {'Collapse'}")
    for r in all_results:
        if "error" not in r:
            log.info(f"{r['exp']:<5} {r['learning_rate']:<10} "
                     f"{r['exploration_fraction']:<8} "
                     f"{r['mean_reward_last50']:<12} {r['max_reward']:<10} "
                     f"{len(r.get('collapse_events', []))}")


if __name__ == "__main__":
    train_dqn()