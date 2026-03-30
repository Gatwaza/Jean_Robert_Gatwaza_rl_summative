"""
DQN Training Script — CPR Position Assessment RL
=================================================
Runs 10 hyperparameter experiments using Stable-Baselines3 DQN.

Features:
- Vectorised environments (SubprocVecEnv)
- Automatic checkpoint saving after every experiment
- Resume from last completed experiment if interrupted
- TensorBoard logging
- Results saved as JSON for report generation
"""

import os
import sys
import json
import time
import numpy as np
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import CPREnv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DQN_Training")

def _safe_load_results(path):
    """Load JSON results, returning [] on any corruption or missing file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            content = f.read().strip()
        if not content:
            return []
        return json.loads(content)
    except Exception as e:
        log.warning(f"Results file corrupt ({e}), attempting recovery: {path}")
        # Try to recover partial records using incremental parsing
        recovered = []
        try:
            decoder = json.JSONDecoder()
            inner = content.lstrip().lstrip('[')
            pos = 0
            while pos < len(inner):
                chunk = inner[pos:].lstrip(' \n\t,')
                if not chunk or chunk.startswith(']'):
                    break
                obj, end = decoder.raw_decode(chunk)
                recovered.append(obj)
                pos += len(inner[pos:]) - len(chunk) + end
        except Exception:
            pass
        if recovered:
            log.info(f"Recovered {len(recovered)} records from corrupt file")
            _atomic_save(path, recovered)
        else:
            log.warning("No recoverable data — resetting results file")
            import shutil
            shutil.copy(path, path + ".bak")
            _atomic_save(path, [])
        return recovered


def _atomic_save(path, data):
    """Write JSON atomically: write to .tmp then rename, preventing corruption."""
    tmp = path + ".tmp"
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, default=lambda o:
                float(o) if hasattr(o, '__float__') else int(o)
                if hasattr(o, '__int__') else str(o))
        os.replace(tmp, path)   # atomic on POSIX, near-atomic on Windows
    except Exception as e:
        log.error(f"Failed to save results to {path}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)



# ---------------------------------------------------------------------------
# Hyperparameter Grid  (10 experiments) — Focused around best found on 500k
# ---------------------------------------------------------------------------
DQN_EXPERIMENTS = [
    # Exp 1 — Best baseline from prior: LR=0.01, gamma=0.95
    dict(
        learning_rate=1e-2, gamma=0.95, batch_size=32, buffer_size=20_000,
        exploration_fraction=0.15, exploration_final_eps=0.05,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
    # Exp 2 — Slightly smaller LR
    dict(
        learning_rate=8e-3, gamma=0.95, batch_size=32, buffer_size=20_000,
        exploration_fraction=0.15, exploration_final_eps=0.05,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
    # Exp 3 — Larger LR
    dict(
        learning_rate=1.2e-2, gamma=0.95, batch_size=32, buffer_size=20_000,
        exploration_fraction=0.15, exploration_final_eps=0.05,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
    # Exp 4 — Gamma 0.93
    dict(
        learning_rate=1e-2, gamma=0.93, batch_size=32, buffer_size=20_000,
        exploration_fraction=0.15, exploration_final_eps=0.05,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
    # Exp 5 — Gamma 0.96
    dict(
        learning_rate=1e-2, gamma=0.96, batch_size=32, buffer_size=20_000,
        exploration_fraction=0.15, exploration_final_eps=0.05,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
    # Exp 6 — Buffer 30k
    dict(
        learning_rate=1e-2, gamma=0.95, batch_size=32, buffer_size=30_000,
        exploration_fraction=0.15, exploration_final_eps=0.05,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
    # Exp 7 — Buffer 10k
    dict(
        learning_rate=1e-2, gamma=0.95, batch_size=32, buffer_size=10_000,
        exploration_fraction=0.15, exploration_final_eps=0.05,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
    # Exp 8 — Higher exploration: fraction 0.3, final_eps 0.1
    dict(
        learning_rate=1e-2, gamma=0.95, batch_size=32, buffer_size=20_000,
        exploration_fraction=0.3, exploration_final_eps=0.1,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
    # Exp 9 — Higher exploration: fraction 0.4, final_eps 0.1
    dict(
        learning_rate=1e-2, gamma=0.95, batch_size=32, buffer_size=20_000,
        exploration_fraction=0.4, exploration_final_eps=0.1,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
    # Exp 10 — Highest exploration: fraction 0.5, final_eps 0.15
    dict(
        learning_rate=1e-2, gamma=0.95, batch_size=32, buffer_size=20_000,
        exploration_fraction=0.5, exploration_final_eps=0.15,
        learning_starts=500, train_freq=4, target_update_interval=250,
        net_arch=[64, 64], total_timesteps=500_000,
    ),
]

# ---------------------------------------------------------------------------
# Reward logger callback
# ---------------------------------------------------------------------------
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        # FIX (bug 1): one running total per parallel env, keyed by env index.
        # A single shared float incorrectly accumulates rewards across all envs,
        # inflating episode rewards by ~n_envs and corrupting all reported metrics.
        self._current: dict[int, float] = {}

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones",   [])
        for env_idx, (r, d) in enumerate(zip(rewards, dones)):
            self._current[env_idx] = self._current.get(env_idx, 0.0) + float(r)
            if d:
                self.episode_rewards.append(self._current[env_idx])
                self._current[env_idx] = 0.0
        return True


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def make_env():
    env = CPREnv(max_steps=200, difficulty="medium")
    return Monitor(env)


def train_dqn():
    results_path = "results/dqn_results.json"
    models_dir = "models/dqn"
    logs_dir = "logs/dqn"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load previous results for resume
    all_results = []
    if os.path.exists(results_path):
        all_results = _safe_load_results(results_path)
        log.info(f"Resuming — {len(all_results)} experiments already complete")

    start_exp = len(all_results)

    for exp_idx, params in enumerate(DQN_EXPERIMENTS):
        if exp_idx < start_exp:
            log.info(f"Skipping experiment {exp_idx+1} (already done)")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"DQN Experiment {exp_idx+1}/10")
        log.info(f"Params: {params}")
        log.info(f"{'='*60}")

        p = params.copy()


        total_ts = p.pop("total_timesteps")
        net_arch = p.pop("net_arch")
        # FIX (bug 4): SB3 >= 1.7 requires dict(pi=[...], vf=[...]) to create
        # separate Q-network layers. The old list form still works for DQN (which
        # has no separate vf head), but align with the PPO fix for consistency.
        policy_kwargs = dict(net_arch=net_arch)  # DQN: list form is correct here

        try:
            vec_env = make_vec_env(make_env, n_envs=4)

            reward_cb = RewardLoggerCallback()
            checkpoint_cb = CheckpointCallback(
                save_freq=25_000,
                save_path=f"{models_dir}/exp_{exp_idx+1}",
                name_prefix="dqn_cpr",
            )

            model = DQN(
                "MlpPolicy",
                vec_env,
                verbose=0,
                tensorboard_log=logs_dir,
                policy_kwargs=policy_kwargs,
                **p,
            )

            t0 = time.time()
            model.learn(
                total_timesteps=total_ts,
                callback=[reward_cb, checkpoint_cb],
                tb_log_name=f"exp_{exp_idx+1}",
            )
            elapsed = time.time() - t0

            # Save final model
            model_path = f"{models_dir}/dqn_exp_{exp_idx+1}_final"
            model.save(model_path)

            # Compute stats
            ep_rews = reward_cb.episode_rewards
            mean_reward = float(np.mean(ep_rews[-50:])) if ep_rews else 0.0
            max_reward = float(np.max(ep_rews)) if ep_rews else 0.0
            episodes = len(ep_rews)

            result = {
                "exp": exp_idx + 1,
                "learning_rate": params.get("learning_rate"),
                "gamma": params.get("gamma"),
                "buffer_size": params.get("buffer_size"),
                "batch_size": params.get("batch_size"),
                "exploration_fraction": params.get("exploration_fraction"),
                "net_arch": str(net_arch),
                "mean_reward_last50": round(mean_reward, 3),
                "max_reward": round(max_reward, 3),
                "total_episodes": episodes,
                "train_time_s": round(elapsed, 1),
                "reward_curve": ep_rews[-200:],  # last 200 eps for plotting
            }

            all_results.append(result)

            # Save after every experiment for resume safety
            _atomic_save(results_path, all_results)

            log.info(f"  ✓ Mean reward (last 50): {mean_reward:.2f}  |  Max: {max_reward:.2f}")
            log.info(f"  ✓ Saved → {model_path}")

        except Exception as e:
            log.error(f"  ✗ Experiment {exp_idx+1} failed: {e}")
            # Save partial result with error flag
            all_results.append({"exp": exp_idx + 1, "error": str(e)})
            _atomic_save(results_path, all_results)

        finally:
            try:
                vec_env.close()
            except Exception:
                pass

    log.info("\n" + "="*60)
    log.info("DQN Training Complete")
    log.info(f"Results saved to {results_path}")

    # Print summary table
    log.info("\n{:<5} {:<10} {:<8} {:<12} {:<12}".format(
        "Exp", "LR", "Gamma", "Mean Rew", "Max Rew"))
    for r in all_results:
        if "error" not in r:
            log.info("{:<5} {:<10} {:<8} {:<12} {:<12}".format(
                r["exp"], r["learning_rate"], r["gamma"],
                r["mean_reward_last50"], r["max_reward"]))


if __name__ == "__main__":
    train_dqn()