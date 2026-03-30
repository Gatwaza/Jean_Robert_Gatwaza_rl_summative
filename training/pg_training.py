"""
Policy Gradient Training — REINFORCE & PPO
==========================================
10 experiments each for REINFORCE (via SB3 A2C as SB3 has no standalone
REINFORCE — we implement vanilla REINFORCE as custom wrapper) and PPO.

Both use the same CPREnv for objective comparison.
Resume-safe: checks existing JSON results before starting.
"""

import os
import sys
import json
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import CPREnv

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PG_Training")

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
# PPO Hyperparameter Grid (10 experiments) — focused around strong prior DQN signal
# ---------------------------------------------------------------------------
PPO_EXPERIMENTS = [
    # Exp 1 — Base candidate (LR=1e-3, gamma=0.95, strong exploitation)
    dict(learning_rate=1e-3, gamma=0.95, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.001, clip_range=0.2,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
    # Exp 2 — Lower LR
    dict(learning_rate=5e-4, gamma=0.95, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.001, clip_range=0.2,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
    # Exp 3 — Higher LR
    dict(learning_rate=2e-3, gamma=0.95, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.001, clip_range=0.2,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
    # Exp 4 — Lower gamma
    dict(learning_rate=1e-3, gamma=0.90, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.001, clip_range=0.2,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
    # Exp 5 — Higher gamma
    dict(learning_rate=1e-3, gamma=0.99, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.001, clip_range=0.2,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
    # Exp 6 — Lower entropy
    dict(learning_rate=1e-3, gamma=0.95, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.0001, clip_range=0.2,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
    # Exp 7 — Higher entropy
    dict(learning_rate=1e-3, gamma=0.95, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.01, clip_range=0.2,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
    # Exp 8 — Smaller batch
    dict(learning_rate=1e-3, gamma=0.95, gae_lambda=0.95, n_steps=2048,
         batch_size=32, n_epochs=10, ent_coef=0.001, clip_range=0.2,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
    # Exp 9 — Larger batch
    dict(learning_rate=1e-3, gamma=0.95, gae_lambda=0.95, n_steps=2048,
         batch_size=128, n_epochs=10, ent_coef=0.001, clip_range=0.2,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
    # Exp 10 — Alternative clip range
    dict(learning_rate=1e-3, gamma=0.95, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.001, clip_range=0.1,
         max_grad_norm=0.5, net_arch=[64, 64], total_timesteps=500_000),
]

# ---------------------------------------------------------------------------
# REINFORCE Hyperparameter Grid (10 experiments)
# Pure policy gradient, Monte Carlo returns, no baseline by default
# ---------------------------------------------------------------------------
REINFORCE_EXPERIMENTS = [
    dict(lr=1e-3, gamma=0.99, hidden=[64, 64],     use_baseline=False, entropy_coef=0.0,  episodes=3000),
    dict(lr=5e-4, gamma=0.99, hidden=[64, 64],     use_baseline=True,  entropy_coef=0.0,  episodes=3000),
    dict(lr=1e-3, gamma=0.95, hidden=[64, 64],     use_baseline=False, entropy_coef=0.01, episodes=3000),
    dict(lr=2e-3, gamma=0.99, hidden=[128, 128],   use_baseline=False, entropy_coef=0.0,  episodes=3000),
    dict(lr=5e-4, gamma=0.99, hidden=[256],        use_baseline=True,  entropy_coef=0.01, episodes=3000),
    dict(lr=1e-3, gamma=0.97, hidden=[64, 64],     use_baseline=True,  entropy_coef=0.02, episodes=3000),
    dict(lr=1e-4, gamma=0.99, hidden=[64, 64, 64], use_baseline=False, entropy_coef=0.0,  episodes=3000),
    dict(lr=2e-3, gamma=0.95, hidden=[128, 64],    use_baseline=True,  entropy_coef=0.01, episodes=3000),
    dict(lr=3e-4, gamma=0.99, hidden=[64, 64],     use_baseline=False, entropy_coef=0.05, episodes=3000),
    dict(lr=1e-3, gamma=0.99, hidden=[128, 128],   use_baseline=True,  entropy_coef=0.02, episodes=3000),
]


# ---------------------------------------------------------------------------
# Callback for PPO reward tracking
# ---------------------------------------------------------------------------
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.entropy_history: list[float] = []
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

    def _on_rollout_end(self) -> None:
        # FIX (bug 3): was a silent no-op (try: pass). Now actually measures entropy
        # so policy collapse during training is visible in logs.
        if not hasattr(self.model, "rollout_buffer"):
            return
        try:
            import torch
            buf = self.model.rollout_buffer
            if not buf.full:
                return
            obs_t = torch.FloatTensor(
                buf.observations.reshape(-1, buf.observations.shape[-1])
            )
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_t)
                ent  = dist.entropy().mean().item()
            self.entropy_history.append(ent)
            if len(self.entropy_history) % 10 == 0:
                warn = "  ⚠ collapsing — raise ent_coef" if ent < 0.5 else ""
                log.info(f"  Policy entropy: {ent:.4f}{warn}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Vanilla REINFORCE Implementation
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)
        self.baseline_net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)

    def baseline(self, x):
        return self.baseline_net(x).squeeze(-1)


def reinforce_train_single(params: dict, exp_idx: int, save_dir: str) -> dict:
    """Train one REINFORCE experiment."""
    lr = params["lr"]
    gamma = params["gamma"]
    hidden = params["hidden"]
    use_baseline = params["use_baseline"]
    entropy_coef = params["entropy_coef"]
    total_episodes = params["episodes"]

    env = CPREnv(max_steps=200, difficulty="medium")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim, hidden)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Checkpoint resume
    ckpt_path = f"{save_dir}/reinforce_exp_{exp_idx+1}.pt"
    episode_rewards = []
    entropy_history = []
    start_ep = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        policy.load_state_dict(ckpt["policy_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        episode_rewards = ckpt.get("rewards", [])
        entropy_history = ckpt.get("entropy", [])
        start_ep = len(episode_rewards)
        log.info(f"  Resuming from episode {start_ep}")

    t0 = time.time()
    SAVE_EVERY = 500

    for ep in range(start_ep, total_episodes):
        obs, _ = env.reset()
        log_probs, rewards_ep, entropies, baselines = [], [], [], []
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            dist = policy(obs_t)
            action = dist.sample()
            lp = dist.log_prob(action)
            ent = dist.entropy()

            if use_baseline:
                bv = policy.baseline(obs_t)
                baselines.append(bv)

            obs, r, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(lp)
            rewards_ep.append(r)
            entropies.append(ent)

        # Compute discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns)

        # Normalise returns
        if len(returns) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy loss
        if use_baseline and baselines:
            baseline_vals = torch.cat(baselines)
            advantages = returns_t - baseline_vals.detach()
            baseline_loss = nn.functional.mse_loss(baseline_vals, returns_t)
        else:
            advantages = returns_t
            baseline_loss = torch.tensor(0.0)

        log_probs_t = torch.stack(log_probs).squeeze()
        entropies_t = torch.stack(entropies).squeeze()

        if log_probs_t.dim() == 0:
            log_probs_t = log_probs_t.unsqueeze(0)
            entropies_t = entropies_t.unsqueeze(0)
            advantages = advantages.unsqueeze(0)

        policy_loss = -(log_probs_t * advantages).mean()
        entropy_bonus = -entropy_coef * entropies_t.mean()
        loss = policy_loss + entropy_bonus + 0.5 * baseline_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        ep_reward = sum(rewards_ep)
        episode_rewards.append(ep_reward)
        entropy_history.append(float(entropies_t.mean().detach()))

        if (ep + 1) % SAVE_EVERY == 0:
            torch.save({
                "policy_state": policy.state_dict(),
                "optim_state": optimizer.state_dict(),
                "rewards": episode_rewards,
                "entropy": entropy_history,
            }, ckpt_path)
            mean_rew = np.mean(episode_rewards[-50:])
            log.info(f"    Ep {ep+1}/{total_episodes}  mean_rew={mean_rew:.2f}")

    elapsed = time.time() - t0
    env.close()

    # Final save
    torch.save({
        "policy_state": policy.state_dict(),
        "optim_state": optimizer.state_dict(),
        "rewards": episode_rewards,
        "entropy": entropy_history,
    }, ckpt_path)

    return {
        "exp": exp_idx + 1,
        "learning_rate": lr,
        "gamma": gamma,
        "use_baseline": use_baseline,
        "entropy_coef": entropy_coef,
        "hidden_arch": str(hidden),
        "mean_reward_last50": round(float(np.mean(episode_rewards[-50:])), 3),
        "max_reward": round(float(np.max(episode_rewards)), 3),
        "total_episodes": total_episodes,
        "train_time_s": round(elapsed, 1),
        "reward_curve": episode_rewards[-200:],
        "entropy_curve": entropy_history[-200:],
    }


# ---------------------------------------------------------------------------
# PPO training routine
# ---------------------------------------------------------------------------
def make_env():
    return Monitor(CPREnv(max_steps=200, difficulty="medium"))


def train_ppo():
    results_path = "results/ppo_results.json"
    models_dir = "models/pg/ppo"
    logs_dir = "logs/ppo"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    all_results = []
    if os.path.exists(results_path):
        all_results = _safe_load_results(results_path)
        log.info(f"Resuming PPO — {len(all_results)} experiments done")

    start_exp = len(all_results)

    for exp_idx, params in enumerate(PPO_EXPERIMENTS):
        if exp_idx < start_exp:
            log.info(f"Skipping PPO exp {exp_idx+1}")
            continue

        log.info(f"\n{'='*60}\nPPO Experiment {exp_idx+1}/10\nParams: {params}\n{'='*60}")

        p = params.copy()


        total_ts = p.pop("total_timesteps")
        net_arch = p.pop("net_arch")
        # FIX (bug 4): SB3 >= 1.7 requires dict(pi=[...], vf=[...]) to create
        # separate actor/critic networks. The old list form silently creates a shared
        # trunk, which is suboptimal for actor-critic methods like PPO.
        policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))

        try:
            vec_env = make_vec_env(make_env, n_envs=4)
            reward_cb = RewardLoggerCallback()
            ckpt_cb = CheckpointCallback(
                save_freq=50_000,
                save_path=f"{models_dir}/exp_{exp_idx+1}",
                name_prefix="ppo_cpr",
            )

            model = PPO(
                "MlpPolicy", vec_env,
                verbose=0,
                tensorboard_log=logs_dir,
                policy_kwargs=policy_kwargs,
                **p,
            )

            t0 = time.time()
            model.learn(total_timesteps=total_ts,
                        callback=[reward_cb, ckpt_cb],
                        tb_log_name=f"ppo_exp_{exp_idx+1}")
            elapsed = time.time() - t0

            model_path = f"{models_dir}/ppo_exp_{exp_idx+1}_final"
            model.save(model_path)

            ep_rews = reward_cb.episode_rewards
            result = {
                "exp": exp_idx + 1,
                "learning_rate": params.get("learning_rate"),
                "gamma": params.get("gamma"),
                "ent_coef": params.get("ent_coef"),
                "clip_range": params.get("clip_range"),
                "n_steps": params.get("n_steps"),
                "batch_size": params.get("batch_size"),
                "n_epochs": params.get("n_epochs"),
                "net_arch": str(net_arch),
                "mean_reward_last50": round(float(np.mean(ep_rews[-50:])) if ep_rews else 0.0, 3),
                "max_reward": round(float(np.max(ep_rews)) if ep_rews else 0.0, 3),
                "total_episodes": len(ep_rews),
                "train_time_s": round(elapsed, 1),
                "reward_curve": ep_rews[-200:],
            }

            all_results.append(result)
            _atomic_save(results_path, all_results)

            log.info(f"  ✓ Mean reward: {result['mean_reward_last50']:.2f}  Max: {result['max_reward']:.2f}")

        except Exception as e:
            log.error(f"  ✗ PPO Exp {exp_idx+1} failed: {e}")
            all_results.append({"exp": exp_idx + 1, "error": str(e)})
            _atomic_save(results_path, all_results)
        finally:
            try:
                vec_env.close()
            except Exception:
                pass

    log.info("PPO Training Complete")


def train_reinforce():
    results_path = "results/reinforce_results.json"
    save_dir = "models/pg/reinforce"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    all_results = []
    if os.path.exists(results_path):
        all_results = _safe_load_results(results_path)
        log.info(f"Resuming REINFORCE — {len(all_results)} experiments done")

    start_exp = len(all_results)

    for exp_idx, params in enumerate(REINFORCE_EXPERIMENTS):
        if exp_idx < start_exp:
            log.info(f"Skipping REINFORCE exp {exp_idx+1}")
            continue

        log.info(f"\n{'='*60}\nREINFORCE Experiment {exp_idx+1}/10\nParams: {params}\n{'='*60}")

        try:
            result = reinforce_train_single(params, exp_idx, save_dir)
            all_results.append(result)
            _atomic_save(results_path, all_results)
            log.info(f"  ✓ Mean reward: {result['mean_reward_last50']:.2f}")
        except Exception as e:
            log.error(f"  ✗ REINFORCE Exp {exp_idx+1} failed: {e}")
            all_results.append({"exp": exp_idx + 1, "error": str(e)})
            _atomic_save(results_path, all_results)

    log.info("REINFORCE Training Complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "reinforce", "all"], default="all")
    args = parser.parse_args()

    if args.algo in ("reinforce", "all"):
        train_reinforce()
    if args.algo in ("ppo", "all"):
        train_ppo()