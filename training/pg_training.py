"""
Policy Gradient Training — REINFORCE & PPO  (v2 — Anti-Collapse Edition)
=========================================================================

Key changes from v1 (informed by live run showing 64-72% MONITOR_PULSE):

  PPO_EXPERIMENTS:
    • ent_coef raised to 0.10-0.30 across all experiments (was 0.05-0.10)
      — High entropy is the primary defence against action collapse.
    • Added experiments with vf_coef tuning (critic helps actor stay diverse)
    • total_timesteps raised to 750_000 (was 500_000) for stable convergence
    • Curriculum difficulty: easy→medium→hard across experiment blocks

  CollapseDetectorCallback:
    • Monitors action distribution every 2048 steps
    • Logs a RED WARNING if any action > 50% of recent steps
    • Records entropy history so it appears in the JSON results

  RewardLoggerCallback (unchanged correctness, added entropy tracking)

Usage:
    python training/pg_training.py --algo ppo
    python training/pg_training.py --algo reinforce
    python training/pg_training.py --algo all
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
from collections import deque
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
        log.warning(f"Results file corrupt ({e}), resetting: {path}")
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
# PPO Hyperparameter Grid — anti-collapse focus
# ---------------------------------------------------------------------------
# Curriculum map: experiments 1-3 → easy, 4-7 → medium, 8-10 → hard
PPO_DIFFICULTY = {
    **{i: "easy"   for i in range(1, 4)},
    **{i: "medium" for i in range(4, 8)},
    **{i: "hard"   for i in range(8, 11)},
}

PPO_EXPERIMENTS = [
    # Exp 1 — High entropy baseline, easy curriculum
    dict(learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.15, clip_range=0.2,
         vf_coef=0.5, max_grad_norm=0.5, net_arch=[128, 128],
         total_timesteps=750_000),

    # Exp 2 — Even stronger entropy regularisation
    dict(learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.20, clip_range=0.2,
         vf_coef=0.5, max_grad_norm=0.5, net_arch=[128, 128],
         total_timesteps=750_000),

    # Exp 3 — Higher LR + entropy for fast early learning
    dict(learning_rate=1e-3, gamma=0.99, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.20, clip_range=0.2,
         vf_coef=0.5, max_grad_norm=0.5, net_arch=[128, 128],
         total_timesteps=750_000),

    # Exp 4 — Medium difficulty, bigger net, moderate entropy
    dict(learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.15, clip_range=0.2,
         vf_coef=0.5, max_grad_norm=0.5, net_arch=[256, 256],
         total_timesteps=750_000),

    # Exp 5 — More rollout steps, reduces variance
    dict(learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, n_steps=4096,
         batch_size=128, n_epochs=10, ent_coef=0.15, clip_range=0.2,
         vf_coef=0.5, max_grad_norm=0.5, net_arch=[128, 128],
         total_timesteps=750_000),

    # Exp 6 — Wider net, high entropy, medium curriculum
    dict(learning_rate=5e-4, gamma=0.99, gae_lambda=0.98, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.18, clip_range=0.2,
         vf_coef=0.6, max_grad_norm=0.5, net_arch=[256, 128],
         total_timesteps=750_000),

    # Exp 7 — Slow LR, strong entropy, tighter clip
    dict(learning_rate=1e-4, gamma=0.99, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=15, ent_coef=0.25, clip_range=0.15,
         vf_coef=0.5, max_grad_norm=0.5, net_arch=[128, 128],
         total_timesteps=750_000),

    # Exp 8 — Hard curriculum, very strong entropy
    dict(learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, n_steps=2048,
         batch_size=64, n_epochs=10, ent_coef=0.25, clip_range=0.2,
         vf_coef=0.5, max_grad_norm=0.5, net_arch=[256, 256],
         total_timesteps=750_000),

    # Exp 9 — Hard, larger batch, entropy decay via linear schedule
    dict(learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, n_steps=4096,
         batch_size=128, n_epochs=10, ent_coef=0.30, clip_range=0.2,
         vf_coef=0.5, max_grad_norm=0.5, net_arch=[128, 128],
         total_timesteps=750_000),

    # Exp 10 — Best-of-above synthesis
    dict(learning_rate=3e-4, gamma=0.99, gae_lambda=0.97, n_steps=2048,
         batch_size=64, n_epochs=12, ent_coef=0.20, clip_range=0.2,
         vf_coef=0.5, max_grad_norm=0.5, net_arch=[256, 256],
         total_timesteps=750_000),
]


# ---------------------------------------------------------------------------
# REINFORCE Hyperparameter Grid — also bumped entropy
# ---------------------------------------------------------------------------
REINFORCE_EXPERIMENTS = [
    dict(lr=1e-3, gamma=0.99, hidden=[64,  64],     use_baseline=False, entropy_coef=0.05,  episodes=4000),
    dict(lr=5e-4, gamma=0.99, hidden=[64,  64],     use_baseline=True,  entropy_coef=0.05,  episodes=4000),
    dict(lr=1e-3, gamma=0.95, hidden=[64,  64],     use_baseline=False, entropy_coef=0.10,  episodes=4000),
    dict(lr=2e-3, gamma=0.99, hidden=[128, 128],    use_baseline=False, entropy_coef=0.05,  episodes=4000),
    dict(lr=5e-4, gamma=0.99, hidden=[256],         use_baseline=True,  entropy_coef=0.10,  episodes=4000),
    dict(lr=1e-3, gamma=0.97, hidden=[64,  64],     use_baseline=True,  entropy_coef=0.15,  episodes=4000),
    dict(lr=1e-4, gamma=0.99, hidden=[64,  64,  64],use_baseline=False, entropy_coef=0.05,  episodes=4000),
    dict(lr=2e-3, gamma=0.95, hidden=[128, 64],     use_baseline=True,  entropy_coef=0.10,  episodes=4000),
    dict(lr=3e-4, gamma=0.99, hidden=[64,  64],     use_baseline=False, entropy_coef=0.20,  episodes=4000),
    dict(lr=1e-3, gamma=0.99, hidden=[128, 128],    use_baseline=True,  entropy_coef=0.15,  episodes=4000),
]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class RewardLoggerCallback(BaseCallback):
    """Tracks episode rewards correctly across multiple parallel envs."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: list = []
        self.entropy_history:  list = []
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

    def _on_rollout_end(self):
        if not hasattr(self.model, "rollout_buffer"):
            return
        try:
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
        except Exception:
            pass


class CollapseDetectorCallback(BaseCallback):
    """
    Detects action collapse during training and logs RED warnings.
    Records per-action frequency every rollout for the results JSON.
    """

    def __init__(self, n_actions: int = 12, window: int = 2048):
        super().__init__()
        self.n_actions       = n_actions
        self.window          = window
        self._action_buf:    deque = deque(maxlen=window)
        self.collapse_events: list = []   # (timestep, action, fraction)
        self.diversity_history: list = []  # rolling unique-action fractions

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", [])
        for a in (actions if hasattr(actions, "__iter__") else [actions]):
            self._action_buf.append(int(a))

        if len(self._action_buf) == self.window and self.num_timesteps % self.window == 0:
            counts   = np.bincount(list(self._action_buf), minlength=self.n_actions)
            fracs    = counts / counts.sum()
            top_idx  = int(np.argmax(fracs))
            top_frac = float(fracs[top_idx])
            unique   = int(np.sum(fracs > 0.01))

            self.diversity_history.append({
                "t": self.num_timesteps,
                "unique_actions": unique,
                "top_action": top_idx,
                "top_frac": round(top_frac, 3),
            })

            if top_frac > 0.50:
                from environment.custom_env import ACTION_NAMES
                aname = ACTION_NAMES[top_idx] if top_idx < len(ACTION_NAMES) else str(top_idx)
                log.warning(
                    f"\033[91m[COLLAPSE] t={self.num_timesteps:,} | "
                    f"{aname} = {top_frac:.0%} of {self.window} steps | "
                    f"unique={unique} — consider raising ent_coef\033[0m"
                )
                self.collapse_events.append({
                    "timestep": self.num_timesteps,
                    "action":   top_idx,
                    "fraction": round(top_frac, 3),
                })
            elif top_frac < 0.30:
                log.info(
                    f"[diversity ✓] t={self.num_timesteps:,} | "
                    f"unique={unique}/12 | top={top_frac:.0%}"
                )
        return True


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def make_env(difficulty: str = "medium"):
    def _inner():
        env = CPREnv(max_steps=200, difficulty=difficulty)
        return Monitor(env)
    return _inner


def train_ppo():
    results_path = "results/ppo_results.json"
    models_dir   = "models/pg/ppo"
    logs_dir     = "logs/ppo"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    all_results = _safe_load_results(results_path)
    start_exp   = len(all_results)
    log.info(f"PPO: resuming from experiment {start_exp + 1}/10")

    for exp_idx, params in enumerate(PPO_EXPERIMENTS):
        exp_num    = exp_idx + 1
        difficulty = PPO_DIFFICULTY.get(exp_num, "medium")

        if exp_idx < start_exp:
            log.info(f"Skipping PPO exp {exp_num} (already done)")
            continue

        log.info(f"\n{'='*64}\nPPO Experiment {exp_num}/10  [difficulty={difficulty}]")
        log.info(f"Params: {params}\n{'='*64}")

        p        = params.copy()
        total_ts = p.pop("total_timesteps")
        net_arch = p.pop("net_arch")
        policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))

        try:
            vec_env     = make_vec_env(make_env(difficulty), n_envs=4)
            reward_cb   = RewardLoggerCallback()
            collapse_cb = CollapseDetectorCallback()
            ckpt_cb     = CheckpointCallback(
                save_freq=50_000,
                save_path=f"{models_dir}/exp_{exp_num}",
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
            model.learn(
                total_timesteps=total_ts,
                callback=[reward_cb, collapse_cb, ckpt_cb],
                tb_log_name=f"ppo_exp_{exp_num}",
            )
            elapsed = time.time() - t0

            model_path = f"{models_dir}/ppo_exp_{exp_num}_final"
            model.save(model_path)

            ep_rews = reward_cb.episode_rewards
            result = {
                "exp":                exp_num,
                "difficulty":         difficulty,
                "learning_rate":      params["learning_rate"],
                "gamma":              params["gamma"],
                "ent_coef":           params["ent_coef"],
                "clip_range":         params["clip_range"],
                "n_steps":            params["n_steps"],
                "batch_size":         params["batch_size"],
                "n_epochs":           params["n_epochs"],
                "net_arch":           str(net_arch),
                "mean_reward_last50": round(float(np.mean(ep_rews[-50:])) if ep_rews else 0.0, 3),
                "max_reward":         round(float(np.max(ep_rews))         if ep_rews else 0.0, 3),
                "total_episodes":     len(ep_rews),
                "train_time_s":       round(elapsed, 1),
                "reward_curve":       ep_rews[-200:],
                "entropy_curve":      reward_cb.entropy_history[-100:],
                "collapse_events":    collapse_cb.collapse_events,
                "diversity_summary":  collapse_cb.diversity_history[-20:],
            }

            all_results.append(result)
            _atomic_save(results_path, all_results)

            n_collapse = len(collapse_cb.collapse_events)
            log.info(
                f"  ✓ Mean={result['mean_reward_last50']:.2f}  "
                f"Max={result['max_reward']:.2f}  "
                f"CollapseEvents={n_collapse}  "
                f"Saved→{model_path}"
            )
            if n_collapse > 3:
                log.warning(
                    f"  ⚠  {n_collapse} collapse events — model may not generalise well. "
                    f"Consider experiment with higher ent_coef."
                )

        except Exception as e:
            log.error(f"  ✗ PPO Exp {exp_num} failed: {e}", exc_info=True)
            all_results.append({"exp": exp_num, "error": str(e)})
            _atomic_save(results_path, all_results)
        finally:
            try: vec_env.close()
            except Exception: pass

    log.info("PPO Training Complete")


# ---------------------------------------------------------------------------
# REINFORCE (vanilla policy gradient)
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return Categorical(logits=self.net(x))


SAVE_EVERY = 200


def reinforce_train_single(params: dict, exp_idx: int, save_dir: str) -> dict:
    lr           = params["lr"]
    gamma        = params["gamma"]
    hidden       = params["hidden"]
    use_baseline = params["use_baseline"]
    entropy_coef = params["entropy_coef"]
    total_ep     = params["episodes"]

    ckpt_path    = f"{save_dir}/reinforce_exp_{exp_idx + 1}.pt"
    env          = CPREnv(max_steps=200, difficulty="medium")

    policy  = PolicyNet(OBS_DIM, 12, hidden)
    opt     = optim.Adam(policy.parameters(), lr=lr)
    baseline= 0.0

    episode_rewards: list = []
    entropy_history: list = []
    action_counts        = np.zeros(12)
    t0                   = time.time()

    # Rolling window for collapse detection
    recent_actions: deque = deque(maxlen=500)

    for ep in range(total_ep):
        obs, _    = env.reset()
        states, actions_ep, rewards_ep, logprobs_ep, entropies_ep = [], [], [], [], []
        done      = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            dist  = policy(obs_t)
            a     = dist.sample()
            obs, r, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated
            states.append(obs_t)
            actions_ep.append(a)
            rewards_ep.append(r)
            logprobs_ep.append(dist.log_prob(a))
            entropies_ep.append(dist.entropy())
            action_counts[a.item()] += 1
            recent_actions.append(a.item())

        # Monte Carlo returns
        G, returns = 0.0, []
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t    = torch.FloatTensor(returns)
        logprobs_t   = torch.stack(logprobs_ep)
        entropies_t  = torch.stack(entropies_ep)

        if use_baseline:
            ep_mean = float(returns_t.mean())
            baseline = 0.95 * baseline + 0.05 * ep_mean
            advantages = returns_t - baseline
        else:
            advantages = returns_t

        # Normalise advantages
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        policy_loss  = -(logprobs_t * advantages.detach()).mean()
        entropy_bonus = -entropy_coef * entropies_t.mean()
        loss          = policy_loss + entropy_bonus

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        ep_reward = sum(rewards_ep)
        episode_rewards.append(ep_reward)
        entropy_history.append(float(entropies_t.mean().detach()))

        if (ep + 1) % SAVE_EVERY == 0:
            torch.save({
                "policy_state": policy.state_dict(),
                "optim_state":  opt.state_dict(),
                "rewards":      episode_rewards,
                "entropy":      entropy_history,
            }, ckpt_path)
            mean_rew = float(np.mean(episode_rewards[-50:]))
            # Collapse check
            if len(recent_actions) >= 100:
                counts = np.bincount(list(recent_actions), minlength=12)
                top_frac = float(counts.max() / counts.sum())
                if top_frac > 0.60:
                    from environment.custom_env import ACTION_NAMES
                    top_a = int(counts.argmax())
                    log.warning(
                        f"\033[91m  [COLLAPSE] Ep {ep+1} | "
                        f"{ACTION_NAMES[top_a]} = {top_frac:.0%} in last 500 steps\033[0m"
                    )
            log.info(f"    Ep {ep+1}/{total_ep}  mean_rew={mean_rew:.2f}  "
                     f"H={entropy_history[-1]:.3f}")

    elapsed = time.time() - t0
    env.close()

    torch.save({
        "policy_state": policy.state_dict(),
        "optim_state":  opt.state_dict(),
        "rewards":      episode_rewards,
        "entropy":      entropy_history,
    }, ckpt_path)

    return {
        "exp":                exp_idx + 1,
        "learning_rate":      lr,
        "gamma":              gamma,
        "use_baseline":       use_baseline,
        "entropy_coef":       entropy_coef,
        "hidden_arch":        str(hidden),
        "mean_reward_last50": round(float(np.mean(episode_rewards[-50:])), 3),
        "max_reward":         round(float(np.max(episode_rewards)), 3),
        "total_episodes":     total_ep,
        "train_time_s":       round(elapsed, 1),
        "reward_curve":       episode_rewards[-200:],
        "entropy_curve":      entropy_history[-200:],
    }


OBS_DIM = 56   # matches custom_env.py v4


def train_reinforce():
    results_path = "results/reinforce_results.json"
    save_dir     = "models/pg/reinforce"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    all_results = _safe_load_results(results_path)
    start_exp   = len(all_results)

    for exp_idx, params in enumerate(REINFORCE_EXPERIMENTS):
        if exp_idx < start_exp:
            log.info(f"Skipping REINFORCE exp {exp_idx + 1}")
            continue
        log.info(f"\n{'='*64}\nREINFORCE Experiment {exp_idx+1}/10  {params}\n{'='*64}")
        try:
            result = reinforce_train_single(params, exp_idx, save_dir)
            all_results.append(result)
            _atomic_save(results_path, all_results)
            log.info(f"  ✓ Mean={result['mean_reward_last50']:.2f}")
        except Exception as e:
            log.error(f"  ✗ REINFORCE Exp {exp_idx+1} failed: {e}", exc_info=True)
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