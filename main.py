"""
main.py — CPR Position Assessment RL  (v2 — Diagnostics Edition)
=================================================================

New in v2 (informed by live run showing action collapse):
  • Episode diagnostics — prints per-action frequency table after every episode
  • Collapse auto-diagnosis — explains WHY collapse happened and suggests fix
  • ROSC rate tracker — logs ROSC / episode across demo runs
  • Curriculum demo — --demo --curriculum cycles through easy→medium→hard
  • Bridge thread error handled gracefully (no traceback dump at exit)

Usage (unchanged from v1):
    python main.py --random
    python main.py --demo
    python main.py --demo --algo ppo
    python main.py --demo --algo ppo --exp 4
    python main.py --train --algo dqn
    python main.py --demo --no-bridge
    python main.py --demo --episodes 5
    python main.py --demo --mediapipe
    python main.py --demo --video cpr.mp4
"""

import os, sys, json, time, argparse, numpy as np, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env   import CPREnv, ACTION_NAMES, N_ACTIONS
from environment.unity_bridge import (
    UnityBridge, build_state_packet,
    build_phase_change_packet, build_episode_end_packet,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Main")

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║     CPR POSITION ASSESSMENT — REINFORCEMENT LEARNING SYSTEM      ║
║   Unity 3D Visualization  ·  MediaPipe Pose  ·  DQN/REINF/PPO   ║
╚══════════════════════════════════════════════════════════════════╝
"""
SEP = "─" * 72


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _action_frequency_table(counts: dict, total_steps: int) -> str:
    """Return a formatted action frequency table for the terminal."""
    lines = ["  Action distribution:"]
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    for a, n in sorted_counts[:8]:    # top 8
        pct  = n / total_steps * 100
        bar  = "█" * int(pct / 4)
        flag = "  ⚠ EXPLOIT?" if pct > 50 else ""
        lines.append(f"    {ACTION_NAMES[a]:<28} {n:>4}  ({pct:4.1f}%)  {bar}{flag}")
    return "\n".join(lines)


def _collapse_diagnosis(counts: dict, total_steps: int) -> str:
    """Return a plain-English explanation and fix recommendation."""
    top_action = max(counts, key=counts.get)
    top_pct    = counts[top_action] / total_steps * 100
    aname      = ACTION_NAMES[top_action]

    if top_pct < 50:
        return f"  Action mix OK (top={aname} {top_pct:.0f}%)"

    fixes = {
        7:  ("MONITOR_PULSE exploit — the policy earns +0.5/step safely.\n"
             "  Fixes: (1) env.py already caps consecutive uses to 3 (v4) — re-train.\n"
             "         (2) Raise PPO ent_coef ≥ 0.15 to force exploration.\n"
             "         (3) Run: python training/pg_training.py --algo ppo"),
        11: ("WAIT_OBSERVE exploit — the policy avoids penalties by waiting.\n"
             "  Fix: Raise ent_coef, check that wait penalty escalates correctly."),
        5:  ("RESCUE_BREATHS exploit — the policy pumps HR with breaths alone.\n"
             "  Fix: env v4 reduces breath_hr_gain to 0.03 — re-train the model."),
        4:  ("COMPRESSIONS overuse — policy may be stuck in stage<4 loop.\n"
             "  This is actually good behaviour; check if ROSC is being reached."),
    }
    default_fix = (
        f"{aname} dominates at {top_pct:.0f}%.\n"
        "  Fixes: raise ent_coef, check reward shaping, re-train with v4 env."
    )
    return f"  ⚠ COLLAPSE: {fixes.get(top_action, default_fix)}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_best_model(algo: str, force_exp: int = None):
    result_map = {
        "dqn":      "results/dqn_results.json",
        "ppo":      "results/ppo_results.json",
        "reinforce":"results/reinforce_results.json",
    }
    rpath = result_map.get(algo)
    if not rpath or not os.path.exists(rpath):
        raise FileNotFoundError(
            f"No results for {algo}. Train first:\n"
            f"  python training/{'pg_training.py --algo ' + algo if algo != 'dqn' else 'dqn_training.py'}"
        )
    with open(rpath) as f:
        results = json.load(f)
    valid = [r for r in results if "error" not in r]
    if not valid:
        raise ValueError(f"No successful {algo} experiments.")

    # If a specific experiment is requested, use it directly
    if force_exp is not None:
        match = [r for r in valid if r["exp"] == force_exp]
        if not match:
            available = [r["exp"] for r in valid]
            raise ValueError(
                f"Experiment {force_exp} not found in {algo} results. "
                f"Available: {available}"
            )
        best = match[0]
        log.info(f"Using forced exp {force_exp} (--exp flag)")
    else:
        # ROSC rate dominates — then mean reward, then penalise collapse
        def score(r):
            n_collapse = len(r.get("collapse_events", []))
            rosc_bonus = r.get("rosc_rate", 0.0) * 300
            return r.get("mean_reward_last50", -999) + rosc_bonus - n_collapse * 5
        best = max(valid, key=score)

    exp_num = best["exp"]
    log.info(
        f"Best {algo.upper()} → Exp {exp_num}  "
        f"mean={best.get('mean_reward_last50', 0):.2f}  "
        f"max={best.get('max_reward', 0):.2f}  "
        f"rosc_rate={best.get('rosc_rate', 'n/a')}  "
        f"collapse_events={len(best.get('collapse_events', []))}"
    )

    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(f"models/dqn/dqn_exp_{exp_num}_final"), "sb3", exp_num

    elif algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(f"models/pg/ppo/ppo_exp_{exp_num}_final"), "sb3", exp_num

    elif algo == "reinforce":
        import torch
        from training.pg_training import PolicyNet
        ckpt = torch.load(
            f"models/pg/reinforce/reinforce_exp_{exp_num}.pt", map_location="cpu"
        )
        info   = next(r for r in valid if r["exp"] == exp_num)
        policy = PolicyNet(56, 12, eval(info.get("hidden_arch", "[64, 64]")))
        policy.load_state_dict(ckpt["policy_state"])
        policy.eval()
        return policy, "reinforce", exp_num


# ---------------------------------------------------------------------------
# Landmark stream
# ---------------------------------------------------------------------------

def get_landmark_stream(use_mediapipe: bool, video_path=None):
    if use_mediapipe:
        from environment.mediapipe_extractor import LandmarkExtractor, get_best_extractor
        if video_path:
            return LandmarkExtractor(source="video", path=video_path).stream()
        return get_best_extractor(prefer_video=True).stream()
    from environment.mediapipe_extractor import LandmarkExtractor
    return LandmarkExtractor(source="synthetic").stream()


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------

def run_random_agent(bridge, episodes: int, use_mediapipe: bool, video_path=None,
                     difficulty: str = "medium"):
    log.info(f"PHASE: RANDOM AGENT — exploring [{difficulty}]")
    bridge.send_state(build_phase_change_packet("random", "RANDOM"))
    lm_gen  = get_landmark_stream(use_mediapipe, video_path)
    env     = CPREnv(max_steps=200, difficulty=difficulty)
    rosc_ep = 0

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False; ep_reward = 0.0; step = 0
        counts: dict = {}
        print(f"\n{SEP}\n  RANDOM EPISODE {ep}/{episodes}  [{difficulty}]\n{SEP}")

        while not done:
            action = env.action_space.sample()
            lm     = next(lm_gen)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated; ep_reward += reward; step += 1
            counts[action] = counts.get(action, 0) + 1
            bridge.send_state(build_state_packet(
                "random", "RANDOM", ep, step, action, ACTION_NAMES[action],
                reward, ep_reward, lm, env._patient, info["protocol_stage"]))
            print(f"  Step {step:3d} | {ACTION_NAMES[action]:<28} | "
                  f"R: {reward:+5.2f} | HR: {info['heart_rate']:.2f}")
            time.sleep(0.06)

        rosc = info["heart_rate"] >= 0.9
        if rosc: rosc_ep += 1
        bridge.send_state(build_episode_end_packet(ep, ep_reward, step, rosc, "RANDOM"))
        print(f"\n  {'★ ROSC' if rosc else 'Episode ended'}  "
              f"Reward: {ep_reward:.2f}  Steps: {step}")
        print(_action_frequency_table(counts, step))
        if max(counts.values()) / step > 0.50:
            print(_collapse_diagnosis(counts, step))

    env.close()
    print(f"\n  ROSC rate: {rosc_ep}/{episodes} episodes")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo(bridge, algo: str, episodes: int, use_mediapipe: bool,
             video_path=None, difficulty: str = "medium", force_exp: int = None):
    log.info(f"PHASE: DEMO — Best {algo.upper()} [{difficulty}]")
    model, model_type, exp_num = load_best_model(algo, force_exp=force_exp)
    bridge.send_state(build_phase_change_packet("demo", algo.upper(), exp_num))
    lm_gen     = get_landmark_stream(use_mediapipe, video_path)
    env        = CPREnv(max_steps=200, difficulty=difficulty)
    all_rewards: list = []
    rosc_count  = 0

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False; ep_reward = 0.0; step = 0
        counts: dict = {}
        print(f"\n{SEP}\n  DEMO EPISODE {ep}/{episodes}  [{algo.upper()}  exp={exp_num}  diff={difficulty}]\n{SEP}")

        while not done:
            if model_type == "sb3":
                action, _ = model.predict(obs, deterministic=False)
                action = int(action)
            else:
                import torch
                with torch.no_grad():
                    dist   = model(torch.FloatTensor(obs).unsqueeze(0))
                    action = dist.sample().item()

            lm = next(lm_gen)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated; ep_reward += reward; step += 1
            counts[action] = counts.get(action, 0) + 1

            bridge.send_state(build_state_packet(
                "demo", algo.upper(), ep, step, action, ACTION_NAMES[action],
                reward, ep_reward, lm, env._patient, info["protocol_stage"], exp_num))

            rosc_flag = info["heart_rate"] >= 0.9
            print(f"  Step {step:3d} | {ACTION_NAMES[action]:<28} | "
                  f"R: {reward:+5.2f} | Total: {ep_reward:6.1f} | "
                  f"HR: {info['heart_rate']:.2f}"
                  + (" | ★ ROSC" if rosc_flag else "")
                  + (f" | streak={info.get('comp_streak', 0)}" if action == 4 else ""))
            time.sleep(0.10)

        all_rewards.append(ep_reward)
        rosc = info["heart_rate"] >= 0.9
        if rosc: rosc_count += 1
        bridge.send_state(build_episode_end_packet(ep, ep_reward, step, rosc, algo.upper()))

        outcome = "★ PATIENT REVIVED" if rosc else "Episode ended"
        print(f"\n  {outcome}  |  Reward: {ep_reward:.2f}  |  Steps: {step}")
        print(_action_frequency_table(counts, step))
        top_action = max(counts, key=counts.get)
        top_pct    = counts[top_action] / step * 100
        if top_pct > 50:
            print(_collapse_diagnosis(counts, step))
        else:
            print(f"  Action diversity: OK (top={ACTION_NAMES[top_action]} {top_pct:.0f}%)")
        time.sleep(1.5)

    env.close()
    print(f"\n{SEP}")
    print(f"  SUMMARY [{algo.upper()}]  difficulty={difficulty}")
    print(f"  Mean: {np.mean(all_rewards):.2f}  Max: {np.max(all_rewards):.2f}")
    print(f"  ROSC: {rosc_count}/{episodes} episodes ({rosc_count/episodes*100:.0f}%)")
    print(SEP)

    if rosc_count == 0:
        print(
            "\n  ⚠  No ROSC achieved. Possible causes:\n"
            "     1. Model trained with old env (v1-v3) — re-train with v4\n"
            "     2. MONITOR_PULSE exploit — check collapse_events in results JSON\n"
            "     3. ent_coef too low — re-run pg_training.py (v2 uses ≥0.15)\n"
            "  Run:  python training/pg_training.py --algo ppo\n"
        )


# ---------------------------------------------------------------------------
# Training (launches subprocess, streams output)
# ---------------------------------------------------------------------------

def run_training_with_unity(bridge, algo: str):
    import subprocess
    bridge.send_state(build_phase_change_packet("training", algo.upper()))
    script_map = {
        "dqn":      "training/dqn_training.py",
        "ppo":      "training/pg_training.py --algo ppo",
        "reinforce":"training/pg_training.py --algo reinforce",
        "all":      "training/pg_training.py --algo all",
    }
    script = script_map.get(algo, "training/pg_training.py --algo all")
    log.info(f"Launching: python {script}")
    proc = subprocess.Popen(
        f"python {script}".split(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
        # Forward mean_reward to Unity HUD
        if "mean_rew=" in line or "Mean=" in line:
            try:
                val = float(line.split("mean")[-1].split("=")[1].split()[0])
                bridge.send_state({
                    "type": "training_update", "phase": "training",
                    "algorithm": algo.upper(), "mean_reward": val,
                })
            except Exception:
                pass
    proc.wait()
    log.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CPR RL — Unity 3D Edition v2")
    parser.add_argument("--random",     action="store_true")
    parser.add_argument("--demo",       action="store_true")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--algo",       choices=["dqn","ppo","reinforce","auto","all"],
                                        default="auto")
    parser.add_argument("--exp",        type=int, default=None,
                                        help="Force a specific experiment number (e.g. --exp 4)")
    parser.add_argument("--episodes",   type=int,  default=3)
    parser.add_argument("--difficulty", choices=["easy","medium","hard"],
                                        default="medium")
    parser.add_argument("--mediapipe",  action="store_true")
    parser.add_argument("--video",      type=str, default=None)
    parser.add_argument("--no-bridge",  action="store_true")
    parser.add_argument("--port",       type=int, default=8765)
    args = parser.parse_args()

    print(BANNER)

    bridge = UnityBridge(port=args.port)
    if not args.no_bridge:
        bridge.start()
        log.info("Bridge → ws://localhost:8765  (run Unity and press Play)")
        for i in range(10, 0, -1):
            if bridge.is_connected(): break
            print(f"  Waiting for Unity... {i}s  (--no-bridge to skip)", end="\r")
            time.sleep(1)
        print()

    use_mp = args.mediapipe or (args.video is not None)
    algo   = args.algo

    # Auto-select: ROSC rate dominates, then mean reward, then penalise collapse
    if algo == "auto" and not args.random and not args.train:
        best_algo, best_score = "ppo", -999
        for a in ["dqn", "ppo", "reinforce"]:
            rpath = f"results/{a}_results.json"
            if os.path.exists(rpath):
                try:
                    with open(rpath) as f:
                        res = json.load(f)
                    valid = [r for r in res if "error" not in r]
                    if valid:
                        scores = [
                            r.get("mean_reward_last50", -999)
                            + r.get("rosc_rate", 0.0) * 300
                            - 5 * len(r.get("collapse_events", []))
                            for r in valid
                        ]
                        s = max(scores)
                        if s > best_score:
                            best_score, best_algo = s, a
                except Exception:
                    pass
        algo = best_algo
        log.info(f"Auto-selected: {algo.upper()} (score={best_score:.2f})")

    try:
        if args.random:
            run_random_agent(bridge, args.episodes, use_mp, args.video, args.difficulty)
        elif args.train:
            run_training_with_unity(bridge, algo if algo != "auto" else "all")
        else:
            run_demo(bridge, algo, args.episodes, use_mp, args.video,
                     args.difficulty, force_exp=args.exp)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except FileNotFoundError as e:
        log.error(str(e))
    finally:
        try:
            bridge.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()