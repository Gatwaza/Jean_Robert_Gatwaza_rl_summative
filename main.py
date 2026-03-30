"""
main.py — CPR Position Assessment RL  (Unity 3D Edition)
=========================================================
Entry point with full Unity 3D visualization via WebSocket bridge.

Phases:
  --random     → Random agent explores (demo before any training)
  --demo       → Best trained model, Unity 3D + terminal verbose
  --train      → Run all 30 experiments with live Unity tracking

Usage:
    python main.py --random                  # Demo env without model
    python main.py --demo                    # Best model auto-selected
    python main.py --demo --algo ppo         # Specific algorithm
    python main.py --train --algo dqn        # Train DQN with Unity tracking
    python main.py --demo --no-bridge        # Run without Unity (terminal only)
    python main.py --demo --episodes 5
    python main.py --demo --mediapipe        # Use MediaPipe / CPR-Coach video
    python main.py --demo --video cpr.mp4    # Use a specific video file
"""

import os, sys, json, time, argparse, numpy as np, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env   import CPREnv, ACTION_NAMES
from environment.unity_bridge import (
    UnityBridge, build_state_packet,
    build_phase_change_packet, build_episode_end_packet,
)

logging.basicConfig(level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("Main")

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║     CPR POSITION ASSESSMENT — REINFORCEMENT LEARNING SYSTEM      ║
║   Unity 3D Visualization  ·  MediaPipe Pose  ·  DQN/REINF/PPO   ║
╚══════════════════════════════════════════════════════════════════╝
"""
SEP = "─" * 70


def load_best_model(algo):
    result_map = {"dqn": "results/dqn_results.json",
                  "ppo": "results/ppo_results.json",
                  "reinforce": "results/reinforce_results.json"}
    rpath = result_map.get(algo)
    if not rpath or not os.path.exists(rpath):
        raise FileNotFoundError(f"No results for {algo}. Train first: python training/{algo}_training.py")
    with open(rpath) as f:
        results = json.load(f)
    valid = [r for r in results if "error" not in r]
    if not valid:
        raise ValueError(f"No successful {algo} experiments.")
    best = max(valid, key=lambda r: r.get("mean_reward_last50", -999))
    exp_num = best["exp"]
    log.info(f"Best {algo.upper()} → Experiment {exp_num} (mean={best['mean_reward_last50']:.2f})")

    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(f"models/dqn/dqn_exp_{exp_num}_final"), "sb3", exp_num
    elif algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(f"models/pg/ppo/ppo_exp_{exp_num}_final"), "sb3", exp_num
    elif algo == "reinforce":
        import torch
        from training.pg_training import PolicyNet
        ckpt = torch.load(f"models/pg/reinforce/reinforce_exp_{exp_num}.pt", map_location="cpu")
        info = next(r for r in valid if r["exp"] == exp_num)
        policy = PolicyNet(53, 12, eval(info.get("hidden_arch", "[64, 64]")))
        policy.load_state_dict(ckpt["policy_state"])
        policy.eval()
        return policy, "reinforce", exp_num


def get_landmark_stream(use_mediapipe, video_path=None):
    """Returns a generator of landmark arrays (51,)."""
    if use_mediapipe:
        from environment.mediapipe_extractor import LandmarkExtractor, get_best_extractor
        if video_path:
            return LandmarkExtractor(source="video", path=video_path).stream()
        return get_best_extractor(prefer_video=True).stream()
    from environment.mediapipe_extractor import LandmarkExtractor
    return LandmarkExtractor(source="synthetic").stream()


def run_random_agent(bridge, episodes, use_mediapipe, video_path=None):
    log.info("PHASE: RANDOM AGENT — Environment exploration (no model)")
    bridge.send_state(build_phase_change_packet("random", "RANDOM"))
    lm_gen = get_landmark_stream(use_mediapipe, video_path)
    env = CPREnv(max_steps=200, difficulty="medium")
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False; ep_reward = 0.0; step = 0
        print(f"\n{SEP}\n  RANDOM EPISODE {ep}/{episodes}\n{SEP}")
        while not done:
            action = env.action_space.sample()
            lm = next(lm_gen)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated; ep_reward += reward; step += 1
            bridge.send_state(build_state_packet(
                "random", "RANDOM", ep, step, action, ACTION_NAMES[action],
                reward, ep_reward, lm, env._patient, info["protocol_stage"]))
            print(f"  Step {step:3d} | {ACTION_NAMES[action]:<24} | "
                  f"Reward: {reward:+5.2f} | HR: {info['heart_rate']:.2f}")
            time.sleep(0.08)
        bridge.send_state(build_episode_end_packet(ep, ep_reward, step,
            info["heart_rate"] >= 0.9, "RANDOM"))
        print(f"\n  Episode ended | Total: {ep_reward:.2f} | Steps: {step}")
    env.close()


def run_demo(bridge, algo, episodes, use_mediapipe, video_path=None):
    log.info(f"PHASE: DEMO — Best {algo.upper()} model")
    model, model_type, exp_num = load_best_model(algo)
    bridge.send_state(build_phase_change_packet("demo", algo.upper(), exp_num))
    lm_gen = get_landmark_stream(use_mediapipe, video_path)
    env = CPREnv(max_steps=200, difficulty="medium")
    all_rewards = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False; ep_reward = 0.0; step = 0
        print(f"\n{SEP}\n  DEMO EPISODE {ep}/{episodes}  [{algo.upper()}]\n{SEP}")
        action_counts: dict[str, int] = {}
        while not done:
            if model_type == "sb3":
                # FIX (bug 2): deterministic=True locks the policy to its argmax
                # action permanently when it has collapsed — hiding the problem.
                # Use stochastic sampling in demo so collapse becomes immediately
                # visible in the action distribution printed at episode end.
                action, _ = model.predict(obs, deterministic=False)
                action = int(action)
            else:
                import torch
                with torch.no_grad():
                    dist = model(torch.FloatTensor(obs).unsqueeze(0))
                    action = dist.sample().item()
            action_counts[ACTION_NAMES[action]] = action_counts.get(ACTION_NAMES[action], 0) + 1
            lm = next(lm_gen)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated; ep_reward += reward; step += 1
            bridge.send_state(build_state_packet(
                "demo", algo.upper(), ep, step, action, ACTION_NAMES[action],
                reward, ep_reward, lm, env._patient, info["protocol_stage"], exp_num))
            rosc_flag = info["heart_rate"] >= 0.9
            print(f"  Step {step:3d} | {ACTION_NAMES[action]:<24} | "
                  f"Reward: {reward:+5.2f} | Total: {ep_reward:6.1f} | "
                  f"HR: {info['heart_rate']:.2f}"
                  + (" | ROSC ★" if rosc_flag else ""))
            time.sleep(0.12)

        all_rewards.append(ep_reward)
        rosc = info["heart_rate"] >= 0.9
        bridge.send_state(build_episode_end_packet(ep, ep_reward, step, rosc, algo.upper()))
        print(f"\n  {'★ ROSC — PATIENT REVIVED' if rosc else 'Episode ended'}  |  "
              f"Reward: {ep_reward:.2f}  |  Steps: {step}")
        # FIX (bug 2 cont.): surface action collapse explicitly.
        # If one action dominates >80% of steps the policy has collapsed and the
        # model needs to be retrained with higher ent_coef.
        top_action, top_count = max(action_counts.items(), key=lambda x: x[1])
        collapse_pct = top_count / max(step, 1) * 100
        if collapse_pct > 80:
            print(f"  ⚠  Action collapse: '{top_action}' = {top_count}/{step} steps "
                  f"({collapse_pct:.0f}%). Policy likely undertrained — raise ent_coef.")
        else:
            top4 = sorted(action_counts.items(), key=lambda x: -x[1])[:4]
            print("  Action mix: " + " | ".join(f"{a}: {c}" for a, c in top4))
        time.sleep(1.5)

    env.close()
    print(f"\n{SEP}\n  SUMMARY [{algo.upper()}]  "
          f"Mean: {np.mean(all_rewards):.2f}  "
          f"Max: {np.max(all_rewards):.2f}\n{SEP}")


def run_training_with_unity(bridge, algo):
    import subprocess
    bridge.send_state(build_phase_change_packet("training", algo.upper()))
    script = {"dqn": "training/dqn_training.py",
               "ppo": "training/pg_training.py --algo ppo",
               "reinforce": "training/pg_training.py --algo reinforce",
               "all": "training/pg_training.py --algo all"}.get(algo, "training/pg_training.py --algo all")
    log.info(f"Running: python {script}")
    proc = subprocess.Popen(f"python {script}".split(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        if "Mean reward" in line:
            try:
                bridge.send_state({"type": "training_update", "phase": "training",
                                    "algorithm": algo.upper(),
                                    "mean_reward": float(line.split(":")[-1].strip().split()[0])})
            except Exception:
                pass
    proc.wait()
    log.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="CPR RL — Unity 3D Edition")
    parser.add_argument("--random",    action="store_true", help="Random agent demo")
    parser.add_argument("--demo",      action="store_true", help="Demo best model in Unity")
    parser.add_argument("--train",     action="store_true", help="Train with Unity tracking")
    parser.add_argument("--algo",      choices=["dqn","ppo","reinforce","auto","all"], default="auto")
    parser.add_argument("--episodes",  type=int, default=3)
    parser.add_argument("--mediapipe", action="store_true", help="Use MediaPipe / CPR-Coach video")
    parser.add_argument("--video",     type=str, default=None, help="Path to CPR video file")
    parser.add_argument("--no-bridge", action="store_true", help="Disable Unity bridge")
    parser.add_argument("--port",      type=int, default=8765)
    args = parser.parse_args()

    print(BANNER)

    bridge = UnityBridge(port=args.port)
    if not args.no_bridge:
        bridge.start()
        log.info("Unity bridge listening on ws://localhost:8765")
        log.info("→ Open Unity project and press Play. Waiting up to 10s...")
        for i in range(10, 0, -1):
            if bridge.is_connected(): break
            print(f"  Waiting for Unity... {i}s  (run with --no-bridge to skip)", end="\r")
            time.sleep(1)
        print()

    use_mp = args.mediapipe or (args.video is not None)
    algo   = args.algo

    if algo == "auto" and not args.random and not args.train:
        best_algo, best_mean = "ppo", -999
        for a in ["dqn","ppo","reinforce"]:
            rpath = f"results/{a}_results.json"
            if os.path.exists(rpath):
                with open(rpath) as f:
                    res = json.load(f)
                valid = [r for r in res if "error" not in r]
                if valid:
                    m = max(r.get("mean_reward_last50",-999) for r in valid)
                    if m > best_mean: best_mean, best_algo = m, a
        algo = best_algo
        log.info(f"Auto-selected: {algo.upper()} (mean={best_mean:.2f})")

    try:
        if args.random:
            run_random_agent(bridge, args.episodes, use_mp, args.video)
        elif args.train:
            run_training_with_unity(bridge, algo if algo != "auto" else "all")
        else:
            run_demo(bridge, algo, args.episodes, use_mp, args.video)
    except KeyboardInterrupt:
        log.info("Interrupted.")
    except FileNotFoundError as e:
        log.error(str(e))
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()