"""
extract_results.py
==================
Run this script to print a full summary of all training results
in a copy-paste-ready format for the report.

Usage:
    python extract_results.py

Outputs:
  - Console: formatted tables for DQN, REINFORCE, PPO
  - results/training_summary.json  — full data for report generation
  - results/plot_data.json         — reward curves for plotting
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_FILES = {
    "DQN":       "results/dqn_results.json",
    "REINFORCE": "results/reinforce_results.json",
    "PPO":       "results/ppo_results.json",
}

# ── Column definitions per algorithm ────────────────────────────────────────
DQN_COLS       = ["exp","learning_rate","gamma","buffer_size","batch_size",
                   "exploration_fraction","net_arch","mean_reward_last50","max_reward"]
REINFORCE_COLS = ["exp","learning_rate","gamma","use_baseline","entropy_coef",
                   "hidden_arch","mean_reward_last50","max_reward","total_episodes"]
PPO_COLS       = ["exp","learning_rate","gamma","ent_coef","clip_range",
                   "n_steps","batch_size","n_epochs","net_arch",
                   "mean_reward_last50","max_reward"]

COL_MAP = {"DQN": DQN_COLS, "REINFORCE": REINFORCE_COLS, "PPO": PPO_COLS}

# ── Headers for display ──────────────────────────────────────────────────────
DQN_HEADERS = {
    "exp":"Exp","learning_rate":"LR","gamma":"Gamma","buffer_size":"Buffer",
    "batch_size":"Batch","exploration_fraction":"Expl. Frac","net_arch":"Net Arch",
    "mean_reward_last50":"Mean Reward","max_reward":"Max Reward"
}
REINFORCE_HEADERS = {
    "exp":"Exp","learning_rate":"LR","gamma":"Gamma","use_baseline":"Baseline",
    "entropy_coef":"Entropy β","hidden_arch":"Hidden","mean_reward_last50":"Mean Reward",
    "max_reward":"Max Reward","total_episodes":"Episodes"
}
PPO_HEADERS = {
    "exp":"Exp","learning_rate":"LR","gamma":"Gamma","ent_coef":"Ent Coef",
    "clip_range":"Clip ε","n_steps":"N Steps","batch_size":"Batch",
    "n_epochs":"Epochs","net_arch":"Net Arch",
    "mean_reward_last50":"Mean Reward","max_reward":"Max Reward"
}
HEADER_MAP = {"DQN": DQN_HEADERS, "REINFORCE": REINFORCE_HEADERS, "PPO": PPO_HEADERS}


def load(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return [r for r in data if "error" not in r]
    except Exception as e:
        print(f"  WARNING: could not load {path}: {e}")
        return []


def fmt_val(v):
    """Format a value for table display."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 0.01:
            return f"{v:.0e}"
        return f"{v:.4g}"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    return str(v)


def print_table(algo, results, cols, headers):
    if not results:
        print(f"  No completed experiments found.")
        return

    # Find best
    best = max(results, key=lambda r: r.get("mean_reward_last50", -999))

    # Build rows
    rows = []
    for r in results:
        row = [fmt_val(r.get(c)) for c in cols]
        rows.append((r.get("exp"), row))

    # Column widths
    col_widths = [max(len(headers.get(c, c)), max(len(row[i]) for _, row in rows))
                  for i, c in enumerate(cols)]

    # Header line
    header_line = " | ".join(
        headers.get(c, c).ljust(col_widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * w for w in col_widths)

    print(f"\n  {header_line}")
    print(f"  {sep}")
    for exp_num, row in rows:
        is_best = (exp_num == best.get("exp"))
        line = " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(row))
        marker = " ★" if is_best else ""
        print(f"  {line}{marker}")

    print(f"\n  Best: Experiment {best['exp']}  "
          f"Mean Reward: {best.get('mean_reward_last50', '?'):.3f}  "
          f"Max: {best.get('max_reward', '?'):.3f}")


def print_summary():
    print("\n" + "="*72)
    print("  CPR RL TRAINING RESULTS SUMMARY")
    print("="*72)

    summary = {}
    plot_data = {}

    for algo, path in RESULT_FILES.items():
        print(f"\n{'─'*72}")
        print(f"  {algo}")
        print(f"{'─'*72}")

        results = load(path)
        cols    = COL_MAP[algo]
        headers = HEADER_MAP[algo]

        if not results:
            print(f"  No results found at: {path}")
            print(f"  Run: python training/{algo.lower()}_training.py")
            summary[algo] = {"status": "not_trained", "experiments": 0}
            continue

        print(f"  Loaded {len(results)} experiment(s)")
        print_table(algo, results, cols, headers)

        best = max(results, key=lambda r: r.get("mean_reward_last50", -999))
        summary[algo] = {
            "status": "trained",
            "experiments": len(results),
            "best_exp": best.get("exp"),
            "best_mean_reward": round(best.get("mean_reward_last50", 0), 3),
            "best_max_reward": round(best.get("max_reward", 0), 3),
            "all_mean_rewards": [
                round(r.get("mean_reward_last50", 0), 3) for r in results],
            "all_max_rewards": [
                round(r.get("max_reward", 0), 3) for r in results],
            "best_params": {k: best.get(k) for k in cols if k not in
                            ("exp","mean_reward_last50","max_reward","reward_curve",
                             "entropy_curve","total_episodes")},
        }

        # Collect reward curves for plotting
        plot_data[algo] = {
            "best_reward_curve":   best.get("reward_curve", []),
            "all_final_rewards":   [r.get("mean_reward_last50", 0) for r in results],
            "entropy_curve":       best.get("entropy_curve", []),  # REINFORCE/PPO only
        }

    # ── Comparison ────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  ALGORITHM COMPARISON")
    print(f"{'='*72}")
    print(f"\n  {'Algorithm':<12} {'Experiments':<14} {'Best Exp':<10} "
          f"{'Mean Reward':<14} {'Max Reward'}")
    print(f"  {'─'*12} {'─'*14} {'─'*10} {'─'*14} {'─'*12}")
    for algo, s in summary.items():
        if s["status"] == "trained":
            print(f"  {algo:<12} {s['experiments']:<14} "
                  f"{str(s['best_exp']):<10} "
                  f"{s['best_mean_reward']:<14.3f} "
                  f"{s['best_max_reward']:.3f}")
        else:
            print(f"  {algo:<12} {'Not trained'}")

    # ── Save files ────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("results/plot_data.json", "w") as f:
        json.dump(plot_data, f, indent=2)

    print(f"\n  Saved → results/training_summary.json")
    print(f"  Saved → results/plot_data.json")
    print("\n  Copy the tables above into your report.")
    print("  The JSON files contain exact values for all report figures.\n")


if __name__ == "__main__":
    print_summary()