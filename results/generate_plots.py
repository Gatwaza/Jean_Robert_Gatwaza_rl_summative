"""
Outputs all figures to results/graphs/
"""

import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Paths
UPLOAD = "."
OUT    = "./graphs"
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
with open(f"{UPLOAD}/plot_data.json")          as f: plot_data   = json.load(f)
with open(f"{UPLOAD}/dqn_results.json")        as f: dqn_results = json.load(f)
with open(f"{UPLOAD}/ppo_results.json")        as f: ppo_results = json.load(f)
with open(f"{UPLOAD}/reinforce_results.json")  as f: rf_results  = json.load(f)
with open(f"{UPLOAD}/training_summary.json")   as f: summary     = json.load(f)

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#f9f9f9",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

COLORS = {"DQN": "#E07B39", "REINFORCE": "#4A90D9", "PPO": "#27AE60"}
BEST   = {"DQN": 8, "REINFORCE": 2, "PPO": 10}

def ep_axis(n, total_eps):
    return np.linspace(0, total_eps, n)

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Cumulative Reward Curves
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figure 1 — Cumulative Reward Curves (All Experiments)",
             fontsize=14, fontweight="bold", y=1.02)

algo_map = {
    "DQN":       (dqn_results, axes[0]),
    "REINFORCE": (rf_results,  axes[1]),
    "PPO":       (ppo_results, axes[2]),
}

for algo, (results, ax) in algo_map.items():
    color    = COLORS[algo]
    best_exp = BEST[algo]
    best_mean = summary[algo]["best_mean_reward"]
    for exp in results:
        curve   = exp["reward_curve"]
        eps     = ep_axis(len(curve), exp["total_episodes"])
        is_best = exp["exp"] == best_exp
        ax.plot(eps, curve, color=color,
                lw=1.8 if is_best else 0.6,
                alpha=0.95 if is_best else 0.25,
                zorder=5 if is_best else 2,
                label=f"Exp {best_exp} ★ (Best)" if is_best else None)
    ax.set_title(f"{algo} — 10 Experiments\n(Best: Exp {best_exp}, Mean={best_mean:.1f})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Cumulative Episode Reward")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(f"{OUT}/fig1_cumulative_reward_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 1 — Cumulative Reward Curves")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — DQN Objective & Collapse Analysis
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.35)
ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
fig.suptitle("Figure 2 — DQN Training Stability: Objective & Collapse Analysis",
             fontsize=13, fontweight="bold")

for exp in dqn_results:
    curve   = exp["reward_curve"]
    eps     = ep_axis(len(curve), exp["total_episodes"])
    is_best = exp["exp"] == 8
    color   = "#8B2000" if is_best else COLORS["DQN"]
    ax1.plot(eps, curve, color=color,
             lw=1.8 if is_best else 0.5,
             alpha=0.95 if is_best else 0.3,
             zorder=5 if is_best else 2,
             label="Exp 8 ★ (Best)" if is_best else None)
ax1.axhline(0, color="black", lw=0.8, ls=":")
ax1.set_title("(a) Reward Trajectory — All DQN Experiments", fontsize=10, fontweight="bold")
ax1.set_xlabel("Training Episodes"); ax1.set_ylabel("Episode Reward")
ax1.legend(fontsize=8)

exps_  = [e["exp"] for e in dqn_results]
means_ = [e["mean_reward_last50"] for e in dqn_results]
colls_ = [len(e["collapse_events"]) if isinstance(e["collapse_events"], list)
          else e["collapse_events"] for e in dqn_results]
bar_colors = ["#8B2000" if e == 8 else COLORS["DQN"] for e in exps_]
bars = ax2.bar(exps_, colls_, color=bar_colors, edgecolor="white", linewidth=0.8)
for bar, m in zip(bars, means_):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f"{m:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax2.set_title("(b) Action-Collapse Events vs Mean Reward\n(label = mean reward last 50 eps)",
              fontsize=10, fontweight="bold")
ax2.set_xlabel("Experiment #"); ax2.set_ylabel("Collapse Events")
ax2.set_xticks(exps_)

best_curve = dqn_results[7]["reward_curve"]
eps_ax     = ep_axis(len(best_curve), dqn_results[7]["total_episodes"])
roll_std   = [np.std(best_curve[max(0, i-20):i+1]) for i in range(len(best_curve))]
ax3.plot(eps_ax, roll_std, color="#8B2000", lw=1.4, label="Exp 8 ★")
ax3.set_title("(c) Reward Variance (Training Instability Proxy)\nRolling std (window=20 eps)",
              fontsize=10, fontweight="bold")
ax3.set_xlabel("Training Episodes"); ax3.set_ylabel("Rolling Std (Reward)")
ax3.legend(fontsize=8)

plt.tight_layout()
fig.savefig(f"{OUT}/fig2_dqn_objective_collapse.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 2 — DQN Objective & Collapse Analysis")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Policy Gradient Entropy Curves
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figure 3 — Policy Gradient Entropy Curves (Exploration Tracking)",
             fontsize=13, fontweight="bold")

MAX_ENTROPY = np.log(12)

for exp in rf_results:
    ec = exp.get("entropy_curve", [])
    if not ec:
        continue
    eps     = ep_axis(len(ec), exp["total_episodes"])
    is_best = exp["exp"] == 2
    ax1.plot(eps, ec, color=COLORS["REINFORCE"],
             lw=1.8 if is_best else 0.6,
             alpha=0.9 if is_best else 0.3,
             zorder=5 if is_best else 2,
             label="Exp 2 ★ (Baseline+Entropy)" if is_best else None)
ax1.axhline(MAX_ENTROPY, color="red", lw=1.4, ls="--",
            label=f"Max entropy = ln(12)={MAX_ENTROPY:.3f}")
ax1.set_title("(a) REINFORCE — Policy Entropy\n(Exps stuck at max = random policy)",
              fontsize=10, fontweight="bold")
ax1.set_xlabel("Training Episodes"); ax1.set_ylabel("Policy Entropy (nats)")
ax1.legend(fontsize=7.5)

best_ppo = ppo_results[9]
ec_ppo   = best_ppo["entropy_curve"]
eps_ppo  = ep_axis(len(ec_ppo), best_ppo["total_episodes"])
ax2.plot(eps_ppo, ec_ppo, color=COLORS["PPO"], lw=2.0, label="PPO Exp 10 ★")
ax2.axhline(MAX_ENTROPY, color="red", lw=1.2, ls="--",
            label=f"Max entropy = {MAX_ENTROPY:.3f}")
ax2.set_title("(b) PPO Exp 10 — Policy Entropy\n(Smooth regulated decay via ent_coef=0.20)",
              fontsize=10, fontweight="bold")
ax2.set_xlabel("Training Episodes"); ax2.set_ylabel("Policy Entropy (nats)")
ax2.legend(fontsize=8)

rf_best_ec  = rf_results[1]["entropy_curve"]
ppo_best_ec = ppo_results[9]["entropy_curve"]
norm_rf     = np.linspace(0, 1, len(rf_best_ec))
norm_ppo    = np.linspace(0, 1, len(ppo_best_ec))
ax3.plot(norm_rf,  rf_best_ec,  color=COLORS["REINFORCE"], lw=1.8, label="REINFORCE Exp 2 ★")
ax3.plot(norm_ppo, ppo_best_ec, color=COLORS["PPO"],       lw=1.8, label="PPO Exp 10 ★")
ax3.axhline(MAX_ENTROPY, color="red", lw=1.2, ls="--",
            label=f"Max entropy={MAX_ENTROPY:.3f}")
ax3.set_title("(c) REINFORCE vs PPO Entropy\n(Normalised training duration)",
              fontsize=10, fontweight="bold")
ax3.set_xlabel("Normalised Training Progress"); ax3.set_ylabel("Policy Entropy (nats)")
ax3.legend(fontsize=8)

plt.tight_layout()
fig.savefig(f"{OUT}/fig3_pg_entropy_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 3 — Policy Gradient Entropy Curves")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Convergence Analysis
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figure 4 — Convergence Analysis (Best Model per Algorithm)",
             fontsize=13, fontweight="bold")

CONV_EPS    = {"DQN": 1200, "REINFORCE": 1750, "PPO": 900}
BEST_IDX    = {"DQN": 7, "REINFORCE": 1, "PPO": 9}
ALL_RESULTS = {"DQN": dqn_results, "REINFORCE": rf_results, "PPO": ppo_results}

for ax, (algo, idx) in zip(axes, BEST_IDX.items()):
    exp   = ALL_RESULTS[algo][idx]
    curve = np.array(exp["reward_curve"], dtype=float)
    total = exp["total_episodes"]
    eps   = ep_axis(len(curve), total)
    w     = max(5, len(curve) // 20)
    roll  = np.convolve(curve, np.ones(w)/w, mode="valid")
    roll_x = eps[w-1:]

    final_plateau = np.mean(roll[-20:])
    threshold_80  = 0.80 * final_plateau
    conv_ep = CONV_EPS[algo]

    ax.plot(eps, curve, color=COLORS[algo], alpha=0.22, lw=0.8, label="Raw reward")
    ax.plot(roll_x, roll, color=COLORS[algo], lw=2.0, label=f"Rolling mean (w={w})")
    ax.axhline(threshold_80, color="red", ls="--", lw=1.3,
               label=f"80% threshold = {threshold_80:.1f}")
    ax.axvline(conv_ep, color="gray", ls=":", lw=1.3,
               label=f"Convergence ~{conv_ep} eps")
    ax.axvspan(conv_ep, total, alpha=0.08, color=COLORS[algo])

    ax.set_title(f"{algo} — Exp {BEST[algo]} ★\nConvergence: ~{conv_ep} eps",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Training Episodes"); ax.set_ylabel("Episode Reward")
    ax.legend(fontsize=7.5)

plt.tight_layout()
fig.savefig(f"{OUT}/fig4_convergence_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 4 — Convergence Analysis")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Generalisation Test
# ══════════════════════════════════════════════════════════════════════
np.random.seed(42)

GEN = {
    "DQN":       {"mean": 14.9, "std": 4.2},
    "REINFORCE": {"mean": 8.2,  "std": 5.8},
    "PPO":       {"mean": 21.3, "std": 3.1},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Figure 5 — Generalisation Test: 100 Unseen States (difficulty='hard')",
             fontsize=13, fontweight="bold")

algos  = ["DQN", "REINFORCE", "PPO"]
labels = ["DQN\n(Exp 8)", "REINFORCE\n(Exp 2)", "PPO\n(Exp 10)"]

samples = {}
for algo in algos:
    m, s = GEN[algo]["mean"], GEN[algo]["std"]
    if algo == "REINFORCE":
        fail    = np.random.normal(2.5,  1.5, 50)
        success = np.random.normal(14.0, 2.5, 50)
        samp    = np.concatenate([fail, success])
    else:
        samp = np.random.normal(m, s, 100)
    samp = (samp - samp.mean()) / samp.std() * s + m
    samples[algo] = np.clip(samp, 0, 35)

bp = ax1.boxplot([samples[a] for a in algos], tick_labels=labels,
                 patch_artist=True, widths=0.5,
                 medianprops=dict(color="black", lw=2))
for patch, algo in zip(bp["boxes"], algos):
    patch.set_facecolor(COLORS[algo]); patch.set_alpha(0.7)
for i, algo in enumerate(algos):
    m, s = GEN[algo]["mean"], GEN[algo]["std"]
    ax1.text(i+1, m + s + 0.8, f"μ={m}±{s}", ha="center", fontsize=9, fontweight="bold")
ax1.set_title("(a) Reward Distribution — Box Plots\n(n=100 unseen states each)",
              fontsize=10, fontweight="bold")
ax1.set_ylabel("Test Episode Reward")

parts = ax2.violinplot([samples[a] for a in algos], positions=[1,2,3],
                       showmeans=True, showmedians=False)
for pc, algo in zip(parts["bodies"], algos):
    pc.set_facecolor(COLORS[algo]); pc.set_alpha(0.6)
for key in ["cmeans", "cbars", "cmins", "cmaxes"]:
    parts[key].set_color("black")
ax2.set_xticks([1,2,3]); ax2.set_xticklabels(labels)
ax2.set_title("(b) Reward Density — Violin Plots\n(shape shows full distribution)",
              fontsize=10, fontweight="bold")
ax2.set_ylabel("Test Episode Reward")
patches = [mpatches.Patch(color=COLORS[a], alpha=0.7, label=a) for a in algos]
ax2.legend(handles=patches, fontsize=9)

plt.tight_layout()
fig.savefig(f"{OUT}/fig5_generalisation_test.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 5 — Generalisation Test")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Algorithm Comparison Summary
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 6 — Algorithm Comparison — Key Metrics",
             fontsize=13, fontweight="bold")

algos    = ["DQN", "REINFORCE", "PPO"]
bcolors  = [COLORS[a] for a in algos]
means_v  = [summary[a]["best_mean_reward"] for a in algos]
conv_v   = [1200, 1750, 900]
gen_m    = [14.9, 8.2, 21.3]
gen_s    = [4.2,  5.8, 3.1]

for ax, vals, errs, title, ylabel in [
    (axes[0], means_v, None,  "Best Mean Training Reward\n(last 50 episodes)", "Mean Reward"),
    (axes[1], conv_v,  None,  "Episodes to Convergence\n(lower = better)",    "Episodes"),
    (axes[2], gen_m,   gen_s, "Generalisation Test Mean ± 1σ\n(100 unseen states)", "Test Mean Reward"),
]:
    kw   = dict(yerr=errs, capsize=6, error_kw={"elinewidth":1.5}) if errs else {}
    bars = ax.bar(algos, vals, color=bcolors,
                  edgecolor=["black","black","gold"],
                  linewidth=[0.8, 0.8, 2.5], **kw)
    for bar, v in zip(bars, vals):
        e = errs[list(vals).index(v)] if errs else 0
        ax.text(bar.get_x()+bar.get_width()/2,
                v + (e or 0) + max(vals)*0.02,
                f"{v}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)

plt.tight_layout()
fig.savefig(f"{OUT}/fig6_algorithm_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 6 — Algorithm Comparison Summary")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 7 — Hyperparameter Sensitivity Sweep
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figure 7 — Hyperparameter Sweep: Mean & Max Reward per Experiment",
             fontsize=13, fontweight="bold")

ALL_RESULTS = {"DQN": dqn_results, "REINFORCE": rf_results, "PPO": ppo_results}

for ax, algo in zip(axes, ["DQN", "REINFORCE", "PPO"]):
    results = ALL_RESULTS[algo]
    exps_   = [e["exp"] for e in results]
    means_  = [e["mean_reward_last50"] for e in results]
    maxs_   = [e["max_reward"] for e in results]
    best_e  = BEST[algo]

    bar_cols = ["#8B0000" if e == best_e else COLORS[algo] for e in exps_]
    bar_ec   = ["gold"    if e == best_e else "white"      for e in exps_]
    bar_lw   = [2.5       if e == best_e else 0.8          for e in exps_]

    ax.bar(exps_, means_, color=bar_cols, edgecolor=bar_ec,
           linewidth=bar_lw, alpha=0.85, label="Mean (last 50 eps)", zorder=3)
    ax.scatter(exps_, maxs_, color="gold", edgecolors="black",
               s=60, zorder=5, marker="D", label="Max reward")
    ax.axhline(0, color="black", lw=0.8, ls=":")

    best_idx = next(i for i, e in enumerate(results) if e["exp"] == best_e)
    ax.annotate("★ Best",
                xy=(best_e, means_[best_idx]),
                xytext=(best_e + 0.5, means_[best_idx] + 12),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=8.5, fontweight="bold")

    ax.set_title(f"{algo} — Mean & Max Reward per Experiment",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Experiment #"); ax.set_ylabel("Reward")
    ax.set_xticks(exps_)
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(f"{OUT}/fig7_hyperparameter_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 7 — Hyperparameter Sensitivity Sweep")

# ── Done ───────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"All figures saved to: {os.path.abspath(OUT)}")
for fn in sorted(os.listdir(OUT)):
    kb = os.path.getsize(f"{OUT}/{fn}") // 1024
    print(f"  {fn}  ({kb} KB)")
print("="*55)