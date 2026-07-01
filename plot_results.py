"""
LLM-NAS experiment results — multi-panel visualization
Usage: python plot_results.py
Outputs: plots/  (folder next to this script, one PNG per chart)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.join(BASE, "experiments")

TRIAL_PATHS = {
    "jannis":  os.path.join(EXP, "jannis/nas_v2/trials_index.json"),
    "helena":  os.path.join(EXP, "helena/nas_v2/trials_index.json"),
    "pol":     os.path.join(EXP, "pol/nas_v2/trials_index.json"),
    "ECG5000": os.path.join(EXP, "domains/ECG5000/nas_v2/trials_index.json"),
    "HAR":     os.path.join(EXP, "domains/HAR/nas_v2/trials_index.json"),
    "ELEC2":   os.path.join(EXP, "domains/ELEC2/nas_v2/trials_index.json"),
}

ACTION_STAT_PATHS = {
    "jannis":  os.path.join(EXP, "jannis/nas_v2/action_stats.json"),
    "helena":  os.path.join(EXP, "helena/nas_v2/action_stats.json"),
    "pol":     os.path.join(EXP, "pol/nas_v2/action_stats.json"),
    "ECG5000": os.path.join(EXP, "domains/ECG5000/nas_v2/action_stats.json"),
    "HAR":     os.path.join(EXP, "domains/HAR/nas_v2/action_stats.json"),
    "ELEC2":   os.path.join(EXP, "domains/ELEC2/nas_v2/action_stats.json"),
}

SUMMARY_FILES = {
    "main":       os.path.join(EXP, "summary_main.json"),
    "biomedical": os.path.join(EXP, "summary_biomedical.json"),
    "finance":    os.path.join(EXP, "summary_finance.json"),
}

# ── colours ───────────────────────────────────────────────────────────────────
PALETTE = {
    "jannis":  "#4C72B0",
    "helena":  "#DD8452",
    "pol":     "#55A868",
    "ECG5000": "#C44E52",
    "HAR":     "#8172B2",
    "ELEC2":   "#937860",
}

DATASET_ORDER = ["jannis", "helena", "pol", "ECG5000", "HAR", "ELEC2"]

# ── load data ─────────────────────────────────────────────────────────────────
def load_trials():
    trials = {}
    for ds, path in TRIAL_PATHS.items():
        with open(path) as f:
            raw = json.load(f)
        # handle both list and {"trials": [...]} formats
        if isinstance(raw, dict) and "trials" in raw:
            trials[ds] = raw["trials"]
        else:
            trials[ds] = raw
    return trials

def load_action_stats():
    stats = {}
    for ds, path in ACTION_STAT_PATHS.items():
        with open(path) as f:
            stats[ds] = json.load(f)
    return stats

def load_summaries():
    rows = []
    for group, path in SUMMARY_FILES.items():
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            entry["group"] = group
            rows.append(entry)
    return rows

# ── helpers ───────────────────────────────────────────────────────────────────
def running_best(values):
    best = -np.inf
    out = []
    for v in values:
        if v > best:
            best = v
        out.append(best)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def savefig(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {path}")


def main():
    trials_data   = load_trials()
    action_stats  = load_action_stats()
    summaries     = load_summaries()

    out_dir = os.path.join(BASE, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Metric comparison bar chart ───────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#F8F9FA")

    # sort by DATASET_ORDER
    summaries_sorted = sorted(summaries, key=lambda e: DATASET_ORDER.index(e["name"]))
    ds_labels = [e["name"] for e in summaries_sorted]

    methods = [
        ("CatBoost",     "catboost",        "#5B8DB8"),
        ("LightGBM",     "lightgbm",        "#7CB9E8"),
        ("Random NAS",   "random_nas",      "#AACB8B"),
        ("Optuna",       "optuna",          "#F4C875"),
        ("Naive LLM",    "naive_llm",       "#D4A0C8"),
        ("LLM-NAS",      "llm_nas_v2",      "#E07B54"),
        ("NAS Ensemble", "llm_nas_ensemble","#5DAD72"),
    ]

    n_methods = len(methods)
    x = np.arange(len(ds_labels))
    w = 0.11
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * w
    bar_kw = dict(edgecolor="white", linewidth=0.6)

    all_vals = []
    for offset, (label, key, color) in zip(offsets, methods):
        vals = [e.get(key) or 0 for e in summaries_sorted]
        all_vals.extend(vals)
        bars = ax1.bar(x + offset, vals, w, label=label, color=color, **bar_kw)

    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_labels, fontsize=10)
    ax1.set_ylabel("Primary metric", fontsize=10)
    ax1.set_title("1 · All baselines vs LLM-NAS vs NAS Ensemble", fontsize=12, fontweight="bold", pad=8)
    ax1.legend(fontsize=8, ncol=7, loc="lower right")
    ax1.set_ylim(min(all_vals) - 0.05, max(all_vals) + 0.04)
    ax1.set_facecolor("#FFFFFF")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)
    savefig(fig, "01_baseline_comparison.png", out_dir)

    # ── 2. Per-trial scatter ──────────────────────────────────────────────────
    fig, ax2 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F9FA")

    for ds in DATASET_ORDER:
        vals = [t["primary"] for t in trials_data[ds]]
        ax2.scatter(range(len(vals)), vals,
                    color=PALETTE[ds], s=30, alpha=0.75, label=ds, zorder=3)

    ax2.set_xlabel("Trial index", fontsize=10)
    ax2.set_ylabel("Primary metric", fontsize=10)
    ax2.set_title("2 · Primary metric per trial", fontsize=12, fontweight="bold", pad=8)
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_facecolor("#FFFFFF")
    ax2.grid(linestyle="--", alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)
    savefig(fig, "02_trial_scatter.png", out_dir)

    # ── 3. Running best ───────────────────────────────────────────────────────
    fig, ax3 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F9FA")

    for ds in DATASET_ORDER:
        vals = [t["primary"] for t in trials_data[ds]]
        rb   = running_best(vals)
        ax3.plot(range(len(rb)), rb, color=PALETTE[ds], lw=2, label=ds, marker="o",
                 markersize=3, markevery=4)

    ax3.set_xlabel("Trial index", fontsize=10)
    ax3.set_ylabel("Running best", fontsize=10)
    ax3.set_title("3 · Running best metric over trials", fontsize=12, fontweight="bold", pad=8)
    ax3.legend(fontsize=8, ncol=2)
    ax3.set_facecolor("#FFFFFF")
    ax3.grid(linestyle="--", alpha=0.35)
    ax3.spines[["top", "right"]].set_visible(False)
    savefig(fig, "03_running_best.png", out_dir)

    # ── 4. Action usage heatmap ───────────────────────────────────────────────
    fig, ax4 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F9FA")

    # collect all action names that have at least one usage
    all_actions = list(next(iter(action_stats.values())).keys())
    used_actions = [a for a in all_actions
                    if any(action_stats[ds][a]["count"] > 0 for ds in DATASET_ORDER)]

    matrix = np.array([[action_stats[ds][a]["count"] for a in used_actions]
                        for ds in DATASET_ORDER], dtype=float)

    im = ax4.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax4.set_xticks(range(len(used_actions)))
    ax4.set_xticklabels([a.replace("_", "\n") for a in used_actions],
                        fontsize=7, ha="center")
    ax4.set_yticks(range(len(DATASET_ORDER)))
    ax4.set_yticklabels(DATASET_ORDER, fontsize=9)
    ax4.set_title("4 · Action usage count (heatmap)", fontsize=12, fontweight="bold", pad=8)

    for i in range(len(DATASET_ORDER)):
        for j in range(len(used_actions)):
            v = int(matrix[i, j])
            if v > 0:
                ax4.text(j, i, str(v), ha="center", va="center",
                         fontsize=9, fontweight="bold",
                         color="white" if v >= matrix.max() * 0.6 else "#333")

    plt.colorbar(im, ax=ax4, shrink=0.8, label="count")
    savefig(fig, "04_action_heatmap.png", out_dir)

    # ── 5. Best-trial val_history curves ─────────────────────────────────────
    fig, ax5 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F9FA")

    for ds in DATASET_ORDER:
        best_trial = max(trials_data[ds], key=lambda t: t["primary"])
        vh = best_trial.get("val_history", [])
        if vh:
            epochs = np.linspace(0, 1, len(vh))
            ax5.plot(epochs, vh, color=PALETTE[ds], lw=1.8, label=ds)

    ax5.set_xlabel("Training progress (normalised)", fontsize=10)
    ax5.set_ylabel("Val metric", fontsize=10)
    ax5.set_title("5 · Val curve — best trial per dataset", fontsize=12, fontweight="bold", pad=8)
    ax5.legend(fontsize=8, ncol=2)
    ax5.set_facecolor("#FFFFFF")
    ax5.grid(linestyle="--", alpha=0.35)
    ax5.spines[["top", "right"]].set_visible(False)
    savefig(fig, "05_val_curves_best.png", out_dir)

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
