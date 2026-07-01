"""
LLM-NAS: 25 шагов поиска с подписями операций.
Один файл = один датасет.

Usage: python plot_nas_steps.py
Output: plots/steps_<dataset>.png
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def op_label(op):
    if op == "cold_start":               return "cold\nstart"
    if op == "propose":                  return "propose"
    if op == "random_refine_fallback":   return "fallback"
    if op.startswith("refine:"):
        action = op.split(":", 1)[1]
        if action in ("None", ""):       return None   # не подписываем пустые refine
        return action.replace("_", "\n")
    return op

def load_trials(path):
    with open(path) as f:
        raw = json.load(f)
    return raw["trials"] if isinstance(raw, dict) and "trials" in raw else raw

def running_best(values):
    best, out = -np.inf, []
    for v in values:
        best = max(best, v)
        out.append(best)
    return out

def main():
    out_dir = os.path.join(BASE, "plots")
    os.makedirs(out_dir, exist_ok=True)

    for ds, path in TRIAL_PATHS.items():
        trials = load_trials(path)
        vals   = [t["primary"] for t in trials]
        span   = max(vals) - min(vals) if max(vals) != min(vals) else 0.01

        fig, ax = plt.subplots(figsize=(13, 5))

        ax.plot(range(len(vals)), vals, color="#4C72B0", lw=1.5,
                marker="o", markersize=6, zorder=3)

        # подписи операций — чередуем выше/ниже чтобы не перекрывались
        for i, t in enumerate(trials):
            label = op_label(t["op"])
            if label is None:
                continue
            above = (i % 2 == 0)
            yoff  = span * 0.12 if above else -span * 0.18
            va    = "bottom" if above else "top"
            ax.annotate(
                label,
                xy=(i, vals[i]),
                xytext=(i, vals[i] + yoff),
                ha="center", va=va,
                fontsize=6.5,
                color="#333",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6),
            )

        ax.set_title(f"{ds}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trial", fontsize=10)
        ax.set_ylabel("Primary metric", fontsize=10)
        ax.set_xlim(-0.5, len(vals) - 0.5)
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"steps_{ds}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()
