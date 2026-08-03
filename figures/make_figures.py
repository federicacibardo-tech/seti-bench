"""Reproduce the analysis figures from the recorded experiment results.

The measured values are embedded below so the figures can be regenerated without
rerunning the models. Produces:
    - detectability vs SNR for the reliable (low false-positive) models
    - false-positive rate per model
    - recovery vs false-alarm scatter
    - per-model detectability panels with 2-sigma error bars
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1E2761"

# Recovery rate (%) per SNR level, and false-positive rate on pure noise (%).
DATA = {
    "Qwen3.5-9B":    {"recovery": {7: 0, 9: 0, 11: 10, 13: 90, 15: 100, 17: 100,
                                   19: 100, 21: 100, 23: 100, 25: 100}, "fp": 0,
                      "color": "#2E8B57"},
    "Qwen3.5-2B":    {"recovery": {7: 10, 9: 0, 11: 50, 13: 100, 15: 100, 17: 100,
                                   19: 100, 21: 100, 23: 100, 25: 100}, "fp": 10,
                      "color": "#3CB371"},
    "Gemma 4B":      {"recovery": {7: 0, 9: 0, 11: 0, 13: 20, 15: 30, 17: 70,
                                   19: 80, 21: 80, 23: 100, 25: 78}, "fp": 0,
                      "color": "#8B4513"},
    "Qwen2.5-VL 7B": {"recovery": {7: 100, 9: 100, 11: 100, 13: 100, 15: 100, 17: 100,
                                   19: 100, 21: 100, 23: 100, 25: 100}, "fp": 100,
                      "color": "#F96167"},
    "Qwen3.5-0.8B":  {"recovery": {7: 100, 9: 100, 11: 100, 13: 100, 15: 100, 17: 100,
                                   19: 100, 21: 100, 23: 100, 25: 100}, "fp": 100,
                      "color": "#4A90D9"},
    "LLaVA 7B":      {"recovery": {7: 100, 9: 100, 11: 100, 13: 100, 15: 100, 17: 100,
                                   19: 100, 21: 100, 23: 100, 25: 100}, "fp": 100,
                      "color": "#E8873A"},
    "Gemma 2B":      {"recovery": {7: 80, 9: 90, 11: 80, 13: 90, 15: 100, 17: 100,
                                   19: 100, 21: 100, 23: 100, 25: 100}, "fp": 97,
                      "color": "#9B59B6"},
}

RELIABLE = ["Qwen3.5-9B", "Qwen3.5-2B", "Gemma 4B"]


def n_at(snr):
    return 9 if snr == 25 else 10


def two_sigma(p, n):
    return 2 * np.sqrt((p / 100) * (1 - p / 100) / n) * 100


def figure_reliable():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.patch.set_facecolor("white")
    for name in RELIABLE:
        curve = DATA[name]["recovery"]
        snrs = sorted(curve)
        rec = [curve[s] for s in snrs]
        err = [two_sigma(curve[s], n_at(s)) for s in snrs]
        ax.errorbar(snrs, rec, yerr=err, fmt="o-", color=DATA[name]["color"],
                    lw=2.2, ms=6, capsize=4, capthick=1.3,
                    label=f"{name}  (FP {DATA[name]['fp']}%)")
    ax.set_xlabel("Candidate SNR (width fixed = 3 Hz)", fontsize=12, color=NAVY)
    ax.set_ylabel("Recovery rate (%)", fontsize=12, color=NAVY)
    ax.set_title("Detectability vs SNR - reliable (low false-positive) models\n"
                 "These curves are real detection thresholds",
                 fontsize=13, color=NAVY, fontweight="bold", pad=12)
    ax.set_ylim(-5, 108)
    ax.axhline(50, color="gray", ls=":", alpha=0.4)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95,
              title="model (false positives on pure noise)")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("detectability_reliable.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def figure_false_positives():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor("white")
    order = sorted(DATA, key=lambda m: DATA[m]["fp"])
    fps = [DATA[m]["fp"] for m in order]
    colors = ["#2E8B57" if f <= 10 else "#F96167" for f in fps]
    ax.bar(range(len(order)), fps, color=colors, width=0.62, edgecolor="white", linewidth=1.5)
    for i, f in enumerate(fps):
        ax.text(i, f + 2, f"{f}%", ha="center", fontsize=11, fontweight="bold", color=NAVY)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("False positive rate on pure noise (%)", fontsize=12, color=NAVY)
    ax.set_title("False positives on noise_only - genuine detection vs guessing\n"
                 "Green = reliable (flags a candidate only when a signal is present)",
                 fontsize=13, color=NAVY, fontweight="bold", pad=12)
    ax.set_ylim(0, 112)
    ax.axhline(50, color="gray", ls=":", alpha=0.35)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("false_positives.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def figure_scatter():
    fig, ax = plt.subplots(figsize=(9.5, 7))
    fig.patch.set_facecolor("white")
    for name, d in DATA.items():
        low_snr = np.mean([d["recovery"][s] for s in [7, 9, 11, 13]])
        ax.scatter(d["fp"], low_snr, s=260, color=d["color"], edgecolor="white",
                   linewidth=2, zorder=3)
        ax.annotate(name, (d["fp"], low_snr), fontsize=10, fontweight="bold",
                    color=NAVY, xytext=(8, 6), textcoords="offset points")
    ax.axvspan(-5, 20, alpha=0.08, color="green")
    ax.text(7, 95, "USEFUL\n(detects faint\nsignals, few\nfalse alarms)", fontsize=10,
            color="#2E8B57", fontweight="bold", ha="center", va="top")
    ax.text(75, 95, "GUESSING\n(flags a candidate\non everything)", fontsize=10,
            color="#F96167", fontweight="bold", ha="center", va="top")
    ax.set_xlabel("False positive rate on pure noise (%)  ->  worse", fontsize=12, color=NAVY)
    ax.set_ylabel("Recovery at low SNR 7-13 (%)", fontsize=12, color=NAVY)
    ax.set_title("Faint-signal recovery vs false alarms\n"
                 "Only the low-FP models (left) are genuinely detecting",
                 fontsize=13, color=NAVY, fontweight="bold", pad=12)
    ax.set_xlim(-8, 112)
    ax.set_ylim(-5, 108)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("recovery_vs_false_alarm.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def figure_per_model():
    for name in RELIABLE:
        curve = DATA[name]["recovery"]
        snrs = sorted(curve)
        rec = np.array([curve[s] for s in snrs])
        err = np.array([two_sigma(curve[s], n_at(s)) for s in snrs])
        color = DATA[name]["color"]

        threshold = next((s for s in snrs if curve[s] >= 50), None)

        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        fig.patch.set_facecolor("white")
        ax.fill_between(snrs, rec - err, rec + err, color=color, alpha=0.15)
        ax.errorbar(snrs, rec, yerr=err, fmt="o-", color=color, lw=2.4, ms=7,
                    capsize=4, capthick=1.4)
        ax.axhline(50, color="gray", ls=":", alpha=0.5)
        if threshold:
            ax.axvline(threshold, color=color, ls="--", alpha=0.6)
            ax.annotate(f"detection threshold\n~SNR {threshold}", xy=(threshold, 50),
                        xytext=(threshold + 1.5, 28), fontsize=11, color=color,
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))
        ax.set_xlabel("Candidate SNR (width fixed = 3 Hz)", fontsize=12, color=NAVY)
        ax.set_ylabel("Recovery rate (%)", fontsize=12, color=NAVY)
        ax.set_title(f"{name} - detectability vs SNR\n"
                     f"False positives on pure noise: {DATA[name]['fp']}%   "
                     f"(error bars 2 sigma)",
                     fontsize=13, color=NAVY, fontweight="bold", pad=12)
        ax.set_ylim(-8, 110)
        ax.set_xticks(snrs)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        slug = name.lower().replace(".", "").replace("-", "").replace(" ", "")
        plt.savefig(f"detectability_{slug}.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()


if __name__ == "__main__":
    figure_reliable()
    figure_false_positives()
    figure_scatter()
    figure_per_model()
    print("Figures written.")
