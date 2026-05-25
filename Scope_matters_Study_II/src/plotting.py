"""
src/plotting.py
Reproduces the three plots described in the paper:
  Plot 1 — Answer Quality   : F1 + BERTScore per language × scope
  Plot 2 — Retrieval Quality: RHR per language × scope
  Plot 3 — Decision Logic   : HP-LBS bars + best-scope annotation
"""

from typing import Dict, List

SCOPE_COLORS = {"mono": "#a78bfa", "cross": "#60a5fa", "full": "#34d399"}
STRATEGY_COLORS = {"mono": "#7C3AED", "cross": "#2563EB", "full": "#059669"}
STRAT_LABEL  = {"mono": "Mono", "cross": "Cross+EN", "full": "Full-Multi"}


def plot_all(
    rows: List[Dict],
    results_dir: str,
    prefix: str = "plot",
    lbs_threshold: float = 1.0,
) -> None:
    """Generate and save all three figures."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("  [skip] pip install matplotlib to enable plotting")
        return

    import os
    os.makedirs(f"{results_dir}/figures", exist_ok=True)

    lang_names = [r["name"] for r in rows]
    scopes     = ["mono", "cross", "full"]
    x          = np.arange(len(rows))
    w          = 0.25

    # ── Plot 1: F1 + BERTScore ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Answer Quality: F1 & BERTScore per Language × Scope\n"
        "(★ = chosen scope for that language)",
        fontsize=13,
    )
    for metric, ax, ylabel in [
        ("f1",         ax1, "F1 Score (%)"),
        ("bertscore",  ax2, "BERTScore F1 (%)"),
    ]:
        for ki, scope in enumerate(scopes):
            vals = [r.get(f"{scope}_{metric}", 0) for r in rows]
            ax.bar(x + (ki - 1) * w, vals, w,
                   label=STRAT_LABEL[scope],
                   color=SCOPE_COLORS[scope], alpha=0.85)
        for i, r in enumerate(rows):
            s   = r["strategy"]
            val = r.get(f"{s}_{metric}", 0)
            off = {"mono": -w, "cross": 0, "full": w}[s]
            ax.annotate("★", xy=(i + off, val + 0.5),
                        ha="center", fontsize=11, color="gold",
                        fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(lang_names, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(ylabel)
    plt.tight_layout()
    p = f"{results_dir}/figures/{prefix}_answer_quality.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Plot 1 → {p}")
    plt.close()

    # ── Plot 2: RHR ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        "Retrieval Hit Rate (RHR) per Language × Scope\n"
        "(fraction of queries where gold answer span is in retrieved chunks)",
        fontsize=13,
    )
    for ki, scope in enumerate(scopes):
        vals = [r.get(f"{scope}_rhr", 0) for r in rows]
        ax.bar(x + (ki - 1) * w, vals, w,
               label=STRAT_LABEL[scope],
               color=SCOPE_COLORS[scope], alpha=0.85)
    for i, r in enumerate(rows):
        s   = r["strategy"]
        val = r.get(f"{s}_rhr", 0)
        off = {"mono": -w, "cross": 0, "full": w}[s]
        ax.annotate("★", xy=(i + off, val + 0.5),
                    ha="center", fontsize=11, color="gold", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(lang_names, rotation=25, ha="right")
    ax.set_ylabel("Retrieval Hit Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = f"{results_dir}/figures/{prefix}_retrieval_quality.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Plot 2 → {p}")
    plt.close()

    # ── Plot 3: HP-LBS + Best Scope ───────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 10))
    fig.suptitle(
        "Decision Logic: Language Benefit Score (HP-F1) + Best Scope",
        fontsize=13,
    )
    lbs_cross = [r.get("lbs_cross", 0) for r in rows]
    lbs_full  = [r.get("lbs_full",  0) for r in rows]
    w2 = 0.35
    bars_c = ax_top.bar(x - w2/2, lbs_cross, w2,
                        label="LBS Cross+EN (HP)",
                        color=SCOPE_COLORS["cross"], alpha=0.85)
    bars_f = ax_top.bar(x + w2/2, lbs_full,  w2,
                        label="LBS Full-Multi (HP)",
                        color=SCOPE_COLORS["full"],  alpha=0.85)
    ax_top.axhline(0,             color="black", linewidth=0.8, linestyle="--")
    ax_top.axhline(lbs_threshold, color="red",   linewidth=1.2, linestyle=":",
                   label=f"Threshold ({lbs_threshold})")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(lang_names, rotation=25, ha="right")
    ax_top.set_ylabel("HP-LBS (HP-F1 gain over mono)")
    ax_top.legend(fontsize=9)
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.set_title("HP-LBS — positive = wider scope helps")
    for bar in list(bars_c) + list(bars_f):
        h = bar.get_height()
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.3 if h >= 0 else -0.9),
            f"{h:+.1f}", ha="center", va="bottom", fontsize=8,
        )

    strat_order = {"mono": 1, "cross": 2, "full": 3}
    bar_colors  = [STRATEGY_COLORS[r["strategy"]] for r in rows]
    ax_bot.bar(x, [strat_order[r["strategy"]] for r in rows],
               color=bar_colors, alpha=0.85, width=0.6)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(lang_names, rotation=25, ha="right")
    ax_bot.set_yticks([1, 2, 3])
    ax_bot.set_yticklabels(["Mono", "Cross+EN", "Full-Multi"])
    ax_bot.set_ylabel("Best Scope")
    ax_bot.set_title("Chosen Retrieval Scope per Language")
    ax_bot.grid(axis="y", alpha=0.3)
    for i, r in enumerate(rows):
        ax_bot.text(i, strat_order[r["strategy"]] + 0.08,
                    STRAT_LABEL[r["strategy"]],
                    ha="center", va="bottom", fontsize=9,
                    color="white", fontweight="bold")
    patches = [mpatches.Patch(color=STRATEGY_COLORS[s], label=STRAT_LABEL[s])
               for s in ["mono", "cross", "full"]]
    ax_bot.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    p = f"{results_dir}/figures/{prefix}_decision_logic.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Plot 3 → {p}")
    plt.close()
