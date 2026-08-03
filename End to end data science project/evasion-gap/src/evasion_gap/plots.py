"""Charts produced from the sweep results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CLEAN_COLOR = "#8a8a8a"
ATTACK_COLOR = "#c0392b"
FIXED_COLOR = "#2e7d5b"


def _save(fig, outfile):
    if outfile:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")


def plot_recall_by_threshold(sweep: pd.DataFrame, operating_points: list, outfile: Path = None):
    """Compare recall under each attack at both thresholds, without any defense."""
    ops = list(operating_points)
    fig, axes = plt.subplots(1, len(ops), figsize=(6.5 * len(ops), 4.2), squeeze=False)

    for ax, op in zip(axes[0], ops):
        df = sweep[
            (sweep["split"] == "toxic")
            & (sweep["defense"] == "none")
            & (sweep["operating_point"] == op.name)
        ].sort_values("rate")
        colors = [CLEAN_COLOR if a == "clean" else ATTACK_COLOR for a in df["attack"]]
        err = [df["rate"] - df["ci_low"], df["ci_high"] - df["rate"]]

        ax.barh(df["attack"], df["rate"], color=colors, xerr=err, ecolor="#444", capsize=3)
        ax.set_xlim(0, 1)
        ax.set_xlabel("recall")
        ax.set_title(
            f"{op.name}\nthreshold={op.threshold:.3f}, benign FPR={op.benign_fpr:.1%}",
            fontsize=10,
        )

    fig.suptitle("Recall under obfuscation, measured at two thresholds", y=1.0)
    fig.tight_layout()
    _save(fig, outfile)
    return fig, axes


def plot_defense_effect(
    sweep: pd.DataFrame, operating_point: str, outfile: Path = None
):
    """Show recall and false positive rate before and after normalization."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    panels = [
        ("toxic", "recall on toxic content (higher is better)", axes[0]),
        ("benign", "false positive rate on benign content (lower is better)", axes[1]),
    ]

    for split, title, ax in panels:
        df = sweep[(sweep["split"] == split) & (sweep["operating_point"] == operating_point)]
        pivot = df.pivot(index="attack", columns="defense", values="rate")
        pivot = pivot.loc[[a for a in df["attack"].unique()]].sort_values("none")

        y = range(len(pivot))
        ax.barh([i + 0.2 for i in y], pivot["none"], height=0.4,
                color=ATTACK_COLOR, label="no defense")
        ax.barh([i - 0.2 for i in y], pivot["normalized"], height=0.4,
                color=FIXED_COLOR, label="normalized")
        ax.set_yticks(list(y))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right")

    fig.suptitle(f"Effect of Unicode normalization at {operating_point}", y=1.02)
    fig.tight_layout()
    _save(fig, outfile)
    return fig, axes
