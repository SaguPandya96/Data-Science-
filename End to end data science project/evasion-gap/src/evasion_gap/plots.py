"""Result plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CLEAN_COLOR = "#8a8a8a"
ATTACK_COLOR = "#c0392b"


def plot_robustness(sweep: pd.DataFrame, operating_points: list, outfile: Path = None):
    """One panel per operating point: recall by attack, with bootstrap error bars.

    Side by side is the point -- the same attacks look harmless at a floor-level
    threshold and devastating at a realistic one.
    """
    ops = list(operating_points)
    fig, axes = plt.subplots(1, len(ops), figsize=(6.5 * len(ops), 4.4), squeeze=False)

    for ax, op in zip(axes[0], ops):
        df = sweep[sweep["operating_point"] == op.name].sort_values("recall")
        colors = [CLEAN_COLOR if a == "clean" else ATTACK_COLOR for a in df["attack"]]
        err = [df["recall"] - df["ci_low"], df["ci_high"] - df["recall"]]

        ax.barh(df["attack"], df["recall"], color=colors, xerr=err, ecolor="#444", capsize=3)
        ax.set_xlim(0, 1)
        ax.set_xlabel("recall")
        ax.set_title(
            f"{op.name}\nthreshold={op.threshold:.3f}  FPR={op.benign_fpr:.1%}",
            fontsize=10,
        )

    fig.suptitle("Evasion is invisible at a floor threshold, severe at a shipping one", y=1.0)
    fig.tight_layout()

    if outfile:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
    return fig, axes
