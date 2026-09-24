"""Draw the README figures from reports/metrics, in light and dark variants."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports/metrics"
FIGURES = ROOT / "reports/figures"

# Blue, orange and green, picked to stay distinguishable for color-blind readers,
# with separate shades for light and dark backgrounds.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}
MODEL_LABELS = {
    "t_learner_logistic": "T-learner, logistic",
    "t_learner_gbm": "T-learner, gradient boosting",
    "transformed_outcome_gbm": "Transformed outcome",
}


def load(name: str) -> dict:
    return json.loads((METRICS / name).read_text())


def new_figure(theme: dict, title: str, subtitle: str, height: float = 4.2):
    fig, ax = plt.subplots(figsize=(8, height), dpi=160)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    fig.text(0.02, 0.965, title, fontsize=13, fontweight="bold", color=theme["ink"], va="top")
    fig.text(0.02, 0.895, subtitle, fontsize=9.5, color=theme["ink_2"], va="top")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name: str, mode: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}_{mode}.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {path.relative_to(ROOT)}")


def segment_lift(theme: dict, mode: str) -> None:
    profile = load("targeting_profile.json")
    effects = load("effects.json")
    rows = [r for r in profile["segment_lifts"] if r["segment"] == "purchased"]
    order = {"both": 0, "mens only": 1, "womens only": 2}
    labels = {"both": "Both categories", "mens only": "Men's only", "womens only": "Women's only"}
    rows.sort(key=lambda r: order[r["level"]], reverse=True)
    overall = effects["Mens E-Mail"]["visit"]["absolute_lift"] * 100

    fig, ax = new_figure(
        theme,
        "Customers who bought both categories respond about twice as strongly",
        "Increase in two-week visit rate from the men's email, by what the customer bought "
        "last year\n(percentage points, 95% confidence interval)",
        height=3.6,
    )
    fig.subplots_adjust(left=0.2, right=0.95, top=0.74, bottom=0.14)
    ax.grid(visible=False)
    ax.axvline(overall, color=theme["axis"], linewidth=1)
    ax.text(overall, 2.55, f"All customers +{overall:.1f}", color=theme["ink_2"],
            fontsize=8.5, ha="center", va="bottom")
    for y, row in enumerate(rows):
        color = theme["series"][0] if row["level"] == "both" else theme["muted"]
        lift, low, high = (row[k] * 100 for k in ("lift", "ci_low", "ci_high"))
        ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(lift, y, "o", color=color, markersize=8,
                markeredgecolor=theme["surface"], markeredgewidth=1.5)
        ax.text(high + 0.35, y, f"+{lift:.1f}  ({row['n']:,} customers)",
                color=theme["ink"], fontsize=9, va="center")
    ax.set_yticks(range(len(rows)), [labels[r["level"]] for r in rows], color=theme["ink"])
    ax.tick_params(axis="y", labelcolor=theme["ink"], labelsize=10)
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlabel("Percentage points", color=theme["muted"], fontsize=9)
    save(fig, "segment_lift", mode)


def budget_gain(theme: dict, mode: str) -> None:
    uplift = load("uplift.json")
    rows = [r for r in uplift["budget_policy"] if r["policy"] != "random"]
    budgets = sorted({r["budget_fraction"] for r in rows})
    models = list(MODEL_LABELS)

    fig, ax = new_figure(
        theme,
        "Only one model beats random targeting, and only at a 10% budget",
        "Extra visits per 1,000 customers compared with emailing the same number at random\n"
        "(95% paired bootstrap interval). Single 70/30 split, 12,784 test customers",
        height=4.4,
    )
    fig.subplots_adjust(left=0.1, right=0.97, top=0.74, bottom=0.2)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.axhline(0, color=theme["axis"], linewidth=1.2)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Extra visits vs random (0 = random)", color=theme["muted"], fontsize=9)
    offsets = [-0.22, 0, 0.22]
    for m, model in enumerate(models):
        color = theme["series"][m]
        for b, budget in enumerate(budgets):
            row = next(r for r in rows if r["policy"] == model and r["budget_fraction"] == budget)
            x = b + offsets[m]
            ax.plot([x, x], [row["gain_ci_low"], row["gain_ci_high"]], color=color,
                    linewidth=2, solid_capstyle="round")
            ax.plot(x, row["gain_vs_random_per_customers"], "o", color=color, markersize=7,
                    markeredgecolor=theme["surface"], markeredgewidth=1.5,
                    label=MODEL_LABELS[model] if b == 0 else None)
            if model == "t_learner_logistic" and budget == budgets[0]:
                ax.annotate(
                    f"+{row['gain_vs_random_per_customers']:.1f}",
                    (x, row["gain_ci_high"]), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color=theme["ink"],
                )
    ax.set_xticks(range(len(budgets)), [f"Email {b:.0%} of the list" for b in budgets],
                  color=theme["ink"])
    ax.tick_params(axis="x", labelcolor=theme["ink"], labelsize=9.5)
    ax.set_xlim(-0.5, len(budgets) - 0.5)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False,
                       fontsize=9, handletextpad=0.3, columnspacing=1.5)
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    save(fig, "budget_gain", mode)


def gain_curves(theme: dict, mode: str) -> None:
    curves = load("uplift.json")["gain_curves"]
    fractions = curves["fraction_targeted"]
    x = [f * 100 for f in fractions]
    total = curves["random_score"][-1]

    fig, ax = new_figure(
        theme,
        "The one model that leads only does so on the first fifth of the list",
        "Extra visits per 1,000 customers from emailing the top-scored share of the list, "
        "compared with\nemailing the same share at random. Single 70/30 split, "
        "12,784 test customers",
        height=4.4,
    )
    fig.subplots_adjust(left=0.1, right=0.97, top=0.74, bottom=0.24)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color=theme["axis"], linewidth=1.2)
    for m, model in enumerate(MODEL_LABELS):
        gap = [value - f * total for value, f in zip(curves[model], fractions)]
        ax.plot(x, gap, color=theme["series"][m], linewidth=2, label=MODEL_LABELS[model])
        if model == "t_learner_logistic":
            peak = max(range(len(gap)), key=gap.__getitem__)
            ax.annotate(f"+{gap[peak]:.1f} at {x[peak]:.0f}% emailed", (x[peak], gap[peak]),
                        xytext=(8, 4), textcoords="offset points", fontsize=9,
                        fontweight="bold", color=theme["ink"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the list emailed, highest-scored first (%)", color=theme["muted"],
                  fontsize=9)
    ax.set_ylabel("Extra visits vs random (0 = random)", color=theme["muted"], fontsize=9)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False,
                       fontsize=9, handlelength=1.5, columnspacing=1.5)
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    save(fig, "gain_curves", mode)


def confirmation(theme: dict, mode: str) -> None:
    conf = load("confirmation.json")
    original = next(
        r for r in load("uplift.json")["budget_policy"]
        if r["policy"] == conf["model"] and r["budget_fraction"] == conf["budget_fraction"]
    )
    repeats = sorted(conf["per_repeat"], key=lambda r: r["gain_per_customers"])

    fig, ax = new_figure(
        theme,
        "The 10% budget result holds up, at a smaller size",
        "Extra visits per 1,000 customers from emailing the logistic T-learner's top 10% "
        "instead of a random 10%\n20 repeated 5-fold cross-fits over all 42,613 customers, "
        "design fixed before the run",
        height=5.2,
    )
    fig.subplots_adjust(left=0.27, right=0.95, top=0.8, bottom=0.1)
    ax.axvline(0, color=theme["axis"], linewidth=1.2)
    rows = [
        ("Original single split", original["gain_vs_random_per_customers"],
         original["gain_ci_low"], original["gain_ci_high"], theme["series"][1], 9, 2),
        ("Confirmation (median)", conf["gain_per_customers"], conf["ci_low"],
         conf["ci_high"], theme["series"][0], 9, 2.5),
    ]
    y = len(repeats) + 3
    labels, ticks = [], []
    for label, value, low, high, color, size, width in rows:
        ax.plot([low, high], [y, y], color=color, linewidth=width, solid_capstyle="round")
        ax.plot(value, y, "o", color=color, markersize=size,
                markeredgecolor=theme["surface"], markeredgewidth=1.5)
        ax.text(high + 0.3, y, f"+{value:.1f}  ({low:.1f} to {high:.1f})",
                color=theme["ink"], fontsize=9, va="center")
        labels.append(label)
        ticks.append(y)
        y -= 1.6
    for i, row in enumerate(repeats):
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=theme["muted"], linewidth=1)
        ax.plot(row["gain_per_customers"], i, "o", color=theme["muted"], markersize=4)
    ticks.append((len(repeats) - 1) / 2)
    labels.append("Each of the 20 repeats\n(97.5% intervals)")
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", labelcolor=theme["ink"], labelsize=9.5)
    ax.set_xlim(-2, 16)
    ax.set_ylim(-1, len(repeats) + 4)
    ax.text(0.15, -0.9, "Random targeting", color=theme["ink_2"], fontsize=8.5, va="bottom")
    ax.set_xlabel("Extra visits per 1,000 customers vs random targeting",
                  color=theme["muted"], fontsize=9)
    save(fig, "confirmation", mode)


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    for mode, theme in THEMES.items():
        segment_lift(theme, mode)
        budget_gain(theme, mode)
        gain_curves(theme, mode)
        confirmation(theme, mode)


if __name__ == "__main__":
    main()
