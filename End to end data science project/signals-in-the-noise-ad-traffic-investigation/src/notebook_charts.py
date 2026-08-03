from __future__ import annotations

import html

import numpy as np


def line_chart_svg(
    labels: list[str],
    expected: np.ndarray,
    actual: np.ndarray,
    title: str,
    y_label: str,
) -> str:
    """Return a small dependency-free SVG calibration chart."""
    width, height = 900, 430
    left, right, top, bottom = 80, 35, 55, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(float(np.max(expected)), float(np.max(actual))) * 1.15
    maximum = max(maximum, 1e-6)
    x_values = np.linspace(left, left + plot_width, len(labels))

    def y(value: float) -> float:
        return top + plot_height - value / maximum * plot_height

    expected_points = " ".join(
        f"{x:.1f},{y(float(value)):.1f}" for x, value in zip(x_values, expected)
    )
    actual_points = " ".join(
        f"{x:.1f},{y(float(value)):.1f}" for x, value in zip(x_values, actual)
    )
    grid: list[str] = []
    for step in range(6):
        value = maximum * step / 5
        y_position = y(value)
        grid.append(
            f'<line x1="{left}" y1="{y_position:.1f}" x2="{left + plot_width}" '
            f'y2="{y_position:.1f}" stroke="#dfe5df"/>'
        )
        grid.append(
            f'<text x="{left - 12}" y="{y_position + 4:.1f}" text-anchor="end" '
            f'fill="#64716a" font-size="12">{100 * value:.1f}%</text>'
        )
    x_labels = "".join(
        f'<text x="{x:.1f}" y="{top + plot_height + 28}" text-anchor="middle" '
        f'fill="#64716a" font-size="12">{html.escape(label)}</text>'
        for x, label in zip(x_values, labels)
    )
    expected_circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y(float(value)):.1f}" r="4" fill="#24634b"/>'
        for x, value in zip(x_values, expected)
    )
    actual_circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y(float(value)):.1f}" r="4" fill="#c66a3d"/>'
        for x, value in zip(x_values, actual)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
<rect width="100%" height="100%" rx="14" fill="#fffdf8"/>
<text x="{left}" y="30" fill="#17231d" font-family="Georgia,serif" font-size="22" font-weight="700">{html.escape(title)}</text>
{''.join(grid)}
<polyline points="{expected_points}" fill="none" stroke="#24634b" stroke-width="3"/>{expected_circles}
<polyline points="{actual_points}" fill="none" stroke="#c66a3d" stroke-width="3"/>{actual_circles}
{x_labels}
<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" fill="#64716a" font-size="13">Prediction decile</text>
<text x="18" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2})" fill="#64716a" font-size="13">{html.escape(y_label)}</text>
<circle cx="{width - 230}" cy="29" r="5" fill="#24634b"/><text x="{width - 218}" y="34" fill="#33423a" font-size="13">Expected</text>
<circle cx="{width - 125}" cy="29" r="5" fill="#c66a3d"/><text x="{width - 113}" y="34" fill="#33423a" font-size="13">Observed</text>
</svg>'''


def risk_histogram_svg(train: np.ndarray, test: np.ndarray) -> str:
    """Return a train-versus-held-out risk distribution as inline SVG."""
    width, height = 900, 410
    left, right, top, bottom = 70, 35, 55, 65
    plot_width, plot_height = width - left - right, height - top - bottom
    bins = np.linspace(0, 1, 21)
    train_counts, _ = np.histogram(train, bins=bins)
    test_counts, _ = np.histogram(test, bins=bins)
    train_share = train_counts / max(train_counts.sum(), 1)
    test_share = test_counts / max(test_counts.sum(), 1)
    maximum = max(float(train_share.max()), float(test_share.max())) * 1.15
    group_width = plot_width / len(train_counts)
    bar_width = group_width * 0.34
    bars: list[str] = []
    for index, (train_value, test_value) in enumerate(zip(train_share, test_share)):
        center = left + group_width * (index + 0.5)
        train_height = train_value / maximum * plot_height
        test_height = test_value / maximum * plot_height
        bars.append(
            f'<rect x="{center - bar_width - 1:.1f}" y="{top + plot_height - train_height:.1f}" '
            f'width="{bar_width:.1f}" height="{train_height:.1f}" fill="#6a9b82"/>'
        )
        bars.append(
            f'<rect x="{center + 1:.1f}" y="{top + plot_height - test_height:.1f}" '
            f'width="{bar_width:.1f}" height="{test_height:.1f}" fill="#d48655"/>'
        )
    ticks = "".join(
        f'<text x="{left + plot_width * value:.1f}" y="{top + plot_height + 26}" '
        f'text-anchor="middle" fill="#64716a" font-size="12">{value:.1f}</text>'
        for value in np.linspace(0, 1, 6)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Quality risk score distribution">
<rect width="100%" height="100%" rx="14" fill="#fffdf8"/>
<text x="{left}" y="30" fill="#17231d" font-family="Georgia,serif" font-size="22" font-weight="700">Quality-risk score distribution</text>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#9aa69f"/>
{''.join(bars)}{ticks}
<text x="{left + plot_width / 2}" y="{height - 15}" text-anchor="middle" fill="#64716a" font-size="13">Training-period empirical percentile</text>
<rect x="{width - 230}" y="20" width="12" height="12" fill="#6a9b82"/><text x="{width - 212}" y="31" fill="#33423a" font-size="13">Train</text>
<rect x="{width - 145}" y="20" width="12" height="12" fill="#d48655"/><text x="{width - 127}" y="31" fill="#33423a" font-size="13">Held out</text>
</svg>'''


def case_rates_svg(
    labels: list[str], observed: np.ndarray, expected: np.ndarray
) -> str:
    """Return a grouped rate comparison for one review case."""
    width, height = 760, 370
    left, right, top, bottom = 75, 35, 55, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(float(observed.max()), float(expected.max())) * 1.2
    centers = np.linspace(
        left + plot_width * 0.25, left + plot_width * 0.75, len(labels)
    )
    bar_width = 65
    bars: list[str] = []
    for center, label, observed_value, expected_value in zip(
        centers, labels, observed, expected
    ):
        observed_height = float(observed_value) / maximum * plot_height
        expected_height = float(expected_value) / maximum * plot_height
        bars.extend(
            [
                f'<rect x="{center - bar_width - 4:.1f}" y="{top + plot_height - observed_height:.1f}" '
                f'width="{bar_width}" height="{observed_height:.1f}" rx="4" fill="#c66a3d"/>',
                f'<rect x="{center + 4:.1f}" y="{top + plot_height - expected_height:.1f}" '
                f'width="{bar_width}" height="{expected_height:.1f}" rx="4" fill="#24634b"/>',
                f'<text x="{center - bar_width / 2 - 4:.1f}" y="{top + plot_height - observed_height - 8:.1f}" '
                f'text-anchor="middle" fill="#33423a" font-size="12">{100 * observed_value:.2f}%</text>',
                f'<text x="{center + bar_width / 2 + 4:.1f}" y="{top + plot_height - expected_height - 8:.1f}" '
                f'text-anchor="middle" fill="#33423a" font-size="12">{100 * expected_value:.2f}%</text>',
                f'<text x="{center:.1f}" y="{top + plot_height + 28}" text-anchor="middle" '
                f'fill="#33423a" font-size="13">{html.escape(label)}</text>',
            ]
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Top case expected versus observed rates">
<rect width="100%" height="100%" rx="14" fill="#fffdf8"/>
<text x="{left}" y="30" fill="#17231d" font-family="Georgia,serif" font-size="22" font-weight="700">First review case: observed versus expected</text>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#9aa69f"/>
{''.join(bars)}
<rect x="{width - 220}" y="20" width="12" height="12" fill="#c66a3d"/><text x="{width - 202}" y="31" fill="#33423a" font-size="13">Observed</text>
<rect x="{width - 120}" y="20" width="12" height="12" fill="#24634b"/><text x="{width - 102}" y="31" fill="#33423a" font-size="13">Expected</text>
</svg>'''
