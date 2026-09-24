"""Run the full readout: experiment checks, effect estimates, uplift models, budget policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from offerlift import data, evaluation, experiment, policy, uplift

ROOT = Path(__file__).resolve().parents[1]


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"Wrote {path.relative_to(ROOT)}")


def experiment_checks(frame, config) -> dict:
    data_cfg, exp_cfg = config["data"], config["experiment"]
    srm = experiment.sample_ratio_mismatch(
        frame[data_cfg["arm_column"]], data_cfg["expected_allocation"], exp_cfg["srm_alpha"]
    )
    balance = {}
    power = {}
    control = frame[frame[data_cfg["arm_column"]] == data_cfg["control_arm"]]
    for arm in data_cfg["treatment_arms"]:
        subset = data.contrast(frame, data_cfg["arm_column"], data_cfg["control_arm"], arm)
        smd = experiment.standardized_mean_differences(
            data.feature_matrix(subset), subset["treated"]
        )
        balance[arm] = {
            "max_smd": float(smd.max()),
            "worst_feature": str(smd.index[0]),
            "balanced": bool(smd.max() < exp_cfg["balance_smd_threshold"]),
        }
    for metric in [exp_cfg["primary_metric"], "conversion"]:
        baseline = float(control[metric].mean())
        power[metric] = {
            "control_rate": baseline,
            "n_per_arm": int(len(control)),
            "minimum_detectable_effect": experiment.minimum_detectable_effect(
                baseline, len(control), exp_cfg["alpha"], exp_cfg["power"]
            ),
        }
    return {"sample_ratio_mismatch": srm.__dict__, "covariate_balance": balance, "power": power}


def effect_estimates(frame, config) -> dict:
    data_cfg, exp_cfg = config["data"], config["experiment"]
    metrics = [exp_cfg["primary_metric"], *exp_cfg["secondary_metrics"]]
    results: dict = {}
    primary_p = {}
    for arm in data_cfg["treatment_arms"]:
        subset = data.contrast(frame, data_cfg["arm_column"], data_cfg["control_arm"], arm)
        results[arm] = {}
        for metric in metrics:
            estimate = experiment.difference_in_means(
                subset[metric], subset["treated"], exp_cfg["alpha"]
            )
            results[arm][metric] = estimate.to_dict()
        adjusted = experiment.cuped_adjust(subset["spend"], subset[exp_cfg["cuped_covariate"]])
        cuped = experiment.difference_in_means(adjusted, subset["treated"], exp_cfg["alpha"])
        raw_se = results[arm]["spend"]["standard_error"]
        results[arm]["spend_cuped"] = {
            **cuped.to_dict(),
            "variance_reduction": float(1 - (cuped.standard_error / raw_se) ** 2),
        }
        primary_p[arm] = results[arm][exp_cfg["primary_metric"]]["p_value"]
    results["primary_holm_adjusted_p"] = experiment.holm_adjust(primary_p)
    return results


def uplift_readout(frame, config) -> dict:
    data_cfg, up_cfg = config["data"], config["uplift"]
    subset = data.contrast(
        frame, data_cfg["arm_column"], data_cfg["control_arm"], up_cfg["treatment_arm"]
    )
    features = data.feature_matrix(subset)
    outcome = subset[up_cfg["outcome"]].to_numpy()
    treated = subset["treated"].to_numpy()
    stratify = treated * 2 + outcome
    x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(
        features,
        outcome,
        treated,
        test_size=up_cfg["test_size"],
        random_state=up_cfg["random_state"],
        stratify=stratify,
    )
    models = {
        "t_learner_logistic": uplift.TLearner(uplift.logistic_classifier()),
        "t_learner_gbm": uplift.TLearner(),
        "transformed_outcome_gbm": uplift.TransformedOutcome(),
    }
    rng = np.random.default_rng(up_cfg["random_state"])
    scores = {"random_score": rng.random(len(y_test))}
    for name, model in models.items():
        scores[name] = model.fit(x_train, y_train, t_train).predict_uplift(x_test)

    ranking = {}
    for name, score in scores.items():
        low, high = evaluation.bootstrap_interval(
            evaluation.qini_coefficient, score, y_test, t_test, random_state=up_cfg["random_state"]
        )
        ranking[name] = {
            "qini_coefficient": evaluation.qini_coefficient(score, y_test, t_test),
            "qini_ci_low": low,
            "qini_ci_high": high,
        }
    table = policy.budget_table(
        {name: score for name, score in scores.items() if name != "random_score"},
        y_test,
        t_test,
        up_cfg["budget_fractions"],
        alpha=config["experiment"]["alpha"],
        random_state=up_cfg["random_state"],
    )
    return {
        "treatment_arm": up_cfg["treatment_arm"],
        "outcome": up_cfg["outcome"],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "ranking": ranking,
        "budget_policy": table.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frame = data.load(ROOT / config["data"]["path"])

    checks = experiment_checks(frame, config)
    write_json(checks, ROOT / "reports/metrics/experiment_checks.json")
    if checks["sample_ratio_mismatch"]["mismatch"]:
        raise SystemExit("Sample ratio mismatch detected; do not trust effect estimates.")
    write_json(effect_estimates(frame, config), ROOT / "reports/metrics/effects.json")
    write_json(uplift_readout(frame, config), ROOT / "reports/metrics/uplift.json")


if __name__ == "__main__":
    main()
