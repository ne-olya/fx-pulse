"""Forecast volatility with GARCH(1,1) and test it as a signal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import _apply_policy
from fxpulse.value_downside_experiment import causal_percentile


IDENTITY_KEYS = ("scope", "model", "gate", "horizon")


def load_config(path: Path | str = Path("configs/garch_gate_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("GARCH config must be preregistered schema_version 1")
    if config.get("gates") != ["no_gate", "low_mid_forecast_vol", "high_forecast_vol"]:
        raise ValueError("GARCH gates differ from the implementation")
    return config


def _garch_forecast(train_returns: pd.Series, test_returns: pd.Series) -> pd.Series:
    train = pd.to_numeric(train_returns, errors="coerce").dropna() * 100
    test = pd.to_numeric(test_returns, errors="coerce") * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = arch_model(train, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(
            disp="off", show_warning=False
        )
    omega = float(fitted.params["omega"])
    alpha = float(fitted.params["alpha[1]"])
    beta = float(fitted.params["beta[1]"])
    last_variance = float(fitted.conditional_volatility.iloc[-1] ** 2)
    last_return = float(train.iloc[-1])
    current_variance = omega + alpha * last_return**2 + beta * last_variance
    forecast = pd.Series(np.nan, index=test_returns.index, dtype=float)
    for index, value in test.items():
        if not np.isfinite(value):
            continue
        next_variance = omega + alpha * float(value) ** 2 + beta * current_variance
        forecast.loc[index] = max(next_variance, 1e-12)
        current_variance = next_variance
    return forecast


def build_forecasts(features: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = features.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    forecast_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for corridor in config["corridors"]:
        source = data.loc[data["corridor"].eq(corridor)].sort_values("timestamp", kind="mergesort").copy()
        for year in config["test_years"]:
            start = pd.Timestamp(year=int(year), month=1, day=1)
            end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
            train = source.loc[source["timestamp"] < start]
            test = source.loc[source["timestamp"].between(start, end, inclusive="left")].copy()
            if len(train["base__return_1"].dropna()) < int(config["minimum_garch_observations"]) or len(test) < 20:
                continue
            test["garch_variance_next"] = _garch_forecast(train["base__return_1"], test["base__return_1"])
            test["naive_variance_next"] = np.square(test["base__volatility_20"] * 100)
            test["actual_variance_next"] = np.square(test["base__return_1"].shift(-1) * 100)
            test["forecast_vol_percentile"] = causal_percentile(
                test["garch_variance_next"],
                lookback=int(config["volatility_rank_lookback"]),
                minimum_history=int(config["minimum_rank_history"]),
            )
            valid = test[["garch_variance_next", "naive_variance_next", "actual_variance_next"]].dropna()
            actual = valid["actual_variance_next"].clip(lower=1e-12)
            garch = valid["garch_variance_next"].clip(lower=1e-12)
            naive = valid["naive_variance_next"].clip(lower=1e-12)
            diagnostics.append(
                {
                    "corridor": corridor,
                    "test_year": int(year),
                    "observations": len(valid),
                    "garch_mse": float(np.square(garch - actual).mean()),
                    "naive_mse": float(np.square(naive - actual).mean()),
                    "garch_qlike": float((actual / garch + np.log(garch)).mean()),
                    "naive_qlike": float((actual / naive + np.log(naive)).mean()),
                }
            )
            forecast_parts.append(
                test[["timestamp", "corridor", "garch_variance_next", "naive_variance_next", "forecast_vol_percentile"]]
            )
    return pd.concat(forecast_parts, ignore_index=True), pd.DataFrame(diagnostics)


def gate_mask(frame: pd.DataFrame, gate: str, threshold: float) -> pd.Series:
    if gate == "no_gate":
        return pd.Series(True, index=frame.index)
    if gate == "low_mid_forecast_vol":
        return frame["forecast_vol_percentile"].le(threshold)
    if gate == "high_forecast_vol":
        return frame["forecast_vol_percentile"].gt(threshold)
    raise ValueError(f"unknown GARCH gate {gate}")


def evaluate(scores: pd.DataFrame, forecasts: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = scores.loc[
        scores["feature_set"].eq(config["score_feature_set"])
        & scores["tolerance_bps"].eq(float(config["tolerance_bps"]))
        & scores["model"].isin(config["models"])
        & scores["horizon"].isin(config["horizons"])
    ].copy()
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="raise")
    data = selected.merge(forecasts, on=["timestamp", "corridor"], how="left", validate="many_to_one")
    policy = config["adaptive_policy"]
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for keys, fold in data.groupby(["scope", "model", "horizon", "test_year"], sort=True):
        scope, model, horizon, year = keys
        fold["week"] = fold["timestamp"].dt.to_period("W").astype(str)
        weekly_rate = fold.groupby(["corridor", "week"])["target"].mean()
        fold["matched_week_hit_rate"] = [
            float(weekly_rate.loc[(corridor, week)])
            for corridor, week in zip(fold["corridor"], fold["week"], strict=True)
        ]
        exposure = int(fold["corridor"].nunique())
        duration = max(float((fold["timestamp"].max() - fold["timestamp"].min()).days) / 7, 1 / 7) * exposure
        base_candidate_parts = [
            adaptive_candidates(
                group,
                share=float(policy["top_score_share"]),
                lookback=int(policy["lookback_observations"]),
                minimum_history=int(policy["minimum_history_observations"]),
            )
            for _, group in fold.groupby("corridor", sort=False)
        ]
        base_candidates = pd.concat(base_candidate_parts, ignore_index=False)
        for gate in config["gates"]:
            candidates = base_candidates.loc[
                gate_mask(base_candidates, gate, float(config["high_vol_percentile"]))
            ]
            selected_parts = [
                _apply_policy(
                    group,
                    cooldown_days=int(policy["cooldown_days"]),
                    weekly_cap=int(policy["weekly_cap"]),
                )
                for _, group in candidates.groupby("corridor", sort=False)
            ]
            dispatched = pd.concat(selected_parts, ignore_index=False) if selected_parts else candidates.iloc[0:0]
            count = len(dispatched)
            base_rate = float(fold["target"].mean())
            hit_rate = float(dispatched["target"].mean()) if count else np.nan
            identity = {"scope": scope, "model": model, "gate": gate, "horizon": int(horizon)}
            fold_rows.append(
                {
                    **identity,
                    "test_year": int(year),
                    "test_count": len(fold),
                    "test_hits": int(fold["target"].sum()),
                    "signal_count": count,
                    "signal_hits": int(dispatched["target"].sum()) if count else 0,
                    "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                    "matched_expected_hits": float(dispatched["matched_week_hit_rate"].sum()) if count else 0.0,
                    "regret_sum_bps": float(dispatched["regret_bps"].sum()) if count else 0.0,
                    "benefit_sum_bps": float(dispatched["benefit_bps"].sum()) if count else 0.0,
                    "duration_weeks": duration,
                }
            )
            if count:
                exported = dispatched[[
                    "timestamp", "corridor", "target", "regret_bps", "benefit_bps", "score",
                    "score_threshold", "forecast_vol_percentile", "matched_week_hit_rate",
                ]].copy()
                for key, value in identity.items():
                    exported[key] = value
                exported["test_year"] = int(year)
                signal_frames.append(exported)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return pd.DataFrame(fold_rows), signals


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(list(IDENTITY_KEYS), sort=True):
        identity = dict(zip(IDENTITY_KEYS, keys, strict=True))
        count = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        hit_rate = hits / count if count else np.nan
        base_rate = base_hits / base_count if base_count else np.nan
        matched_rate = float(group["matched_expected_hits"].sum()) / count if count else np.nan
        valid_lift = group["lift"].dropna()
        rows.append(
            {
                **identity,
                "signals": count,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_random_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                "lift_vs_matched_random": hit_rate / matched_rate if count and matched_rate > 0 else np.nan,
                "regret_mean_bps": float(group["regret_sum_bps"].sum()) / count if count else np.nan,
                "benefit_mean_bps": float(group["benefit_sum_bps"].sum()) / count if count else np.nan,
                "signals_per_week": count / float(group["duration_weeks"].sum()),
                "worst_fold_lift": float(valid_lift.min()) if len(valid_lift) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    config_path: Path | str = Path("configs/garch_gate_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/garch_gate"),
) -> dict[str, object]:
    config = load_config(config_path)
    features_path = Path(config["features"])
    scores_path = Path(config["scores"])
    forecasts, diagnostics = build_forecasts(pd.read_csv(features_path), config)
    folds, signals = evaluate(pd.read_csv(scores_path), forecasts, config)
    summary_frame = summarize(folds)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output / "forecasts.csv", index=False)
    diagnostics.to_csv(output / "forecast_diagnostics.csv", index=False)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "forecast_rows": len(forecasts),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory GARCH gate; each year's parameters are fitted only on earlier returns.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/garch_gate_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/garch_gate"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
