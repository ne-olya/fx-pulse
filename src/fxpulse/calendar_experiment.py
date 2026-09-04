"""Ablate simple and expanded calendar features without using future data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import _add_corridor_dummies, _make_model, _purged_train
from fxpulse.temporal_sequence_experiment import _evaluate_fold, summarize


BASIC_CALENDAR = (
    "base__weekday_sin",
    "base__weekday_cos",
    "base__month_sin",
    "base__month_cos",
    "base__month_end",
)
BASE_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")


def load_config(path: Path | str = Path("configs/calendar_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("calendar config must be preregistered schema_version 1")
    if config.get("feature_sets") != ["no_calendar", "basic_calendar", "expanded_calendar"]:
        raise ValueError("calendar feature sets differ from the implementation")
    return config


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    timestamp = data["timestamp"]
    weekday = timestamp.dt.dayofweek
    day = timestamp.dt.day
    days_in_month = timestamp.dt.days_in_month
    days_to_end = days_in_month - day
    for value in range(7):
        data[f"calendar__weekday_{value}"] = weekday.eq(value).astype(float)
    data["calendar__day_sin"] = np.sin(2 * np.pi * (day - 1) / days_in_month)
    data["calendar__day_cos"] = np.cos(2 * np.pi * (day - 1) / days_in_month)
    data["calendar__days_to_month_end"] = days_to_end / 31
    data["calendar__first_5_days"] = day.le(5).astype(float)
    data["calendar__last_5_days"] = days_to_end.lt(5).astype(float)
    data["calendar__middle_10_15"] = day.between(10, 15).astype(float)
    data["calendar__tax_proxy_25_28"] = day.between(25, 28).astype(float)
    data["calendar__quarter_end_window"] = (
        timestamp.dt.month.isin([3, 6, 9, 12]) & days_to_end.lt(5)
    ).astype(float)
    return data


def feature_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    base = [column for column in frame if column.startswith(BASE_PREFIXES)]
    if feature_set == "no_calendar":
        return [column for column in base if column not in BASIC_CALENDAR]
    if feature_set == "basic_calendar":
        return base
    if feature_set == "expanded_calendar":
        return [*base, *[column for column in frame if column.startswith("calendar__")]]
    raise ValueError(f"unknown calendar feature set {feature_set}")


def evaluate(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_calendar_features(data)
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for scope_kind in config["scopes"]:
        scopes = ["pooled"] if scope_kind == "pooled" else list(config["corridors"])
        for scope in scopes:
            source = frame if scope == "pooled" else frame.loc[frame["corridor"].eq(scope)].copy()
            for horizon in config["horizons"]:
                outcome = f"outcome__regret_{int(horizon)}"
                benefit = f"outcome__benefit_{int(horizon)}"
                usable = source.loc[source[outcome].notna()].copy()
                usable["target"] = usable[outcome].le(float(config["tolerance_bps"])).astype(int)
                usable["regret_bps"] = usable[outcome]
                usable["benefit_bps"] = usable[benefit]
                for feature_set in config["feature_sets"]:
                    raw_columns = feature_columns(usable, feature_set)
                    for year in config["test_years"]:
                        start = pd.Timestamp(year=int(year), month=1, day=1)
                        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                        train = _purged_train(usable, start, int(horizon))
                        test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                        columns = list(raw_columns)
                        if scope == "pooled":
                            train, test, columns = _add_corridor_dummies(
                                train, test, columns, list(config["corridors"])
                            )
                        if (
                            len(train) < int(config["minimum_training_observations"])
                            or len(test) < 20
                            or train["target"].nunique() < 2
                        ):
                            continue
                        model = _make_model(
                            str(config["model"]),
                            iterations=int(config["model_iterations"]),
                            seed=int(config["random_seed"]) + int(year),
                        )
                        model.fit(train[columns], train["target"])
                        test["score"] = model.predict_proba(test[columns])[:, 1]
                        test["week"] = test["timestamp"].dt.to_period("W").astype(str)
                        weekly_rate = test.groupby(["corridor", "week"])["target"].mean()
                        test["matched_week_hit_rate"] = [
                            float(weekly_rate.loc[(corridor, week)])
                            for corridor, week in zip(test["corridor"], test["week"], strict=True)
                        ]
                        selected = _evaluate_fold(test, policy=config["adaptive_policy"])
                        count = len(selected)
                        base_rate = float(test["target"].mean())
                        hit_rate = float(selected["target"].mean()) if count else np.nan
                        exposure = int(test["corridor"].nunique())
                        duration = max(float((test["timestamp"].max() - test["timestamp"].min()).days) / 7, 1 / 7) * exposure
                        identity = {
                            "scope": scope,
                            "model": str(config["model"]),
                            "feature_set": feature_set,
                            "horizon": int(horizon),
                        }
                        fold_rows.append(
                            {
                                **identity,
                                "test_year": int(year),
                                "test_count": len(test),
                                "test_hits": int(test["target"].sum()),
                                "signal_count": count,
                                "signal_hits": int(selected["target"].sum()) if count else 0,
                                "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                                "matched_expected_hits": float(selected["matched_week_hit_rate"].sum()) if count else 0.0,
                                "regret_sum_bps": float(selected["regret_bps"].sum()) if count else 0.0,
                                "benefit_sum_bps": float(selected["benefit_bps"].sum()) if count else 0.0,
                                "duration_weeks": duration,
                            }
                        )
                        if count:
                            exported = selected[[
                                "timestamp", "corridor", "target", "regret_bps", "benefit_bps",
                                "score", "score_threshold", "matched_week_hit_rate",
                            ]].copy()
                            for key, value in identity.items():
                                exported[key] = value
                            exported["test_year"] = int(year)
                            signal_frames.append(exported)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return pd.DataFrame(fold_rows), signals


def run(
    *,
    config_path: Path | str = Path("configs/calendar_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/calendar"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    folds, signals = evaluate(pd.read_csv(input_path), config)
    summary_frame = summarize(folds)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory calendar ablation; 25-28 is only a date proxy, not an audited tax calendar.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/calendar_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/calendar"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
