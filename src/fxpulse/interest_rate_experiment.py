"""Test lagged official CBR key-rate features as a causal ablation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import _add_corridor_dummies, _make_model, _purged_train
from fxpulse.temporal_sequence_experiment import _evaluate_fold, summarize


BASE_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")


def load_config(path: Path | str = Path("configs/interest_rate_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("interest-rate config must be preregistered schema_version 1")
    if config.get("feature_sets") != ["without_interest", "plus_interest"]:
        raise ValueError("interest-rate feature sets differ from the implementation")
    if int(config.get("strict_lag_calendar_days", 0)) < 1:
        raise ValueError("interest-rate data must be lagged by at least one calendar day")
    return config


def add_interest_features(frame: pd.DataFrame, key_rate: pd.DataFrame, *, lag_days: int) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    rates = key_rate[["rate_date", "key_rate_pct"]].copy()
    rates["rate_date"] = pd.to_datetime(rates["rate_date"], errors="raise")
    rates["key_rate_pct"] = pd.to_numeric(rates["key_rate_pct"], errors="raise")
    rates = rates.sort_values("rate_date", kind="mergesort").drop_duplicates("rate_date", keep="last")
    rates["interest__key_rate_change_5"] = rates["key_rate_pct"].diff(5)
    rates["interest__key_rate_change_20"] = rates["key_rate_pct"].diff(20)
    daily_move = rates["key_rate_pct"].diff()
    rates["interest__last_rate_move"] = daily_move.where(daily_move.ne(0)).ffill().fillna(0)
    change_date = rates["rate_date"].where(daily_move.ne(0)).ffill()
    rates["interest__days_since_rate_change"] = (
        rates["rate_date"] - change_date
    ).dt.days.clip(lower=0)
    rates = rates.rename(columns={"key_rate_pct": "interest__key_rate_pct", "rate_date": "known_date"})
    # Moving the effective date forward is an intentionally conservative
    # availability rule: the same-day model never sees a rate row.
    rates["known_at"] = rates["known_date"] + pd.to_timedelta(int(lag_days), unit="D")
    right_columns = [
        "known_at",
        "known_date",
        "interest__key_rate_pct",
        "interest__key_rate_change_5",
        "interest__key_rate_change_20",
        "interest__last_rate_move",
        "interest__days_since_rate_change",
    ]
    merged = pd.merge_asof(
        data.sort_values("timestamp", kind="mergesort"),
        rates[right_columns].sort_values("known_at", kind="mergesort"),
        left_on="timestamp",
        right_on="known_at",
        direction="backward",
    )
    if (merged["known_at"].dropna() > merged.loc[merged["known_at"].notna(), "timestamp"]).any():
        raise AssertionError("future key-rate row entered interest features")
    merged["interest__rate_source_age_days"] = (
        merged["timestamp"] - merged["known_date"]
    ).dt.days
    return merged.drop(columns=["known_at", "known_date"])


def evaluate(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_columns = [column for column in data if column.startswith(BASE_PREFIXES)]
    interest_columns = [column for column in data if column.startswith("interest__")]
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for scope_kind in config["scopes"]:
        scopes = ["pooled"] if scope_kind == "pooled" else list(config["corridors"])
        for scope in scopes:
            source = data if scope == "pooled" else data.loc[data["corridor"].eq(scope)].copy()
            for horizon in config["horizons"]:
                outcome = f"outcome__regret_{int(horizon)}"
                benefit = f"outcome__benefit_{int(horizon)}"
                usable = source.loc[source[outcome].notna()].copy()
                usable["target"] = usable[outcome].le(float(config["tolerance_bps"])).astype(int)
                usable["regret_bps"] = usable[outcome]
                usable["benefit_bps"] = usable[benefit]
                for feature_set in config["feature_sets"]:
                    raw_columns = base_columns if feature_set == "without_interest" else [*base_columns, *interest_columns]
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
    config_path: Path | str = Path("configs/interest_rate_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/interest_rate"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    key_path = Path(config["key_rate_input"])
    data = add_interest_features(
        pd.read_csv(input_path),
        pd.read_csv(key_path),
        lag_days=int(config["strict_lag_calendar_days"]),
    )
    folds, signals = evaluate(data, config)
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
        "key_rate_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory key-rate ablation; key-rate rows are available no earlier than the next calendar day.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/interest_rate_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/interest_rate"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
