"""Test whether lagged MOEX trading activity improves the final CatBoost."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import _feature_importance, _make_model, _purged_train
from fxpulse.temporal_sequence_experiment import _evaluate_fold, summarize
from fxpulse.training_history_experiment import add_training_labels


BASE_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")
EXPECTED_FEATURE_SETS = ["baseline", "plus_liquidity"]


def load_config(path: Path | str = Path("configs/liquidity_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("liquidity experiment must be preregistered schema_version 1")
    if config.get("feature_sets") != EXPECTED_FEATURE_SETS:
        raise ValueError("liquidity feature sets differ from the implementation")
    if int(config.get("market_lag_observations", 0)) < 1:
        raise ValueError("MOEX activity must be lagged by at least one observation")
    return config


def _rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window, min_periods=window).rank(pct=True)


def _activity_block(
    values: pd.Series,
    *,
    prefix: str,
    value_name: str,
    lag: int,
) -> pd.DataFrame:
    """Create level, shock and relative-level features, then lag all of them."""

    clean = pd.to_numeric(values, errors="coerce").where(lambda item: item.ge(0))
    logged = np.log1p(clean)
    result = pd.DataFrame(index=values.index)
    root = f"liquidity__{prefix}_{value_name}"
    result[f"{root}_log_lag{lag}"] = logged.shift(lag)
    result[f"{root}_log_change_1_lag{lag}"] = logged.diff(1).shift(lag)
    result[f"{root}_log_change_5_lag{lag}"] = logged.diff(5).shift(lag)
    for window in (20, 60):
        mean = logged.rolling(window, min_periods=window).mean()
        std = logged.rolling(window, min_periods=window).std()
        result[f"{root}_zscore_{window}_lag{lag}"] = ((logged - mean) / std.replace(0, np.nan)).shift(lag)
    result[f"{root}_percentile_60_lag{lag}"] = _rolling_percentile(logged, 60).shift(lag)
    if value_name == "num_trades":
        result[f"liquidity__{prefix}_inactive_lag{lag}"] = clean.eq(0).astype(float).shift(lag)
    return result


def build_liquidity_features(
    raw: pd.DataFrame,
    *,
    instruments: dict[str, str],
    lag: int,
) -> pd.DataFrame:
    """Build causal daily activity features from the public MOEX history."""

    required = {"trade_date", "secid", "num_trades", "volume_rub"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"MOEX daily input lacks {sorted(missing)}")
    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["trade_date"], errors="raise")
    if data.duplicated(["timestamp", "secid"]).any():
        raise ValueError("MOEX daily input contains duplicate date/instrument rows")

    unknown = set(instruments.values()) - set(data["secid"].astype(str))
    if unknown:
        raise ValueError(f"MOEX daily input lacks instruments {sorted(unknown)}")

    pieces: list[pd.DataFrame] = []
    selected_counts: list[pd.Series] = []
    selected_volumes: list[pd.Series] = []
    for short, secid in instruments.items():
        instrument = data.loc[data["secid"].eq(secid)].set_index("timestamp").sort_index()
        count = pd.to_numeric(instrument["num_trades"], errors="coerce")
        selected_counts.append(count.rename(short))
        pieces.append(_activity_block(count, prefix=short, value_name="num_trades", lag=lag))

        volume = pd.to_numeric(instrument["volume_rub"], errors="coerce")
        if volume.notna().any():
            selected_volumes.append(volume.rename(short))
            pieces.append(_activity_block(volume, prefix=short, value_name="volume_rub", lag=lag))

    counts = pd.concat(selected_counts, axis=1).sum(axis=1, min_count=1).sort_index()
    pieces.append(_activity_block(counts, prefix="fx_total", value_name="num_trades", lag=lag))
    if selected_volumes:
        volumes = pd.concat(selected_volumes, axis=1).sum(axis=1, min_count=1).sort_index()
        pieces.append(_activity_block(volumes, prefix="fx_total", value_name="volume_rub", lag=lag))

    result = pd.concat(pieces, axis=1).sort_index()
    if result.columns.duplicated().any():
        raise RuntimeError("duplicate liquidity feature names")
    return result.reset_index().replace([np.inf, -np.inf], np.nan)


def attach_liquidity_features(
    frame: pd.DataFrame,
    liquidity: pd.DataFrame,
    *,
    carry_days: int,
) -> pd.DataFrame:
    """As-of align already lagged MOEX features without carrying stale weeks."""

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    activity = liquidity.copy()
    activity["timestamp"] = pd.to_datetime(activity["timestamp"], errors="raise")
    if activity["timestamp"].duplicated().any():
        raise ValueError("liquidity features contain duplicate dates")
    data["_input_order"] = np.arange(len(data))
    merged = pd.merge_asof(
        data.sort_values("timestamp"),
        activity.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(int(carry_days), unit="D"),
    )
    return merged.sort_values("_input_order").drop(columns="_input_order").reset_index(drop=True)


def _coverage(raw: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    selected = raw.loc[raw["secid"].isin(config["instruments"].values())].copy()
    volume = pd.to_numeric(selected["volume_rub"], errors="coerce")
    trades = pd.to_numeric(selected["num_trades"], errors="coerce")
    return {
        "raw_rows": len(selected),
        "raw_date_from": str(selected["trade_date"].min()),
        "raw_date_to": str(selected["trade_date"].max()),
        "num_trades_filled_rows": int(trades.notna().sum()),
        "num_trades_positive_rows": int(trades.gt(0).sum()),
        "volume_rub_filled_rows": int(volume.notna().sum()),
        "volume_note": (
            "Public MOEX daily history returned no volume_rub values; the tested ablation therefore "
            "uses num_trades as a trading-activity proxy. Volume features are created automatically "
            "when a populated point-in-time source is supplied."
        ),
    }


def evaluate(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = add_training_labels(data, config)
    frame = frame.loc[frame["timestamp"].ge(pd.Timestamp(config["training_start"]))].copy()
    baseline_columns = [column for column in frame if column.startswith(BASE_PREFIXES)]
    liquidity_columns = [column for column in frame if column.startswith("liquidity__")]
    if not liquidity_columns:
        raise ValueError("no usable liquidity features were built")

    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    for corridor in config["corridors"]:
        corridor_data = frame.loc[frame["corridor"].eq(corridor)].copy()
        for horizon in config["horizons"]:
            horizon = int(horizon)
            outcome = f"outcome__regret_{horizon}"
            benefit = f"outcome__benefit_{horizon}"
            label = f"training_target_{horizon}"
            usable = corridor_data.loc[
                corridor_data[outcome].notna() & corridor_data[label].notna()
            ].copy()
            usable["target"] = usable[outcome].le(float(config["evaluation_tolerance_bps"])).astype(int)
            usable["training_target"] = usable[label].astype(int)
            usable["regret_bps"] = usable[outcome]
            usable["benefit_bps"] = usable[benefit]

            for feature_set in config["feature_sets"]:
                columns = list(baseline_columns)
                if feature_set == "plus_liquidity":
                    columns.extend(liquidity_columns)
                for year in config["test_years"]:
                    start = pd.Timestamp(year=int(year), month=1, day=1)
                    end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                    train = _purged_train(usable, start, horizon)
                    train = train.loc[
                        train["timestamp"].ge(
                            start - pd.DateOffset(years=int(config["rolling_training_years"]))
                        )
                    ].copy()
                    test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                    if (
                        len(train) < int(config["minimum_training_observations"])
                        or len(test) < 20
                        or train["training_target"].nunique() < 2
                    ):
                        continue

                    predictions: list[np.ndarray] = []
                    fitted_models: list[Any] = []
                    for seed in config["ensemble_seeds"]:
                        model = _make_model(
                            str(config["model"]),
                            iterations=int(config["model_iterations"]),
                            seed=int(seed) + int(year) + horizon,
                        )
                        model.fit(train[columns], train["training_target"])
                        predictions.append(model.predict_proba(test[columns])[:, 1])
                        fitted_models.append(model)
                    test["score"] = np.mean(np.column_stack(predictions), axis=1)
                    test["week"] = test["timestamp"].dt.to_period("W").astype(str)
                    weekly_rate = test.groupby("week", sort=False)["target"].mean()
                    test["matched_week_hit_rate"] = test["week"].map(weekly_rate).astype(float)
                    selected = _evaluate_fold(test, policy=config["adaptive_policy"])
                    count = len(selected)
                    base_rate = float(test["target"].mean())
                    hit_rate = float(selected["target"].mean()) if count else np.nan
                    duration = max(float((test["timestamp"].max() - test["timestamp"].min()).days) / 7, 1 / 7)
                    identity = {
                        "scope": corridor,
                        "model": "catboost_3seed_mean",
                        "feature_set": feature_set,
                        "horizon": horizon,
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
                        exported = selected[
                            [
                                "timestamp",
                                "corridor",
                                "target",
                                "training_target",
                                "regret_bps",
                                "benefit_bps",
                                "score",
                                "score_threshold",
                                "matched_week_hit_rate",
                            ]
                        ].copy()
                        for key, value in identity.items():
                            exported[key] = value
                        exported["test_year"] = int(year)
                        signal_frames.append(exported)
                    for model in fitted_models:
                        for feature, importance, _ in _feature_importance(model, columns):
                            importance_rows.append(
                                {
                                    **identity,
                                    "test_year": int(year),
                                    "feature": feature,
                                    "importance": importance,
                                }
                            )
            print(f"completed liquidity ablation {corridor} h={horizon}", flush=True)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return pd.DataFrame(fold_rows), signals, pd.DataFrame(importance_rows)


def paired_comparison(summary_frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["scope", "model", "horizon"]
    metrics = [
        "lift",
        "lift_vs_matched_random",
        "regret_mean_bps",
        "benefit_mean_bps",
        "signals_per_week",
        "worst_fold_lift",
    ]
    baseline = summary_frame.loc[summary_frame["feature_set"].eq("baseline"), [*keys, *metrics]]
    added = summary_frame.loc[
        summary_frame["feature_set"].eq("plus_liquidity"), [*keys, *metrics]
    ]
    result = added.merge(baseline, on=keys, suffixes=("_liquidity", "_baseline"), validate="one_to_one")
    for metric in metrics:
        result[f"delta_{metric}"] = result[f"{metric}_liquidity"] - result[f"{metric}_baseline"]
    return result


def run(
    *,
    config_path: Path | str = Path("configs/liquidity_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/liquidity"),
) -> dict[str, object]:
    config = load_config(config_path)
    raw_path = Path(config["moex_daily_input"])
    input_path = Path(config["input"])
    raw = pd.read_csv(raw_path)
    liquidity = build_liquidity_features(
        raw,
        instruments=dict(config["instruments"]),
        lag=int(config["market_lag_observations"]),
    )
    liquidity_path = Path(config["liquidity_output"])
    liquidity_path.parent.mkdir(parents=True, exist_ok=True)
    liquidity.to_csv(liquidity_path, index=False)

    augmented = attach_liquidity_features(
        pd.read_csv(input_path),
        liquidity,
        carry_days=int(config["maximum_alignment_carry_days"]),
    )
    feature_path = Path(config["feature_output"])
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(feature_path, index=False)

    folds, signals, importance = evaluate(augmented, config)
    summary_frame = summarize(folds)
    comparison = paired_comparison(summary_frame)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "moex_daily_input_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "liquidity_output": str(liquidity_path),
        "liquidity_feature_count": len([c for c in augmented if c.startswith("liquidity__")]),
        "feature_output": str(feature_path),
        "feature_rows": len(augmented),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "coverage": _coverage(raw, config),
        "warning": "Exploratory paired ablation on already reviewed OOT years; do not promote without a new hold-out.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/liquidity_experiment.json"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/next_hypotheses/liquidity"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
