"""Leakage-safe hourly CNY/RUB experiment with earlier-closed CETS factors."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import _feature_importance, _future_outcomes, _make_model, _rolling_percentile, _rolling_zscore


KEYS = ("model", "feature_set", "horizon", "tolerance_bps", "top_score_share", "cooldown_hours")


def load_config(path: Path | str = Path("configs/hourly_factor_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("hourly factor config must be preregistered schema_version 1")
    return config


def _series_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    ordered = frame.sort_values("timestamp", kind="mergesort").copy()
    close = ordered.set_index("timestamp")["close"]
    ret1 = close.pct_change(fill_method=None)
    result = pd.DataFrame(index=close.index)
    for window in (1, 3, 6, 12, 24):
        result[f"{prefix}return_{window}"] = close.pct_change(window, fill_method=None)
    for window in (6, 24, 72):
        result[f"{prefix}volatility_{window}"] = ret1.rolling(window, min_periods=window).std()
    for window in (24, 120, 480):
        result[f"{prefix}percentile_{window}"] = _rolling_percentile(close, window)
        result[f"{prefix}zscore_{window}"] = _rolling_zscore(close, window)
    return result.replace([np.inf, -np.inf], np.nan)


def build_hourly_features(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required = {"dt_msk", "secid", "open", "high", "low", "close"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"hourly input lacks {sorted(missing)}")
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["dt_msk"], errors="raise")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["close"].gt(0)].sort_values(["secid", "timestamp"], kind="mergesort")
    target = frame.loc[frame["secid"].eq(config["target_secid"])].copy().reset_index(drop=True)
    if target["timestamp"].duplicated().any():
        raise ValueError("target hourly candles contain duplicate timestamps")
    features = _series_features(target, "base__").reset_index()
    target = target.merge(features, on="timestamp", how="left", validate="one_to_one")
    target["base__candle_range"] = (target["high"] - target["low"]) / target["close"]
    denominator = target["high"] - target["low"]
    target["base__close_location"] = ((target["close"] - target["low"]) / denominator).where(denominator.gt(0), 0.5)
    minute = target["timestamp"].dt.hour * 60 + target["timestamp"].dt.minute
    weekday = target["timestamp"].dt.dayofweek
    target["base__time_sin"] = np.sin(2 * np.pi * minute / (24 * 60))
    target["base__time_cos"] = np.cos(2 * np.pi * minute / (24 * 60))
    target["base__weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    target["base__weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    tolerance = dt.timedelta(hours=float(config["factor_asof_tolerance_hours"]))
    for secid in config["factor_secids"]:
        factor = frame.loc[frame["secid"].eq(secid)].copy()
        short = secid.lower().replace("_tom", "").replace("000utstom", "usd")
        factor_features = _series_features(factor, f"factor__{short}__").reset_index()
        factor_features[f"factor__{short}__known_at"] = factor_features["timestamp"]
        target = pd.merge_asof(
            target.sort_values("timestamp"),
            factor_features.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=tolerance,
        )
        known = target.pop(f"factor__{short}__known_at")
        if (known.dropna() > target.loc[known.notna(), "timestamp"]).any():
            raise AssertionError("merge_asof admitted a future factor candle")
        target[f"factor__{short}__age_minutes"] = (
            target["timestamp"] - known
        ).dt.total_seconds() / 60
        target[f"factor__{short}__missing"] = known.isna().astype(float)

    target = target.rename(columns={"close": "price"})
    for horizon in config["horizons"]:
        regret, benefit = _future_outcomes(target["price"], int(horizon))
        target[f"outcome__regret_{horizon}"] = regret
        target[f"outcome__benefit_{horizon}"] = benefit
    return target.replace([np.inf, -np.inf], np.nan)


def _hourly_policy(candidates: pd.DataFrame, *, cooldown_hours: int, weekly_cap: int) -> pd.DataFrame:
    ordered = candidates.sort_values("timestamp", kind="mergesort").copy()
    if ordered.empty:
        return ordered
    iso = ordered["timestamp"].dt.isocalendar()
    ordered["_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    counts: dict[str, int] = {}
    selected: list[int] = []
    last: pd.Timestamp | None = None
    cooldown = dt.timedelta(hours=int(cooldown_hours))
    for index, row in ordered.iterrows():
        week = str(row["_week"])
        timestamp = pd.Timestamp(row["timestamp"])
        if counts.get(week, 0) >= weekly_cap:
            continue
        if last is None or timestamp - last >= cooldown:
            selected.append(index)
            counts[week] = counts.get(week, 0) + 1
            last = timestamp
    return ordered.loc[selected].drop(columns="_week")


def _purged_train(frame: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
    past = frame.loc[frame["timestamp"] < start].copy()
    return past.iloc[:-horizon] if len(past) > horizon else past.iloc[0:0]


def evaluate(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_columns = [column for column in data if column.startswith("base__")]
    factor_columns = [column for column in data if column.startswith("factor__")]
    policy = config["adaptive_policy"]
    folds: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    for horizon in config["horizons"]:
        outcome = f"outcome__regret_{horizon}"
        benefit = f"outcome__benefit_{horizon}"
        for tolerance_bps in config["tolerance_bps"]:
            usable = data.loc[data[outcome].notna()].copy()
            usable["target"] = usable[outcome].le(float(tolerance_bps)).astype(int)
            usable["regret_bps"] = usable[outcome]
            usable["benefit_bps"] = usable[benefit]
            for feature_set in config["feature_sets"]:
                columns = base_columns if feature_set == "cny_only" else [*base_columns, *factor_columns]
                for model_name in config["models"]:
                    for year in config["test_years"]:
                        start = pd.Timestamp(year=int(year), month=1, day=1)
                        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                        train = _purged_train(usable, start, int(horizon))
                        test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                        if len(train) < int(config["minimum_training_observations"]) or len(test) < 100 or train["target"].nunique() < 2:
                            continue
                        model = _make_model(
                            model_name,
                            iterations=int(config["model_iterations"]),
                            seed=int(config["random_seed"]) + int(year),
                        )
                        model.fit(train[columns], train["target"])
                        test["score"] = model.predict_proba(test[columns])[:, 1]
                        test["week"] = test["timestamp"].dt.to_period("W").astype(str)
                        test["time_bucket"] = (
                            test["timestamp"].dt.dayofweek.astype(str)
                            + "-"
                            + test["timestamp"].dt.hour.astype(str)
                        )
                        matched = test.groupby("time_bucket")["target"].mean()
                        test["matched_time_hit_rate"] = test["time_bucket"].map(matched)
                        for share in policy["top_score_share"]:
                            candidates = adaptive_candidates(
                                test,
                                share=float(share),
                                lookback=int(policy["lookback_observations"]),
                                minimum_history=int(policy["minimum_history_observations"]),
                            )
                            for cooldown in policy["cooldown_hours"]:
                                selected = _hourly_policy(
                                    candidates,
                                    cooldown_hours=int(cooldown),
                                    weekly_cap=int(policy["weekly_cap"]),
                                )
                                count = len(selected)
                                base_rate = float(test["target"].mean())
                                hit_rate = float(selected["target"].mean()) if count else np.nan
                                duration = max(float((test["timestamp"].max() - test["timestamp"].min()).total_seconds()) / (7 * 86400), 1 / 7)
                                identity = {
                                    "model": model_name,
                                    "feature_set": feature_set,
                                    "horizon": int(horizon),
                                    "tolerance_bps": float(tolerance_bps),
                                    "top_score_share": float(share),
                                    "cooldown_hours": int(cooldown),
                                }
                                folds.append(
                                    {
                                        **identity,
                                        "test_year": int(year),
                                        "test_count": len(test),
                                        "test_hits": int(test["target"].sum()),
                                        "signal_count": count,
                                        "signal_hits": int(selected["target"].sum()) if count else 0,
                                        "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                                        "matched_expected_hits": float(selected["matched_time_hit_rate"].sum()) if count else 0.0,
                                        "regret_sum_bps": float(selected["regret_bps"].sum()) if count else 0.0,
                                        "benefit_sum_bps": float(selected["benefit_bps"].sum()) if count else 0.0,
                                        "duration_weeks": duration,
                                    }
                                )
                                if count:
                                    exported = selected[["timestamp", "target", "regret_bps", "benefit_bps", "score", "score_threshold", "matched_time_hit_rate"]].copy()
                                    for key, value in identity.items():
                                        exported[key] = value
                                    exported["test_year"] = int(year)
                                    signal_frames.append(exported)
                        for feature, importance, signed in _feature_importance(model, columns):
                            importance_rows.append(
                                {
                                    "model": model_name,
                                    "feature_set": feature_set,
                                    "horizon": int(horizon),
                                    "tolerance_bps": float(tolerance_bps),
                                    "test_year": int(year),
                                    "feature": feature,
                                    "importance": importance,
                                    "signed_effect": signed,
                                }
                            )
    return pd.DataFrame(folds), pd.concat(signal_frames, ignore_index=True), pd.DataFrame(importance_rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(list(KEYS), sort=True):
        identity = dict(zip(KEYS, keys, strict=True))
        signals = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        hit_rate = hits / signals if signals else np.nan
        base_rate = base_hits / base_count if base_count else np.nan
        matched_rate = float(group["matched_expected_hits"].sum()) / signals if signals else np.nan
        duration = float(group["duration_weeks"].sum())
        valid_lift = group["lift"].dropna()
        rows.append(
            {
                **identity,
                "folds": len(group),
                "signals": signals,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_time_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if signals and base_rate > 0 else np.nan,
                "lift_vs_matched_time": hit_rate / matched_rate if signals and matched_rate > 0 else np.nan,
                "regret_mean_bps": float(group["regret_sum_bps"].sum()) / signals if signals else np.nan,
                "benefit_mean_bps": float(group["benefit_sum_bps"].sum()) / signals if signals else np.nan,
                "signals_per_week": signals / duration if duration else 0.0,
                "worst_fold_lift": float(valid_lift.min()) if len(valid_lift) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    config_path: Path | str = Path("configs/hourly_factor_experiment.json"),
    processed_path: Path | str = Path("data/processed/moex_cny_hourly_factor_features.csv"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/hourly_factors"),
) -> dict[str, object]:
    config = load_config(config_path)
    raw_path = Path(config["input"])
    data = build_hourly_features(pd.read_csv(raw_path), config)
    processed = Path(processed_path)
    processed.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed, index=False)
    folds, signals, importance = evaluate(data, config)
    summary = summarize(folds)
    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    folds.to_csv(artifacts / "folds.csv", index=False)
    signals.to_csv(artifacts / "signals.csv", index=False)
    importance.to_csv(artifacts / "feature_importance.csv", index=False)
    summary.to_csv(artifacts / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "processed_path": str(processed),
        "rows": len(data),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory; factor timestamps are candle END and are merged backward only.",
    }
    (artifacts / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hourly_factor_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/hourly_factors"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
