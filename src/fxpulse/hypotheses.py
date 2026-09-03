"""Registered, point-in-time FX Pulse hypotheses.

The runner deliberately reports an unavailable hypothesis instead of inventing
its result. In particular, public MOEX data cannot establish whether a signal
survives into the bank's executable customer quote.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.labeling import HORIZONS, evaluate_positions, label_observations
from fxpulse.panel import MSK


HYPOTHESIS_COLUMNS = (
    "hypothesis",
    "status",
    "source_series",
    "direction",
    "horizon",
    "evaluation_period",
    "evaluation_start",
    "evaluation_end",
    "signal_count",
    "eligible_base_count",
    "hit_rate",
    "baseline_hit_rate",
    "lift",
    "benefit_sym_bps",
    "benefit_fwd_bps",
    "benefit_fwd_newey_west_t",
    "signals_per_week",
    "signals_per_month",
    "cluster_share",
    "interval_cv",
    "details",
)
SIGNAL_COLUMNS = (
    "timestamp",
    "value_date",
    "hypothesis",
    "source_series",
    "direction",
    "strength",
    "details",
)
DAILY_DIRECTIONS = {
    "H1_confirmed_reversal": "window_closing",
    "H2_low_with_deceleration": "favorable",
    "H3_reversal_low_or_mid_vol": "window_closing",
    # Second preregistered research batch. These are intentionally broad
    # controls: a negative result is as useful as a candidate for a push.
    "H6_one_day_downside_shock": "favorable",
    "H7_three_day_drawdown_then_bounce": "window_closing",
    "H8_quiet_lower_decile": "favorable",
    "H9_upside_breakout": "window_closing",
    "H10_bollinger_low": "favorable",
}


def hypotheses_sha256(path: Path | str = Path("configs/hypotheses.json")) -> str:
    """Return the hash of the registered model-hypothesis batch."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model_hypotheses(path: Path | str = Path("configs/hypotheses.json")) -> dict[str, Any]:
    """Read and validate the small, immutable-before-run model registry."""

    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Only hypothesis schema_version 1 is supported")
    evaluation = config.get("evaluation")
    models = config.get("models")
    daily_rules = config.get("daily_rules", [])
    if not isinstance(evaluation, dict) or not isinstance(models, list) or not models:
        raise ValueError("hypothesis config requires non-empty evaluation and models")
    if not isinstance(daily_rules, list):
        raise ValueError("daily_rules must be a list")
    required_evaluation = {"target_horizon_observations", "weekly_cap", "minimum_training_observations"}
    missing_evaluation = required_evaluation - set(evaluation)
    if missing_evaluation:
        raise ValueError(f"hypothesis evaluation lacks: {', '.join(sorted(missing_evaluation))}")
    ids: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("each model hypothesis must be an object")
        required_model = {"id", "direction", "target", "algorithm", "l2", "max_iterations", "features"}
        missing_model = required_model - set(model)
        if missing_model:
            raise ValueError(f"model hypothesis lacks: {', '.join(sorted(missing_model))}")
        if model["id"] in ids:
            raise ValueError(f"duplicate model hypothesis id {model['id']!r}")
        ids.add(model["id"])
        if model["direction"] not in {"favorable", "window_closing"}:
            raise ValueError("model direction must be favorable or window_closing")
        if model["target"] not in {"hit_favorable", "hit_closing"}:
            raise ValueError("model target must be hit_favorable or hit_closing")
        if model["algorithm"] != "ridge_logistic_regression":
            raise ValueError("only ridge_logistic_regression is supported")
        if not isinstance(model["features"], list) or not model["features"]:
            raise ValueError("model features must be a non-empty list")
    supported_rules = {
        "high_activity_downside_reversal",
        "high_activity_low_rebound",
        "three_day_selloff_recovery",
    }
    for rule in daily_rules:
        if not isinstance(rule, dict):
            raise ValueError("each daily rule must be an object")
        required_rule = {"id", "kind", "direction", "params"}
        missing_rule = required_rule - set(rule)
        if missing_rule:
            raise ValueError(f"daily rule lacks: {', '.join(sorted(missing_rule))}")
        if rule["id"] in ids:
            raise ValueError(f"duplicate hypothesis id {rule['id']!r}")
        ids.add(rule["id"])
        if rule["kind"] not in supported_rules:
            raise ValueError(f"unsupported daily rule kind {rule['kind']!r}")
        if rule["direction"] not in {"favorable", "window_closing"}:
            raise ValueError("daily rule direction must be favorable or window_closing")
        if not isinstance(rule["params"], dict):
            raise ValueError("daily rule params must be an object")
    return config


def _make_panel(index: pd.Series, prices: pd.Series, series_id: str) -> pd.DataFrame:
    timestamps = pd.to_datetime(index, errors="raise")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(MSK)
    else:
        timestamps = timestamps.dt.tz_convert(MSK)
    return pd.DataFrame(
        {
            "known_at": timestamps,
            "value_date": timestamps.dt.date,
            "series_id": series_id,
            "price": pd.to_numeric(prices, errors="raise").to_numpy(),
            "is_carried": False,
            "meta": [{} for _ in range(len(timestamps))],
        }
    ).reset_index(drop=True)


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=("position", *SIGNAL_COLUMNS))


def load_daily_cny(path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"trade_date", "secid", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"daily data lack: {', '.join(sorted(missing))}")
    raw = raw.loc[raw["secid"].eq("CNYRUB_TOM")].copy()
    raw["timestamp"] = pd.to_datetime(raw["trade_date"], errors="raise").dt.normalize() + pd.DateOffset(hours=23, minutes=59)
    raw["price"] = pd.to_numeric(raw["close"], errors="coerce")
    columns = ["timestamp", "price", *[column for column in ("open", "high", "low", "num_trades") if column in raw]]
    return raw.loc[raw["price"].gt(0), columns].sort_values("timestamp").reset_index(drop=True)


def load_intraday_cny(path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"dt_msk", "secid", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"candle data lack: {', '.join(sorted(missing))}")
    raw = raw.loc[raw["secid"].eq("CNYRUB_TOM")].copy()
    raw["timestamp"] = pd.to_datetime(raw["dt_msk"], errors="raise")
    raw["price"] = pd.to_numeric(raw["close"], errors="coerce")
    return raw.loc[raw["price"].gt(0), ["timestamp", "price"]].sort_values("timestamp").reset_index(drop=True)


def daily_candidate_signals(daily: pd.DataFrame) -> pd.DataFrame:
    """Generate H1–H3 strictly from prices observable at the current close."""

    price = daily["price"].reset_index(drop=True)
    returns = price.pct_change()
    prior_low_90 = price.shift(1).rolling(90, min_periods=90).min()
    sigma_20 = returns.shift(1).rolling(20, min_periods=20).std()
    rebound = price / prior_low_90 - 1
    positive_confirmation = returns.ge(0) & returns.shift(1).ge(0)

    h1 = prior_low_90.lt(price) & rebound.ge(0.5 * sigma_20) & positive_confirmation

    lower_decile = price.le(price.rolling(90, min_periods=90).quantile(0.10))
    decelerating = returns.shift(2).lt(returns.shift(1)) & returns.shift(1).lt(returns) & returns.le(0)
    h2 = lower_decile & decelerating

    realized_volatility = returns.rolling(20, min_periods=20).std()
    past_volatility_cutoff = realized_volatility.shift(1).rolling(252, min_periods=126).quantile(2 / 3)
    h3 = h1 & realized_volatility.le(past_volatility_cutoff)

    h6 = returns.le(-2 * sigma_20)

    drawdown_3 = price / price.shift(3) - 1
    h7 = drawdown_3.le(-2 * sigma_20 * (3**0.5)) & returns.ge(0)

    lower_volatility_cutoff = realized_volatility.shift(1).rolling(252, min_periods=126).quantile(1 / 3)
    h8 = lower_decile & realized_volatility.le(lower_volatility_cutoff)

    prior_high_20 = price.shift(1).rolling(20, min_periods=20).max()
    h9 = price.gt(prior_high_20) & returns.ge(0.5 * sigma_20)

    mean_20 = price.shift(1).rolling(20, min_periods=20).mean()
    std_20 = price.shift(1).rolling(20, min_periods=20).std()
    h10 = price.le(mean_20 - 2 * std_20)

    definitions = (
        ("H1_confirmed_reversal", h1, "window_closing", rebound / sigma_20),
        ("H2_low_with_deceleration", h2, "favorable", (price.rolling(90).quantile(0.10) / price - 1) / sigma_20),
        ("H3_reversal_low_or_mid_vol", h3, "window_closing", rebound / sigma_20),
        ("H6_one_day_downside_shock", h6, "favorable", -returns / sigma_20),
        ("H7_three_day_drawdown_then_bounce", h7, "window_closing", -drawdown_3 / sigma_20),
        ("H8_quiet_lower_decile", h8, "favorable", (price.rolling(90).quantile(0.10) / price - 1) / sigma_20),
        ("H9_upside_breakout", h9, "window_closing", (price / prior_high_20 - 1) / sigma_20),
        ("H10_bollinger_low", h10, "favorable", (mean_20 - price) / std_20),
    )
    rows: list[dict[str, object]] = []
    for name, candidate, direction, strength in definitions:
        for position in daily.index[candidate.fillna(False)]:
            rows.append(
                {
                    "position": int(position),
                    "timestamp": daily.loc[position, "timestamp"],
                    "value_date": pd.Timestamp(daily.loc[position, "timestamp"]).date().isoformat(),
                    "hypothesis": name,
                    "source_series": "MOEX:CNYRUB_TOM:daily",
                    "direction": direction,
                    "strength": float(min(max(strength.loc[position], 0.0), 1.0)),
                    "details": json.dumps({"rule": name}, sort_keys=True),
                }
            )
    return pd.DataFrame(rows) if rows else _empty_signals()


def daily_ohlc_candidate_signals(daily: pd.DataFrame, rules: list[dict[str, Any]]) -> pd.DataFrame:
    """Evaluate registered close/range/activity hypotheses at the daily close.

    ``num_trades`` is used as an open, reproducible activity proxy: the public
    daily ISS history currently has no populated RUB-volume field. Its rolling
    percentile is shifted, so the comparison distribution contains no current
    or future activity.
    """

    if not rules:
        return _empty_signals()
    required = {"timestamp", "price", "open", "high", "low", "num_trades"}
    missing = required - set(daily)
    if missing:
        raise ValueError(f"OHLC hypotheses require: {', '.join(sorted(missing))}")
    price = pd.to_numeric(daily["price"], errors="raise").reset_index(drop=True)
    opening = pd.to_numeric(daily["open"], errors="coerce").reset_index(drop=True)
    high = pd.to_numeric(daily["high"], errors="coerce").reset_index(drop=True)
    low = pd.to_numeric(daily["low"], errors="coerce").reset_index(drop=True)
    activity = pd.to_numeric(daily["num_trades"], errors="coerce").reset_index(drop=True)
    returns = price.pct_change()
    sigma_20 = returns.shift(1).rolling(20, min_periods=20).std()
    activity_cutoffs = {
        float(rule["params"]["activity_percentile"]): activity.shift(1)
        .rolling(252, min_periods=126)
        .quantile(float(rule["params"]["activity_percentile"]))
        for rule in rules
    }
    daily_range = high - low
    close_location = ((price - low) / daily_range).where(daily_range.gt(0))
    rows: list[dict[str, object]] = []
    for rule in rules:
        params = rule["params"]
        activity_cutoff = activity_cutoffs[float(params["activity_percentile"])]
        is_high_activity = activity.ge(activity_cutoff)
        if rule["kind"] == "high_activity_downside_reversal":
            candidate = (
                returns.le(float(params["return_sigma_max"]) * sigma_20)
                & is_high_activity
                & close_location.ge(float(params["close_location_min"]))
            )
            strength = pd.concat(
                [
                    -returns / (abs(float(params["return_sigma_max"])) * sigma_20),
                    activity / activity_cutoff,
                    close_location / float(params["close_location_min"]),
                ],
                axis=1,
            ).min(axis=1)
        elif rule["kind"] == "high_activity_low_rebound":
            lookback = int(params["lookback"])
            prior_low = price.shift(1).rolling(lookback, min_periods=lookback).min()
            rebound = price / prior_low - 1
            candidate = (
                prior_low.lt(price)
                & rebound.ge(float(params["rebound_sigma_min"]) * sigma_20)
                & returns.ge(0)
                & is_high_activity
            )
            strength = pd.concat(
                [
                    rebound / (float(params["rebound_sigma_min"]) * sigma_20),
                    activity / activity_cutoff,
                ],
                axis=1,
            ).min(axis=1)
        elif rule["kind"] == "three_day_selloff_recovery":
            drawdown_3 = price / price.shift(3) - 1
            candidate = (
                drawdown_3.le(float(params["drawdown_sigma_max"]) * sigma_20 * np.sqrt(3))
                & opening.lt(price)
                & is_high_activity
                & close_location.ge(float(params["close_location_min"]))
            )
            strength = pd.concat(
                [
                    -drawdown_3 / (abs(float(params["drawdown_sigma_max"])) * sigma_20 * np.sqrt(3)),
                    activity / activity_cutoff,
                    close_location / float(params["close_location_min"]),
                ],
                axis=1,
            ).min(axis=1)
        else:  # Defensive: loader already prevents an unrecognized kind.
            raise ValueError(f"unsupported daily rule kind {rule['kind']!r}")
        for position in daily.index[candidate.fillna(False)]:
            rows.append(
                {
                    "position": int(position),
                    "timestamp": daily.loc[position, "timestamp"],
                    "value_date": pd.Timestamp(daily.loc[position, "timestamp"]).date().isoformat(),
                    "hypothesis": rule["id"],
                    "source_series": "MOEX:CNYRUB_TOM:daily",
                    "direction": rule["direction"],
                    "strength": float(min(max(strength.loc[position], 0.0), 1.0)),
                    "details": json.dumps({"kind": rule["kind"], "params": params}, sort_keys=True),
                }
            )
    return pd.DataFrame(rows) if rows else _empty_signals()


def daily_model_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build model features available at the daily close, with no labels.

    Every rolling statistic either ends at the current close or is explicitly
    shifted where it defines a historical comparison distribution. The target
    remains in ``label_observations`` and is joined only inside the backtest.
    """

    price = pd.to_numeric(daily["price"], errors="raise").reset_index(drop=True)
    returns = price.pct_change()
    sigma_20 = returns.shift(1).rolling(20, min_periods=20).std()
    prior_low_60 = price.shift(1).rolling(60, min_periods=60).min()
    prior_mean_20 = price.shift(1).rolling(20, min_periods=20).mean()
    historical_volatility = sigma_20.shift(1).rolling(252, min_periods=126).median()
    timestamp = pd.to_datetime(daily["timestamp"], errors="raise")
    weekday_angle = 2 * np.pi * timestamp.dt.dayofweek / 5
    features = pd.DataFrame(
        {
            "position": daily.index.to_numpy(),
            "timestamp": timestamp,
            "return_1_sigma": returns / sigma_20,
            "return_3_sigma": (price / price.shift(3) - 1) / (sigma_20 * np.sqrt(3)),
            "return_5_sigma": (price / price.shift(5) - 1) / (sigma_20 * np.sqrt(5)),
            "distance_ma20_sigma": (price / prior_mean_20 - 1) / sigma_20,
            "distance_low60_sigma": (price / prior_low_60 - 1) / sigma_20,
            "level_percentile_60": price.rolling(60, min_periods=60).rank(pct=True),
            "volatility_ratio_20_252": sigma_20 / historical_volatility,
            "weekday_sin": np.sin(weekday_angle),
            "weekday_cos": np.cos(weekday_angle),
        }
    )
    # A flat series has a mathematically valid zero realized volatility, but a
    # z-score relative to it is undefined. Preserve it as a missing feature so
    # the model's complete-case gate rejects that observation instead of
    # passing +/-inf into fitting or scoring.
    return features.replace([np.inf, -np.inf], np.nan)


def _fit_ridge_logistic(
    features: np.ndarray,
    target: np.ndarray,
    *,
    l2: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a standardized ridge logit using deterministic Newton steps.

    A compact in-repository implementation keeps the signal layer dependent
    only on the project runtime. The scaler and weights are fitted afresh in
    each historical training window, never globally.
    """

    if len(features) != len(target) or len(features) == 0:
        raise ValueError("features and target must have the same non-zero length")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("ridge logistic requires finite feature and target values")
    if len(np.unique(target)) < 2:
        raise ValueError("logistic target must contain both classes")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    weights = np.zeros(design.shape[1])
    penalty = np.diag([0.0, *([l2] * (design.shape[1] - 1))])
    for _ in range(max_iterations):
        logits = np.clip(design @ weights, -35, 35)
        probability = 1 / (1 + np.exp(-logits))
        gradient = (design.T @ (probability - target)) / len(target) + penalty @ weights
        curvature = probability * (1 - probability)
        hessian = (design.T * curvature) @ design / len(target) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("ridge logistic Hessian is singular") from exc
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return weights, mean, scale


def _logistic_probability(features: np.ndarray, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if not np.isfinite(features).all():
        raise ValueError("ridge logistic scoring requires finite feature values")
    standardized = (features - mean) / scale
    logits = np.clip(np.column_stack([np.ones(len(standardized)), standardized]) @ weights, -35, 35)
    return 1 / (1 + np.exp(-logits))


def monthly_logistic_signals(
    daily: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    test_from: pd.Timestamp,
    model_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> pd.DataFrame:
    """Generate H11/H12 from monthly expanding, purged logistic models.

    The current month is out of sample. Its first ``horizon`` prior positions
    are also excluded from training because their labels could touch the test
    month. A weekly budget selects the first already-observed high-score day,
    never the best day after the week has ended.
    """

    horizon = int(evaluation_config["target_horizon_observations"])
    weekly_cap = int(evaluation_config["weekly_cap"])
    minimum_training = int(evaluation_config["minimum_training_observations"])
    if horizon not in HORIZONS:
        raise ValueError(f"target_horizon_observations must be one of {HORIZONS}")
    if weekly_cap <= 0 or minimum_training <= 0:
        raise ValueError("weekly_cap and minimum_training_observations must be positive")

    feature_frame = daily_model_features(daily)
    labels = label_observations(panel, horizon).set_index("position")
    feature_frame["month"] = feature_frame["timestamp"].dt.to_period("M")
    feature_frame["week"] = feature_frame["timestamp"].dt.to_period("W")
    feature_names = list(model_config["features"])
    missing_features = set(feature_names) - set(feature_frame)
    if missing_features:
        raise ValueError(f"model uses unknown features: {', '.join(sorted(missing_features))}")
    rows: list[dict[str, object]] = []
    months = sorted(month for month in feature_frame["month"].unique() if month.start_time >= test_from)
    for month in months:
        test = feature_frame.loc[feature_frame["month"].eq(month)].copy()
        test = test.loc[test["timestamp"] >= test_from]
        if test.empty:
            continue
        first_test_position = int(test["position"].min())
        # Purge exactly the target horizon from the tail before current OOT month.
        train = feature_frame.loc[feature_frame["position"].lt(first_test_position - horizon)].copy()
        train["target"] = train["position"].map(labels[model_config["target"]])
        train = train.dropna(subset=["target", *feature_names])
        if len(train) < minimum_training or train["target"].nunique() < 2:
            continue
        train_matrix = train.loc[:, feature_names].to_numpy(dtype="float64")
        target = train["target"].astype(int).to_numpy()
        weights, mean, scale = _fit_ridge_logistic(
            train_matrix,
            target,
            l2=float(model_config["l2"]),
            max_iterations=int(model_config["max_iterations"]),
        )
        # The 90th percentile is an operational threshold learned strictly
        # from the current training window; it is not optimized on the test.
        train_scores = _logistic_probability(train_matrix, weights, mean, scale)
        threshold = float(np.quantile(train_scores, 0.90))
        candidates = test.dropna(subset=feature_names).copy()
        if candidates.empty:
            continue
        candidates["score"] = _logistic_probability(
            candidates.loc[:, feature_names].to_numpy(dtype="float64"), weights, mean, scale
        )
        candidates = candidates.loc[candidates["score"].ge(threshold)]
        candidates = candidates.groupby("week", sort=False).head(weekly_cap)
        for row in candidates.itertuples(index=False):
            rows.append(
                {
                    "position": int(row.position),
                    "timestamp": row.timestamp,
                    "value_date": pd.Timestamp(row.timestamp).date().isoformat(),
                    "hypothesis": model_config["id"],
                    "source_series": "MOEX:CNYRUB_TOM:daily",
                    "direction": model_config["direction"],
                    "strength": float(row.score),
                    "details": json.dumps(
                        {
                            "algorithm": model_config["algorithm"],
                            "horizon": horizon,
                            "retrained_for": str(month),
                            "score_threshold": threshold,
                            "train_observations": len(train),
                        },
                        sort_keys=True,
                    ),
                }
            )
    return pd.DataFrame(rows) if rows else _empty_signals()


def intraday_session_signals(
    candles: pd.DataFrame,
    *,
    test_from: pd.Timestamp,
    train_months: int = 24,
    horizon: int = 5,
) -> pd.DataFrame:
    """H4: monthly-retrained time-of-day filter plus an observable local dip."""

    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    work = candles.copy().reset_index(drop=True)
    work["day"] = work["timestamp"].dt.date
    work["bucket"] = work["timestamp"].dt.strftime("%H:%M")
    future_price = work["price"].shift(-horizon)
    same_day = work["day"].eq(work["day"].shift(-horizon))
    work["forward_return"] = (future_price / work["price"] - 1).where(same_day)
    work["is_local_dip"] = work["price"].eq(work["price"].rolling(7, min_periods=7).min())
    work["month"] = work["timestamp"].dt.to_period("M")

    rows: list[dict[str, object]] = []
    for month in sorted(period for period in work["month"].unique() if period.start_time >= test_from):
        month_start = month.start_time
        training_start = month_start - pd.DateOffset(months=train_months)
        train = work.loc[(work["timestamp"] >= training_start) & (work["timestamp"] < month_start)].dropna(
            subset=["forward_return"]
        )
        if train.empty:
            continue
        bucket_scores = train.groupby("bucket")["forward_return"].agg(["mean", "count"])
        bucket_scores = bucket_scores.loc[bucket_scores["count"].ge(40)]
        if len(bucket_scores) < 10:
            continue
        threshold = bucket_scores["mean"].quantile(0.90)
        chosen_buckets = set(bucket_scores.index[bucket_scores["mean"].ge(threshold) & bucket_scores["mean"].gt(0)])
        candidates = work.loc[
            work["month"].eq(month) & work["bucket"].isin(chosen_buckets) & work["is_local_dip"],
            ["timestamp", "day", "price", "bucket"],
        ].copy()
        # The first qualifying observation is the only one knowable before a
        # later candidate of that day; this also makes the hypothesis compatible
        # with a strict communication budget.
        candidates = candidates.drop_duplicates("day", keep="first")
        candidates["week"] = candidates["timestamp"].dt.to_period("W")
        # Take the first two events that become visible in a week; ranking them
        # after the week closes would itself be a look-ahead leak.
        candidates = candidates.groupby("week", sort=False).head(2)
        for row in candidates.itertuples(index=False):
            rows.append(
                {
                    "timestamp": row.timestamp,
                    "value_date": pd.Timestamp(row.timestamp).date().isoformat(),
                    "hypothesis": "H4_intraday_session_dip",
                    "source_series": "MOEX:CNYRUB_TOM:10m",
                    "direction": "window_closing",
                    "strength": float(min(bucket_scores.loc[row.bucket, "mean"] / threshold, 1.0)),
                    "details": json.dumps({"bucket": row.bucket, "retrained_for": str(month)}, sort_keys=True),
                }
            )
    return pd.DataFrame(rows) if rows else _empty_signals()


def quote_fidelity_status(path: Path | str) -> dict[str, object]:
    """H5 needs executable bank quotes; public market data are not a substitute."""

    quote_path = Path(path)
    if not quote_path.exists():
        return {
            "hypothesis": "H5_quote_fidelity",
            "status": "not_testable",
            "source_series": "APP_QUOTE vs MOEX:CNYRUB_TOM",
            "direction": None,
            "details": "Нет `app_quotes.csv` с observed_at, app_rate_rub и reference_rate_rub; публичный MOEX не доказывает исполнимую цену клиента.",
        }
    quotes = pd.read_csv(quote_path)
    required = {"observed_at", "app_rate_rub", "reference_rate_rub"}
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"app quote data lack: {', '.join(sorted(missing))}")
    app = pd.to_numeric(quotes["app_rate_rub"], errors="coerce")
    reference = pd.to_numeric(quotes["reference_rate_rub"], errors="coerce")
    deviation_bps = ((app / reference - 1) * 10_000).abs().dropna()
    if deviation_bps.empty:
        return {
            "hypothesis": "H5_quote_fidelity",
            "status": "not_testable",
            "source_series": "APP_QUOTE vs MOEX:CNYRUB_TOM",
            "direction": None,
            "details": "В app quote-файле нет валидных пар котировок.",
        }
    return {
        "hypothesis": "H5_quote_fidelity",
        "status": "evaluated",
        "source_series": "APP_QUOTE vs MOEX:CNYRUB_TOM",
        "direction": None,
        "details": json.dumps(
            {
                "observations": len(deviation_bps),
                "median_abs_deviation_bps": float(deviation_bps.median()),
                "p95_abs_deviation_bps": float(deviation_bps.quantile(0.95)),
            },
            sort_keys=True,
        ),
    }


def _evaluate(
    *,
    hypothesis: str,
    source_series: str,
    direction: str,
    panel: pd.DataFrame,
    labels_by_horizon: dict[int, pd.DataFrame],
    candidate_timestamps: pd.Series,
    test_from: pd.Timestamp,
) -> list[dict[str, object]]:
    timestamp_to_position = {timestamp: position for position, timestamp in enumerate(panel["known_at"])}
    known_timestamps = []
    for timestamp in candidate_timestamps:
        point = pd.Timestamp(timestamp)
        known_timestamps.append(point.tz_localize(MSK) if point.tzinfo is None else point.tz_convert(MSK))
    positions = [timestamp_to_position[timestamp] for timestamp in known_timestamps if timestamp in timestamp_to_position]
    timestamps = panel["known_at"]
    latest = pd.Timestamp(panel["value_date"].max()).normalize()
    windows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = [("five_year", test_from, latest)]
    for year in range(test_from.year, latest.year + 1):
        start = max(test_from, pd.Timestamp(year=year, month=1, day=1))
        end = min(latest, pd.Timestamp(year=year, month=12, day=31))
        windows.append((str(year), start, end))
    rows: list[dict[str, object]] = []
    for period, period_start, period_end in windows:
        lower = period_start.tz_localize(MSK)
        upper = (period_end + pd.DateOffset(days=1)).tz_localize(MSK)
        base_positions = [
            position for position, timestamp in enumerate(timestamps) if lower <= timestamp < upper
        ]
        for horizon in HORIZONS:
            metrics = evaluate_positions(
                panel,
                positions,
                direction=direction,
                horizon=horizon,
                base_positions=base_positions,
                labels=labels_by_horizon[horizon],
            )
            rows.append(
                {
                    "hypothesis": hypothesis,
                    "status": "evaluated",
                    "source_series": source_series,
                    "direction": direction,
                    "horizon": horizon,
                    "evaluation_period": period,
                    "evaluation_start": period_start.date().isoformat(),
                    "evaluation_end": period_end.date().isoformat(),
                    **metrics,
                    "details": "point-in-time rule; all labels are future-only and excluded from features",
                }
            )
    return rows


def run_hypotheses(
    *,
    daily_path: Path | str = Path("data/raw/moex_daily.csv"),
    candles_path: Path | str = Path("data/raw/moex_cny_candles_8y.csv"),
    app_quotes_path: Path | str = Path("data/raw/app_quotes.csv"),
    hypothesis_config_path: Path | str = Path("configs/hypotheses.json"),
    artifact_dir: Path | str = Path("artifacts/hypotheses"),
    test_from: str = "2021-09-03",
) -> dict[str, Any]:
    """Evaluate registered hypotheses and write signals, metrics and metadata."""

    start = pd.Timestamp(test_from).normalize()
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    model_config_path = Path(hypothesis_config_path)
    model_registry = load_model_hypotheses(model_config_path)
    daily = load_daily_cny(daily_path)
    daily_panel = _make_panel(daily["timestamp"], daily["price"], "MOEX:CNYRUB_TOM:daily")
    daily_labels = {horizon: label_observations(daily_panel, horizon) for horizon in HORIZONS}
    base_daily_signals = daily_candidate_signals(daily)
    ohlc_daily_signals = daily_ohlc_candidate_signals(daily, model_registry["daily_rules"])
    daily_parts = [frame for frame in (base_daily_signals, ohlc_daily_signals) if not frame.empty]
    daily_signals = pd.concat(daily_parts, ignore_index=True) if daily_parts else _empty_signals()
    model_signal_frames = [
        monthly_logistic_signals(
            daily,
            daily_panel,
            test_from=start,
            model_config=model,
            evaluation_config=model_registry["evaluation"],
        )
        for model in model_registry["models"]
    ]
    non_empty_model_frames = [frame for frame in model_signal_frames if not frame.empty]
    model_signals = pd.concat(non_empty_model_frames, ignore_index=True) if non_empty_model_frames else _empty_signals()
    daily_out_of_time = daily_signals.loc[daily_signals["timestamp"] >= start].copy()
    daily_output_parts = [frame for frame in (daily_out_of_time, model_signals) if not frame.empty]
    signal_frames = [pd.concat(daily_output_parts, ignore_index=True)] if daily_output_parts else []
    metric_rows: list[dict[str, object]] = []

    for hypothesis, direction in DAILY_DIRECTIONS.items():
        candidates = daily_signals.loc[daily_signals["hypothesis"].eq(hypothesis), "timestamp"]
        metric_rows.extend(
            _evaluate(
                hypothesis=hypothesis,
                source_series="MOEX:CNYRUB_TOM:daily",
                direction=direction,
                panel=daily_panel,
                labels_by_horizon=daily_labels,
                candidate_timestamps=candidates,
                test_from=start,
            )
        )
    for rule in model_registry["daily_rules"]:
        candidates = daily_signals.loc[daily_signals["hypothesis"].eq(rule["id"]), "timestamp"]
        metric_rows.extend(
            _evaluate(
                hypothesis=rule["id"],
                source_series="MOEX:CNYRUB_TOM:daily",
                direction=rule["direction"],
                panel=daily_panel,
                labels_by_horizon=daily_labels,
                candidate_timestamps=candidates,
                test_from=start,
            )
        )
    for model in model_registry["models"]:
        candidates = model_signals.loc[model_signals["hypothesis"].eq(model["id"]), "timestamp"]
        metric_rows.extend(
            _evaluate(
                hypothesis=model["id"],
                source_series="MOEX:CNYRUB_TOM:daily",
                direction=model["direction"],
                panel=daily_panel,
                labels_by_horizon=daily_labels,
                candidate_timestamps=candidates,
                test_from=start,
            )
        )

    candle_path = Path(candles_path)
    if candle_path.exists():
        candles = load_intraday_cny(candle_path)
        intraday_panel = _make_panel(candles["timestamp"], candles["price"], "MOEX:CNYRUB_TOM:10m")
        intraday_labels = {horizon: label_observations(intraday_panel, horizon) for horizon in HORIZONS}
        intraday_signals = intraday_session_signals(candles, test_from=start)
        signal_frames.append(intraday_signals)
        metric_rows.extend(
            _evaluate(
                hypothesis="H4_intraday_session_dip",
                source_series="MOEX:CNYRUB_TOM:10m",
                direction="window_closing",
                panel=intraday_panel,
                labels_by_horizon=intraday_labels,
                candidate_timestamps=intraday_signals["timestamp"],
                test_from=start,
            )
        )
    else:
        metric_rows.append(
            {
                "hypothesis": "H4_intraday_session_dip",
                "status": "not_testable",
                "source_series": "MOEX:CNYRUB_TOM:10m",
                "details": f"Нет пятилетнего candle-файла {candle_path}.",
            }
        )

    metric_rows.append(quote_fidelity_status(app_quotes_path))
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame(columns=SIGNAL_COLUMNS)
    signals = signals.loc[:, [column for column in SIGNAL_COLUMNS if column in signals.columns]]
    metrics = pd.DataFrame(metric_rows, columns=HYPOTHESIS_COLUMNS)
    signals.to_csv(artifact_path / "signals.csv", index=False)
    metrics.to_csv(artifact_path / "metrics.csv", index=False)
    meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "test_from": start.date().isoformat(),
        "daily_path": str(daily_path),
        "candles_path": str(candles_path),
        "app_quotes_path": str(app_quotes_path),
        "hypothesis_config_path": str(model_config_path),
        "hypothesis_config_sha256": hypotheses_sha256(model_config_path),
        "hypotheses": list(DAILY_DIRECTIONS)
        + [rule["id"] for rule in model_registry["daily_rules"]]
        + [model["id"] for model in model_registry["models"]]
        + ["H4_intraday_session_dip", "H5_quote_fidelity"],
        "signals_written": len(signals),
        "metrics_written": len(metrics),
    }
    (artifact_path / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=Path("data/raw/moex_daily.csv"))
    parser.add_argument("--candles", type=Path, default=Path("data/raw/moex_cny_candles_8y.csv"))
    parser.add_argument("--app-quotes", type=Path, default=Path("data/raw/app_quotes.csv"))
    parser.add_argument("--hypothesis-config", type=Path, default=Path("configs/hypotheses.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/hypotheses"))
    parser.add_argument("--test-from", default="2021-09-03")
    args = parser.parse_args(argv)
    summary = run_hypotheses(
        daily_path=args.daily,
        candles_path=args.candles,
        app_quotes_path=args.app_quotes,
        hypothesis_config_path=args.hypothesis_config,
        artifact_dir=args.artifact_dir,
        test_from=args.test_from,
    )
    print(f"Wrote {summary['signals_written']} signals and {summary['metrics_written']} hypothesis rows to {args.artifact_dir}")


if __name__ == "__main__":
    main()


__all__ = ["daily_candidate_signals", "intraday_session_signals", "run_hypotheses"]
