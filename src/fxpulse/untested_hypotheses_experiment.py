"""Paired tests for the data-ready hypotheses in UNTESTED_HYPOTHESES.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import _add_corridor_dummies, _apply_policy, _make_model
from fxpulse.training_history_experiment import add_training_labels


BASELINE = "b0_market_liquidity"
SHORT_CONTROL = "u03_short2y_control"
MODEL_VARIANTS = [
    BASELINE,
    "u01_discrete_hazard",
    "u02_realized_vol_control",
    "u02_intraday_path",
    "u05_residual_factors",
]
DERIVED_VARIANTS = [
    "u03_error_drift_switch",
    "u07_bootstrap_zero_drift",
    "u07_bootstrap_historical_drift",
    "u08_selection_cal_q70",
    "u08_selection_cal_q80",
    "u09_pooled_logistic",
    "u09_group_dro",
]


def load_config(path: Path | str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("experiment config must be preregistered schema_version 1")
    if config.get("corridors") != ["AMD", "KGS", "KZT", "TJS", "UZS"]:
        raise ValueError("the five case corridors are required in fixed order")
    if config.get("horizons") != [3, 5, 10]:
        raise ValueError("registered horizons must be [3, 5, 10]")
    if config["u07"].get("variants") != DERIVED_VARIANTS[1:3]:
        raise ValueError("U07 variants differ from implementation")
    if config["u09"].get("variants") != DERIVED_VARIANTS[-2:]:
        raise ValueError("U09 variants differ from implementation")
    return config


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_intraday_path_features(raw: pd.DataFrame, *, secid: str) -> pd.DataFrame:
    """Describe each completed session and expose it on the next session only."""
    required = {"dt_msk", "secid", "open", "high", "low", "close"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"hourly input lacks {sorted(missing)}")
    data = raw.loc[raw["secid"].eq(secid)].copy()
    data["dt_msk"] = pd.to_datetime(data["dt_msk"], errors="raise")
    data["timestamp"] = data["dt_msk"].dt.normalize()
    data = data.sort_values("dt_msk", kind="mergesort")
    data["hour_return"] = data.groupby("timestamp", sort=False)["close"].pct_change(fill_method=None)
    rows: list[dict[str, float | pd.Timestamp]] = []
    previous_close = np.nan
    for day, group in data.groupby("timestamp", sort=True):
        returns = pd.to_numeric(group["hour_return"], errors="coerce").dropna().to_numpy(dtype=float)
        squares = np.square(returns)
        total_square = float(squares.sum())
        absolute_sum = float(np.abs(returns).sum())
        open_price = float(group.iloc[0]["open"])
        close_price = float(group.iloc[-1]["close"])
        rows.append(
            {
                "timestamp": pd.Timestamp(day),
                "intraday__downside_semivariance_share": (
                    float(np.square(returns[returns < 0]).sum()) / total_square if total_square > 0 else 0.5
                ),
                "intraday__largest_move_share": (
                    float(np.abs(returns).max()) / absolute_sum if len(returns) and absolute_sum > 0 else 0.0
                ),
                "intraday__path_efficiency": (
                    abs(close_price / open_price - 1) / absolute_sum if absolute_sum > 0 else 0.0
                ),
                "intraday__signed_realized_semivariance": (
                    float(np.square(returns[returns > 0]).sum() - np.square(returns[returns < 0]).sum())
                ),
                "intraday__realized_volatility": math.sqrt(total_square),
                "intraday__range": float(group["high"].max() / group["low"].min() - 1),
                "intraday__overnight_gap": (
                    open_price / previous_close - 1 if np.isfinite(previous_close) and previous_close > 0 else np.nan
                ),
                "intraday__observations": float(len(group)),
            }
        )
        previous_close = close_price
    result = pd.DataFrame(rows).sort_values("timestamp", kind="mergesort")
    feature_columns = [column for column in result if column.startswith("intraday__")]
    result[feature_columns] = result[feature_columns].shift(1)
    return result.replace([np.inf, -np.inf], np.nan)


def attach_asof(frame: pd.DataFrame, features: pd.DataFrame, *, carry_days: int) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    extra = features.copy()
    extra["timestamp"] = pd.to_datetime(extra["timestamp"], errors="raise")
    if extra["timestamp"].duplicated().any():
        raise ValueError("as-of features contain duplicate dates")
    data["_order"] = np.arange(len(data))
    result = pd.merge_asof(
        data.sort_values("timestamp"),
        extra.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(int(carry_days), unit="D"),
    )
    return result.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)


def add_residual_factor_features(
    frame: pd.DataFrame,
    *,
    assets: list[str],
    window: int,
    minimum_observations: int,
) -> pd.DataFrame:
    """Remove the CNY/RUB component using coefficients estimated before each row."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    unique = data.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    x = pd.to_numeric(unique["market__cny_return_1"], errors="coerce")
    result = pd.DataFrame(index=unique.index)
    for asset in assets:
        y = pd.to_numeric(unique[f"market__{asset}_return_1"], errors="coerce")
        beta = (
            x.rolling(window, min_periods=minimum_observations).cov(y)
            / x.rolling(window, min_periods=minimum_observations).var().replace(0, np.nan)
        ).shift(1)
        residual = y - beta * x
        root = f"residual__{asset}"
        result[f"{root}_return_1"] = residual
        result[f"{root}_beta"] = beta
        result[f"{root}_zscore_20"] = (
            (residual - residual.rolling(20, min_periods=20).mean())
            / residual.rolling(20, min_periods=20).std().replace(0, np.nan)
        )
    merged = data.merge(result.reset_index(), on="timestamp", how="left", validate="many_to_one")
    return merged.replace([np.inf, -np.inf], np.nan)


def _fold_boundaries(config: dict[str, Any]) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    starts = pd.date_range(
        pd.Timestamp(config["test_from"]),
        pd.Timestamp(config["test_to"]),
        freq=f"{int(config['fold_frequency_months'])}MS",
    )
    result = []
    final = pd.Timestamp(config["test_to"])
    for start in starts:
        end = min(start + pd.DateOffset(months=int(config["fold_frequency_months"])), final)
        if end > start:
            result.append((start, end, f"{start.year}H{1 if start.month <= 6 else 2}"))
    return result


def _fit_catboost_ensemble(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    target: str,
    *,
    config: dict[str, Any],
    seed_offset: int,
) -> np.ndarray:
    predictions = []
    for seed in config["ensemble_seeds"]:
        model = _make_model(
            str(config["model"]),
            iterations=int(config["model_iterations"]),
            seed=int(seed) + int(seed_offset),
        )
        model.fit(train[columns], train[target].astype(int))
        predictions.append(model.predict_proba(test[columns])[:, 1])
    return np.mean(np.column_stack(predictions), axis=1)


def _first_improvement_steps(price: np.ndarray, *, horizon: int, tolerance_bps: float) -> np.ndarray:
    result = np.full(len(price), horizon + 1, dtype=int)
    result[max(0, len(price) - horizon) :] = 0
    for index in range(len(price) - horizon):
        current = price[index]
        for step in range(1, horizon + 1):
            if (current / price[index + step] - 1) * 10_000 > tolerance_bps:
                result[index] = step
                break
    return result


def fit_discrete_hazard(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    *,
    horizon: int,
    tolerance_bps: float,
    config: dict[str, Any],
    seed_offset: int,
) -> np.ndarray:
    ordered = train.sort_values("timestamp", kind="mergesort").copy()
    event_steps = _first_improvement_steps(
        ordered["price"].to_numpy(dtype=float), horizon=horizon, tolerance_bps=tolerance_bps
    )
    ordered["_event_step"] = event_steps
    ordered = ordered.loc[ordered["_event_step"].gt(0)].copy()
    pieces = []
    time_feature = str(config["u01"]["time_feature"])
    for step in range(1, horizon + 1):
        at_risk = ordered.loc[ordered["_event_step"].ge(step), columns].copy()
        at_risk[time_feature] = float(step) / horizon
        at_risk["_hazard_target"] = ordered.loc[ordered["_event_step"].ge(step), "_event_step"].eq(step).astype(int).to_numpy()
        pieces.append(at_risk)
    hazard_train = pd.concat(pieces, ignore_index=True)
    if hazard_train["_hazard_target"].nunique() < 2:
        return np.full(len(test), np.nan)
    prediction_steps = []
    expanded_columns = [*columns, time_feature]
    for seed in config["ensemble_seeds"]:
        model = _make_model(
            str(config["model"]),
            iterations=int(config["model_iterations"]),
            seed=int(seed) + int(seed_offset) + 1000,
        )
        model.fit(hazard_train[expanded_columns], hazard_train["_hazard_target"])
        hazards = []
        for step in range(1, horizon + 1):
            step_test = test[columns].copy()
            step_test[time_feature] = float(step) / horizon
            hazards.append(model.predict_proba(step_test[expanded_columns])[:, 1])
        prediction_steps.append(np.prod(1 - np.clip(np.column_stack(hazards), 0, 1), axis=1))
    return np.mean(np.column_stack(prediction_steps), axis=1)


def bootstrap_path_score(
    history: np.ndarray,
    current_volatility: float,
    *,
    horizon: int,
    tolerance_bps: float,
    paths: int,
    zero_drift: bool,
    scale_bounds: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    values = history[np.isfinite(history)]
    if len(values) < max(126, horizon + 20):
        return np.nan
    block_starts = np.arange(0, len(values) - horizon + 1)
    chosen = rng.choice(block_starts, size=paths, replace=True)
    sampled = np.vstack([values[index : index + horizon] for index in chosen])
    historical_vol = float(np.std(values[-min(252, len(values)) :], ddof=1))
    scale = current_volatility / historical_vol if historical_vol > 0 and np.isfinite(current_volatility) else 1.0
    scale = float(np.clip(scale, scale_bounds[0], scale_bounds[1]))
    if zero_drift:
        sampled = sampled - float(np.mean(values))
    sampled = sampled * scale
    paths_log = np.cumsum(np.log1p(np.clip(sampled, -0.99, None)), axis=1)
    improvement_boundary = -math.log1p(tolerance_bps / 10_000)
    safe = np.min(paths_log, axis=1) >= improvement_boundary
    return float(np.mean(safe))


def adwin_switch_scores(
    baseline_score: np.ndarray,
    short_score: np.ndarray,
    target: np.ndarray,
    *,
    delay: int,
    delta: float,
    minimum_subwindow: int,
    maximum_window: int,
    alarm_hold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic ADWIN-style mean-loss detector with delayed labels."""
    result = np.asarray(baseline_score, dtype=float).copy()
    alarms = np.zeros(len(result), dtype=int)
    losses: list[float] = []
    active_until = -1
    for index in range(len(result)):
        matured = index - int(delay)
        if matured >= 0:
            p = float(np.clip(baseline_score[matured], 1e-6, 1 - 1e-6))
            y = int(target[matured])
            losses.append(float(-(y * math.log(p) + (1 - y) * math.log(1 - p))))
            losses = losses[-int(maximum_window) :]
            if len(losses) >= 2 * int(minimum_subwindow):
                split = len(losses) // 2
                old = np.asarray(losses[:split])
                recent = np.asarray(losses[split:])
                bound = math.sqrt(0.5 * math.log(4 / delta) * (1 / len(old) + 1 / len(recent)))
                if float(recent.mean() - old.mean()) > bound:
                    alarms[index] = 1
                    active_until = index + int(alarm_hold)
                    losses = list(recent)
        if index <= active_until and np.isfinite(short_score[index]):
            result[index] = short_score[index]
    return result, alarms


def _class_weights(target: np.ndarray) -> np.ndarray:
    counts = np.bincount(target.astype(int), minlength=2).astype(float)
    return np.asarray([len(target) / (2 * counts[int(value)]) for value in target], dtype=float)


def pooled_logistic_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    *,
    target: str,
    config: dict[str, Any],
    group_dro: bool,
) -> np.ndarray:
    train_aug, test_aug, expanded = _add_corridor_dummies(
        train, test, list(columns), list(config["corridors"])
    )
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_aug[expanded]))
    x_test = scaler.transform(imputer.transform(test_aug[expanded]))
    y = train_aug[target].to_numpy(dtype=int)
    base_weight = _class_weights(y)
    if not group_dro:
        model = LogisticRegression(C=0.5, max_iter=1000, random_state=int(config["random_seed"]))
        model.fit(x_train, y, sample_weight=base_weight)
        return model.predict_proba(x_test)[:, 1]

    high_vol = train_aug["regime__volatility_percentile_252"].ge(
        float(config["u09"]["volatility_cut"])
    ).astype(int)
    group = train_aug["corridor"].astype(str) + "_v" + high_vol.astype(str)
    group_names = sorted(group.unique())
    group_weight = {name: 1 / len(group_names) for name in group_names}
    model = LogisticRegression(C=0.5, max_iter=1000, warm_start=True, random_state=int(config["random_seed"]))
    for _ in range(int(config["u09"]["outer_iterations"])):
        counts = group.value_counts()
        sample_weight = base_weight * np.asarray(
            [group_weight[name] / counts[name] * len(group) for name in group], dtype=float
        )
        model.fit(x_train, y, sample_weight=sample_weight)
        probability = np.clip(model.predict_proba(x_train)[:, 1], 1e-6, 1 - 1e-6)
        losses = {
            name: log_loss(y[group.eq(name)], probability[group.eq(name)], labels=[0, 1])
            for name in group_names
        }
        for name in group_names:
            group_weight[name] *= math.exp(float(config["u09"]["eta"]) * losses[name])
        total = sum(group_weight.values())
        group_weight = {name: value / total for name, value in group_weight.items()}
    return model.predict_proba(x_test)[:, 1]


def _prepare(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(config["input"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    intraday = add_intraday_path_features(pd.read_csv(config["hourly_input"]), secid=str(config["u02"]["secid"]))
    frame = attach_asof(frame, intraday, carry_days=int(config["u02"]["alignment_carry_days"]))
    frame = add_residual_factor_features(
        frame,
        assets=list(config["u05"]["assets"]),
        window=int(config["u05"]["rolling_window"]),
        minimum_observations=int(config["u05"]["minimum_observations"]),
    )
    return add_training_labels(frame, config).sort_values(["corridor", "timestamp"], kind="mergesort")


def build_scores(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_columns = [column for column in frame if column.startswith(tuple(config["feature_prefixes"]))]
    intraday_columns = [column for column in frame if column.startswith("intraday__")]
    ordinary_intraday_columns = ["intraday__realized_volatility", "intraday__observations"]
    residual_columns = [column for column in frame if column.startswith("residual__")]
    score_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    boundaries = _fold_boundaries(config)

    for corridor in config["corridors"]:
        corridor_data = frame.loc[frame["corridor"].eq(corridor)].copy()
        for horizon_value in config["horizons"]:
            horizon = int(horizon_value)
            outcome = f"outcome__regret_{horizon}"
            benefit = f"outcome__benefit_{horizon}"
            label = f"training_target_{horizon}"
            usable = corridor_data.loc[corridor_data[outcome].notna() & corridor_data[label].notna()].copy()
            usable["target"] = usable[outcome].le(float(config["evaluation_tolerance_bps"])).astype(int)
            usable["regret_bps"] = usable[outcome]
            usable["benefit_bps"] = usable[benefit]
            usable["week"] = usable["timestamp"].dt.to_period("W").astype(str)
            weekly_rate = usable.groupby("week", sort=False)["target"].mean()
            usable["matched_week_hit_rate"] = usable["week"].map(weekly_rate).astype(float)

            for fold_index, (start, end, fold_name) in enumerate(boundaries):
                purged_end = usable.loc[usable["timestamp"].lt(start)].index
                train_all = usable.loc[purged_end].sort_values("timestamp", kind="mergesort")
                train_all = train_all.iloc[:-horizon] if len(train_all) > horizon else train_all.iloc[0:0]
                train = train_all.loc[train_all["timestamp"].ge(start - pd.DateOffset(years=int(config["rolling_training_years"])))].copy()
                short_train = train_all.loc[train_all["timestamp"].ge(start - pd.DateOffset(years=int(config["short_training_years"])))].copy()
                test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                if len(train) < int(config["minimum_training_observations"]) or len(test) < 20:
                    continue
                if train[label].nunique() < 2:
                    continue
                seed_offset = 100 * fold_index + 10 * horizon + list(config["corridors"]).index(corridor)
                predictions = {
                    BASELINE: _fit_catboost_ensemble(train, test, base_columns, label, config=config, seed_offset=seed_offset),
                    "u01_discrete_hazard": fit_discrete_hazard(
                        train,
                        test,
                        base_columns,
                        horizon=horizon,
                        tolerance_bps=float(config["evaluation_tolerance_bps"]),
                        config=config,
                        seed_offset=seed_offset,
                    ),
                    "u02_realized_vol_control": _fit_catboost_ensemble(
                        train,
                        test,
                        [*base_columns, *ordinary_intraday_columns],
                        label,
                        config=config,
                        seed_offset=seed_offset + 1500,
                    ),
                    "u02_intraday_path": _fit_catboost_ensemble(
                        train, test, [*base_columns, *intraday_columns], label, config=config, seed_offset=seed_offset + 2000
                    ),
                    "u05_residual_factors": _fit_catboost_ensemble(
                        train, test, [*base_columns, *residual_columns], label, config=config, seed_offset=seed_offset + 3000
                    ),
                }
                if len(short_train) >= int(config["minimum_training_observations"]):
                    predictions[SHORT_CONTROL] = _fit_catboost_ensemble(
                        short_train, test, base_columns, label, config=config, seed_offset=seed_offset + 4000
                    )
                else:
                    predictions[SHORT_CONTROL] = predictions[BASELINE]

                returns = usable.set_index("timestamp")["base__return_1"]
                rng = np.random.default_rng(int(config["random_seed"]) + seed_offset)
                for drift_variant, zero_drift in (("u07_bootstrap_zero_drift", True), ("u07_bootstrap_historical_drift", False)):
                    path_scores = []
                    for timestamp, current_volatility in zip(
                        test["timestamp"], test["base__volatility_20"], strict=True
                    ):
                        history = returns.loc[returns.index < timestamp].tail(int(config["u07"]["history_observations"])).to_numpy(dtype=float)
                        path_scores.append(
                            bootstrap_path_score(
                                history,
                                float(current_volatility),
                                horizon=horizon,
                                tolerance_bps=float(config["evaluation_tolerance_bps"]),
                                paths=int(config["u07"]["simulation_paths"]),
                                zero_drift=zero_drift,
                                scale_bounds=tuple(map(float, config["u07"]["volatility_scale_bounds"])),
                                rng=rng,
                            )
                        )
                    predictions[drift_variant] = np.asarray(path_scores, dtype=float)

                for variant, score in predictions.items():
                    export = test[
                        ["timestamp", "corridor", "week", "target", "regret_bps", "benefit_bps", "matched_week_hit_rate"]
                    ].copy()
                    export["variant"] = variant
                    export["horizon"] = horizon
                    export["test_fold"] = fold_name
                    export["score"] = score
                    score_parts.append(export)
                diagnostics.append(
                    {
                        "corridor": corridor,
                        "horizon": horizon,
                        "test_fold": fold_name,
                        "train_rows": len(train),
                        "short_train_rows": len(short_train),
                        "test_rows": len(test),
                        "intraday_complete_share": float(test[intraday_columns].notna().all(axis=1).mean()),
                        "residual_complete_share": float(test[residual_columns].notna().all(axis=1).mean()),
                    }
                )
            print(f"completed base/U01/U02/U05/U07 {corridor} h={horizon}", flush=True)

    scores = pd.concat(score_parts, ignore_index=True)
    scores = add_u03_scores(scores, config)
    scores = pd.concat([scores, build_u09_scores(frame, config)], ignore_index=True)
    return scores, pd.DataFrame(diagnostics)


def add_u03_scores(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = [scores]
    for (corridor, horizon), group in scores.groupby(["corridor", "horizon"], sort=True):
        wide = group.loc[group["variant"].isin([BASELINE, SHORT_CONTROL])].pivot(
            index=["timestamp", "corridor", "week", "target", "regret_bps", "benefit_bps", "matched_week_hit_rate", "horizon", "test_fold"],
            columns="variant",
            values="score",
        ).reset_index()
        wide = wide.sort_values("timestamp", kind="mergesort")
        switched, alarms = adwin_switch_scores(
            wide[BASELINE].to_numpy(dtype=float),
            wide[SHORT_CONTROL].to_numpy(dtype=float),
            wide["target"].to_numpy(dtype=int),
            delay=int(horizon),
            delta=float(config["u03"]["detector_delta"]),
            minimum_subwindow=int(config["u03"]["minimum_subwindow"]),
            maximum_window=int(config["u03"]["maximum_window"]),
            alarm_hold=int(config["u03"]["alarm_hold_observations"]),
        )
        export = wide[
            ["timestamp", "corridor", "week", "target", "regret_bps", "benefit_bps", "matched_week_hit_rate", "horizon", "test_fold"]
        ].copy()
        export["variant"] = "u03_error_drift_switch"
        export["score"] = switched
        export["drift_alarm"] = alarms
        output.append(export)
    return pd.concat(output, ignore_index=True, sort=False)


def build_u09_scores(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [column for column in frame if column.startswith(tuple(config["feature_prefixes"]))]
    parts: list[pd.DataFrame] = []
    for horizon_value in config["horizons"]:
        horizon = int(horizon_value)
        outcome = f"outcome__regret_{horizon}"
        benefit = f"outcome__benefit_{horizon}"
        label = f"training_target_{horizon}"
        usable = frame.loc[frame[outcome].notna() & frame[label].notna()].copy()
        usable["target"] = usable[outcome].le(float(config["evaluation_tolerance_bps"])).astype(int)
        usable["regret_bps"] = usable[outcome]
        usable["benefit_bps"] = usable[benefit]
        usable["week"] = usable["timestamp"].dt.to_period("W").astype(str)
        weekly_rate = usable.groupby(["corridor", "week"], sort=False)["target"].mean()
        usable["matched_week_hit_rate"] = [
            float(weekly_rate.loc[(corridor, week)])
            for corridor, week in zip(usable["corridor"], usable["week"], strict=True)
        ]
        for start, end, fold_name in _fold_boundaries(config):
            past = usable.loc[usable["timestamp"].lt(start)].copy()
            # Purge the last h observations separately in every corridor.  A
            # calendar-day cutoff is not enough around long market holidays.
            purged_parts = []
            for _, corridor_past in past.groupby("corridor", sort=False):
                corridor_past = corridor_past.sort_values("timestamp", kind="mergesort")
                purged_parts.append(
                    corridor_past.iloc[:-horizon]
                    if len(corridor_past) > horizon
                    else corridor_past.iloc[0:0]
                )
            train = pd.concat(purged_parts, ignore_index=True)
            train = train.loc[
                train["timestamp"].ge(start - pd.DateOffset(years=int(config["rolling_training_years"])))
            ].copy()
            test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
            if len(train) < int(config["minimum_training_observations"]) * len(config["corridors"]) or len(test) < 100:
                continue
            for variant, is_dro in (("u09_pooled_logistic", False), ("u09_group_dro", True)):
                export = test[
                    ["timestamp", "corridor", "week", "target", "regret_bps", "benefit_bps", "matched_week_hit_rate"]
                ].copy()
                export["variant"] = variant
                export["horizon"] = horizon
                export["test_fold"] = fold_name
                export["score"] = pooled_logistic_scores(
                    train, test, columns, target=label, config=config, group_dro=is_dro
                )
                parts.append(export)
        print(f"completed U09 h={horizon}", flush=True)
    return pd.concat(parts, ignore_index=True)


def select_standard_signals(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    parts = []
    policy = config["adaptive_policy"]
    excluded = {"u08_selection_cal_q70", "u08_selection_cal_q80"}
    for (_, _, _), group in scores.loc[~scores["variant"].isin(excluded)].groupby(
        ["variant", "corridor", "horizon"], sort=True
    ):
        group = group.sort_values("timestamp", kind="mergesort").copy()
        candidates = adaptive_candidates(
            group,
            share=float(policy["top_score_share"]),
            lookback=int(policy["lookback_observations"]),
            minimum_history=int(policy["minimum_history_observations"]),
        )
        selected = _apply_policy(
            candidates,
            cooldown_days=int(policy["cooldown_days"]),
            weekly_cap=int(policy["weekly_cap"]),
        )
        parts.append(selected)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def selection_calibrated_signals(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    parts = []
    policy = config["adaptive_policy"]
    settings = config["u08"]
    baseline = scores.loc[scores["variant"].eq(BASELINE)].copy()
    for (corridor, horizon), group in baseline.groupby(["corridor", "horizon"], sort=True):
        group = group.sort_values("timestamp", kind="mergesort").copy()
        candidates = adaptive_candidates(
            group,
            share=float(policy["top_score_share"]),
            lookback=int(policy["lookback_observations"]),
            minimum_history=int(policy["minimum_history_observations"]),
        ).copy()
        candidate_index = set(candidates.index)
        for quantile in settings["quantiles"]:
            history: list[tuple[float, float]] = []
            accepted = []
            upper = np.full(len(group), np.nan)
            index_positions = {index: position for position, index in enumerate(group.index)}
            for position, (index, row) in enumerate(group.iterrows()):
                matured_position = position - int(horizon)
                if matured_position >= 0:
                    matured_index = group.index[matured_position]
                    if matured_index in candidate_index:
                        matured = group.iloc[matured_position]
                        history.append((float(matured["score"]), float(matured["regret_bps"])))
                        history = history[-int(settings["calibration_observations"]) :]
                if index not in candidate_index or len(history) < int(settings["minimum_calibration_observations"]):
                    continue
                nearest = sorted(history, key=lambda item: abs(item[0] - float(row["score"])))[: int(settings["nearest_scores"])]
                upper[position] = float(np.quantile([item[1] for item in nearest], float(quantile), method="higher"))
                if upper[position] <= float(settings["maximum_regret_bps"]):
                    accepted.append(index)
            candidate_frame = group.loc[accepted].copy()
            candidate_frame["calibrated_regret_upper_bps"] = [upper[index_positions[index]] for index in accepted]
            selected = _apply_policy(
                candidate_frame,
                cooldown_days=int(policy["cooldown_days"]),
                weekly_cap=int(policy["weekly_cap"]),
            )
            selected["variant"] = f"u08_selection_cal_q{int(float(quantile) * 100)}"
            parts.append(selected)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def add_u08_score_views(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Give U08 policies the same baseline predictions and evaluation universe."""
    baseline = scores.loc[scores["variant"].eq(BASELINE)]
    views = [scores]
    for quantile in config["u08"]["quantiles"]:
        view = baseline.copy()
        view["variant"] = f"u08_selection_cal_q{int(float(quantile) * 100)}"
        views.append(view)
    return pd.concat(views, ignore_index=True, sort=False)


def summarize(scores: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corridor_rows = []
    fold_rows = []
    annual_rows = []
    for (variant, corridor, horizon), score_group in scores.groupby(["variant", "corridor", "horizon"], sort=True):
        chosen = signals.loc[
            signals["variant"].eq(variant)
            & signals["corridor"].eq(corridor)
            & signals["horizon"].eq(horizon)
        ]
        corridor_rows.append(_metric_row(score_group, chosen, variant=variant, corridor=corridor, horizon=int(horizon)))
        for fold, fold_scores in score_group.groupby("test_fold", sort=True):
            fold_signals = chosen.loc[chosen["test_fold"].eq(fold)]
            fold_rows.append(_metric_row(fold_scores, fold_signals, variant=variant, corridor=corridor, horizon=int(horizon), test_fold=fold))
        years = score_group["timestamp"].dt.year
        for year in sorted(years.unique()):
            year_scores = score_group.loc[years.eq(year)]
            year_signals = chosen.loc[chosen["timestamp"].dt.year.eq(year)]
            annual_rows.append(_metric_row(year_scores, year_signals, variant=variant, corridor=corridor, horizon=int(horizon), year=int(year)))
    corridor_summary = pd.DataFrame(corridor_rows)
    aggregate_rows = []
    for (variant, horizon), group in corridor_summary.groupby(["variant", "horizon"], sort=True):
        count = int(group["signals"].sum())
        aggregate_rows.append(
            {
                "variant": variant,
                "horizon": int(horizon),
                "corridors": len(group),
                "signals": count,
                "hit_rate": float((group["hits"].sum() / count) if count else np.nan),
                "baseline_hit_rate": float(group["base_hits"].sum() / group["base_count"].sum()),
                "matched_random_hit_rate": float(group["matched_expected_hits"].sum() / count) if count else np.nan,
                "lift": float((group["hits"].sum() / count) / (group["base_hits"].sum() / group["base_count"].sum())) if count else np.nan,
                "lift_vs_matched_random": float(group["hits"].sum() / group["matched_expected_hits"].sum()) if count else np.nan,
                "regret_mean_bps": float(group["regret_sum_bps"].sum() / count) if count else np.nan,
                "regret_p90_bps": _weighted_signal_quantile(signals, variant, int(horizon), 0.9),
                "severe_error_share": float(group["severe_errors"].sum() / count) if count else np.nan,
                "benefit_mean_bps": float(group["benefit_sum_bps"].sum() / count) if count else np.nan,
                "signals_per_week_per_corridor": float(count / group["duration_weeks"].sum()) if count else 0.0,
                "brier_score": float(
                    np.average(group["brier_score"], weights=group["base_count"])
                ),
            }
        )
    return corridor_summary, pd.DataFrame(aggregate_rows), pd.DataFrame(fold_rows + annual_rows)


def _metric_row(score_group: pd.DataFrame, chosen: pd.DataFrame, **identity: object) -> dict[str, object]:
    count = len(chosen)
    duration = max(float((score_group["timestamp"].max() - score_group["timestamp"].min()).days) / 7, 1 / 7)
    hits = int(chosen["target"].sum()) if count else 0
    base_count = len(score_group)
    base_hits = int(score_group["target"].sum())
    matched = float(chosen["matched_week_hit_rate"].sum()) if count else 0.0
    valid_probability = score_group[["score", "target"]].dropna()
    brier = float(
        np.mean(
            np.square(
                np.clip(valid_probability["score"].to_numpy(dtype=float), 0, 1)
                - valid_probability["target"].to_numpy(dtype=float)
            )
        )
    ) if len(valid_probability) else np.nan
    return {
        **identity,
        "signals": count,
        "hits": hits,
        "base_count": base_count,
        "base_hits": base_hits,
        "matched_expected_hits": matched,
        "hit_rate": hits / count if count else np.nan,
        "baseline_hit_rate": base_hits / base_count if base_count else np.nan,
        "lift": (hits / count) / (base_hits / base_count) if count and base_hits else np.nan,
        "lift_vs_matched_random": hits / matched if count and matched > 0 else np.nan,
        "regret_sum_bps": float(chosen["regret_bps"].sum()) if count else 0.0,
        "regret_mean_bps": float(chosen["regret_bps"].mean()) if count else np.nan,
        "regret_p90_bps": float(chosen["regret_bps"].quantile(0.9)) if count else np.nan,
        "severe_errors": int(chosen["regret_bps"].gt(100).sum()) if count else 0,
        "severe_error_share": float(chosen["regret_bps"].gt(100).mean()) if count else np.nan,
        "benefit_sum_bps": float(chosen["benefit_bps"].sum()) if count else 0.0,
        "benefit_mean_bps": float(chosen["benefit_bps"].mean()) if count else np.nan,
        "duration_weeks": duration,
        "signals_per_week": count / duration if count else 0.0,
        "brier_score": brier,
    }


def _weighted_signal_quantile(signals: pd.DataFrame, variant: str, horizon: int, quantile: float) -> float:
    values = signals.loc[
        signals["variant"].eq(variant) & signals["horizon"].eq(horizon), "regret_bps"
    ]
    return float(values.quantile(quantile)) if len(values) else np.nan


def compare_to_baseline(
    corridor: pd.DataFrame,
    aggregate: pd.DataFrame,
    detail: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "lift",
        "lift_vs_matched_random",
        "regret_mean_bps",
        "regret_p90_bps",
        "severe_error_share",
        "benefit_mean_bps",
        "signals_per_week",
        "brier_score",
    ]
    base = corridor.loc[corridor["variant"].eq(BASELINE), ["corridor", "horizon", *metrics]]
    candidates = corridor.loc[corridor["variant"].ne(BASELINE)]
    comparison = candidates.merge(base, on=["corridor", "horizon"], suffixes=("", "_baseline"), validate="many_to_one")
    for metric in metrics:
        comparison[f"delta_{metric}"] = comparison[metric] - comparison[f"{metric}_baseline"]

    primary = int(config["success_gate"]["primary_horizon"])
    fold = detail.loc[detail["test_fold"].notna() & detail["horizon"].eq(primary)].copy()
    fold_base = fold.loc[fold["variant"].eq(BASELINE), ["corridor", "test_fold", "lift_vs_matched_random"]]
    fold_delta = fold.loc[fold["variant"].ne(BASELINE)].merge(
        fold_base, on=["corridor", "test_fold"], suffixes=("", "_baseline"), validate="many_to_one"
    )
    fold_delta["delta"] = fold_delta["lift_vs_matched_random"] - fold_delta["lift_vs_matched_random_baseline"]

    annual = detail.loc[detail["year"].notna() & detail["horizon"].eq(primary)].copy()
    annual_base = annual.loc[annual["variant"].eq(BASELINE), ["corridor", "year", "lift_vs_matched_random"]]
    annual_delta = annual.loc[annual["variant"].ne(BASELINE)].merge(
        annual_base, on=["corridor", "year"], suffixes=("", "_baseline"), validate="many_to_one"
    )
    annual_delta["delta"] = annual_delta["lift_vs_matched_random"] - annual_delta["lift_vs_matched_random_baseline"]

    gates = []
    gate = config["success_gate"]
    primary_aggregate = aggregate.loc[aggregate["horizon"].eq(primary)].set_index("variant")
    for variant, group in comparison.loc[comparison["horizon"].eq(primary)].groupby("variant", sort=True):
        fold_group = fold_delta.loc[fold_delta["variant"].eq(variant)]
        annual_group = annual_delta.loc[annual_delta["variant"].eq(variant)]
        aggregate_row = primary_aggregate.loc[variant]
        checks = {
            "median_delta_pass": float(group["delta_lift_vs_matched_random"].median()) >= float(gate["minimum_median_corridor_delta_same_week_lift"]),
            "positive_corridors_pass": int(group["delta_lift_vs_matched_random"].gt(0).sum()) >= int(gate["minimum_positive_corridors"]),
            "positive_folds_pass": float(fold_group["delta"].gt(0).mean()) >= float(gate["minimum_positive_fold_fraction"]),
            "frequency_pass": float(gate["minimum_signals_per_week"]) <= float(aggregate_row["signals_per_week_per_corridor"]) <= float(gate["maximum_signals_per_week"]),
            "mean_regret_pass": float(group["delta_regret_mean_bps"].mean()) <= 0,
            "severe_error_pass": float(group["delta_severe_error_share"].mean()) <= 0,
            "annual_floor_pass": annual_group["delta"].dropna().min() >= float(gate["maximum_worst_annual_corridor_delta"]),
        }
        gates.append(
            {
                "variant": variant,
                "horizon": primary,
                "median_corridor_delta_same_week_lift": float(group["delta_lift_vs_matched_random"].median()),
                "positive_corridors": int(group["delta_lift_vs_matched_random"].gt(0).sum()),
                "positive_fold_fraction": float(fold_group["delta"].gt(0).mean()),
                "worst_annual_corridor_delta": float(annual_group["delta"].dropna().min()) if annual_group["delta"].notna().any() else np.nan,
                **checks,
                "all_gate_checks_pass": all(checks.values()),
            }
        )
    return comparison, pd.DataFrame(gates)


def bootstrap_deltas(signals: pd.DataFrame, *, config: dict[str, Any]) -> pd.DataFrame:
    horizon = int(config["success_gate"]["primary_horizon"])
    data = signals.loc[signals["horizon"].eq(horizon)].copy()
    data["calendar_week"] = data["timestamp"].dt.to_period("W").astype(str)
    weeks = sorted(data["calendar_week"].unique())
    block_size = int(config["bootstrap_block_weeks"])
    blocks = [weeks[index : index + block_size] for index in range(0, len(weeks), block_size)]
    rng = np.random.default_rng(int(config["random_seed"]) + 99000)
    rows = []
    variants = sorted(set(data["variant"]) - {BASELINE})
    draws_count = int(config["bootstrap_samples"])
    chosen_blocks = rng.integers(0, len(blocks), size=(draws_count, len(blocks)))

    # Aggregate once.  Re-filtering the full signal table inside every draw is
    # equivalent but needlessly turns a small block bootstrap into minutes.
    block_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in [BASELINE, *variants]:
        selected = data.loc[data["variant"].eq(name)]
        weekly = selected.groupby("calendar_week").agg(
            hits=("target", "sum"), expected=("matched_week_hit_rate", "sum")
        )
        block_hits = np.asarray(
            [weekly.reindex(block, fill_value=0)["hits"].sum() for block in blocks], dtype=float
        )
        block_expected = np.asarray(
            [weekly.reindex(block, fill_value=0)["expected"].sum() for block in blocks], dtype=float
        )
        block_lookup[name] = block_hits, block_expected

    baseline_hits, baseline_expected = block_lookup[BASELINE]
    sampled_baseline_expected = baseline_expected[chosen_blocks].sum(axis=1)
    sampled_baseline_lift = np.divide(
        baseline_hits[chosen_blocks].sum(axis=1),
        sampled_baseline_expected,
        out=np.full(draws_count, np.nan),
        where=sampled_baseline_expected > 0,
    )
    for variant in variants:
        hits, expected = block_lookup[variant]
        sampled_expected = expected[chosen_blocks].sum(axis=1)
        sampled_lift = np.divide(
            hits[chosen_blocks].sum(axis=1),
            sampled_expected,
            out=np.full(draws_count, np.nan),
            where=sampled_expected > 0,
        )
        array = sampled_lift - sampled_baseline_lift
        valid = array[np.isfinite(array)]
        p_one_sided = float((1 + np.sum(valid <= 0)) / (len(valid) + 1))
        rows.append(
            {
                "variant": variant,
                "horizon": horizon,
                "delta_same_week_lift_mean": float(valid.mean()),
                "ci_low": float(np.quantile(valid, 0.025)),
                "ci_high": float(np.quantile(valid, 0.975)),
                "p_one_sided": p_one_sided,
            }
        )
    result = pd.DataFrame(rows).sort_values("p_one_sided", kind="mergesort")
    m = len(result)
    adjusted = np.maximum.accumulate(
        np.minimum(1.0, result["p_one_sided"].to_numpy() * np.arange(m, 0, -1))
    )
    result["holm_p"] = adjusted
    return result.sort_values("variant", kind="mergesort")


def run(*, config_path: Path | str, artifact_dir: Path | str) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_config(config_path)
    frame = _prepare(config)
    scores, diagnostics = build_scores(frame, config)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], errors="raise")
    u08 = selection_calibrated_signals(scores, config)
    scores = add_u08_score_views(scores, config)
    standard = select_standard_signals(scores, config)
    signals = pd.concat([standard, u08], ignore_index=True, sort=False)
    corridor, aggregate, detail = summarize(scores, signals)
    comparison, gates = compare_to_baseline(corridor, aggregate, detail, config)
    bootstrap = bootstrap_deltas(signals, config=config)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "scores.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    diagnostics.to_csv(output / "data_diagnostics.csv", index=False)
    corridor.to_csv(output / "corridor_summary.csv", index=False)
    aggregate.to_csv(output / "aggregate_summary.csv", index=False)
    detail.to_csv(output / "fold_and_year_summary.csv", index=False)
    comparison.to_csv(output / "paired_comparison.csv", index=False)
    gates.to_csv(output / "success_gates.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_h5.csv", index=False)
    meta = {
        "config": str(config_path),
        "config_sha256": _hash(config_path),
        "input_sha256": _hash(Path(config["input"])),
        "hourly_input_sha256": _hash(Path(config["hourly_input"])),
        "feature_rows": len(frame),
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "tested_hypotheses": ["U01", "U02", "U03", "U05", "U07", "U08", "U09"],
        "warning": "Exploratory on reviewed 2022-2026 history; no variant is independently confirmed without a new holdout.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/untested_hypotheses_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/untested_hypotheses/data_ready"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
