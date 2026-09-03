"""Leak-free, period-by-period selection of cross-market rule-based signals.

The candidate library is fixed in JSON before a run.  For each quarterly
out-of-time window the runner uses only the expanding, horizon-purged prefix
to choose one interpretable rule; it then evaluates that chosen rule on the
following quarter.  Every factor is lagged by one target observation because
the exact ordering of different MOEX market closes is not yet part of the raw
contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Any

import pandas as pd

from fxpulse.labeling import HORIZONS, evaluate_positions, label_observations
from fxpulse.panel import MSK


FOLD_COLUMNS = (
    "fold",
    "test_start",
    "test_end",
    "training_observations",
    "purged_tail_observations",
    "candidate_rules",
    "selection_status",
    "selected_rule_id",
    "factor_instrument_id",
    "return_window",
    "tail_quantile",
    "tail",
    "direction",
    "train_signal_count",
    "train_signals_per_week",
    "train_hit_rate",
    "train_baseline_hit_rate",
    "train_lift",
    "train_benefit_fwd_bps",
    "test_candidate_signal_count",
    "test_dispatched_signal_count",
    "test_signals_per_week",
    "test_hit_rate",
    "test_baseline_hit_rate",
    "test_lift",
    "test_benefit_fwd_bps",
    "test_frequency_in_policy",
)
LEADERBOARD_COLUMNS = (
    "fold",
    "rule_id",
    "factor_instrument_id",
    "return_window",
    "tail_quantile",
    "tail",
    "direction",
    "signal_count",
    "signals_per_week",
    "hit_rate",
    "baseline_hit_rate",
    "lift",
    "benefit_fwd_bps",
    "meets_minimum_signals",
    "frequency_in_policy",
    "meets_selection_lift",
    "meets_benefit",
    "eligible_for_selection",
)
SIGNAL_COLUMNS = (
    "fold",
    "selection_status",
    "timestamp",
    "value_date",
    "rule_id",
    "factor_instrument_id",
    "direction",
    "strength",
    "communication_allowed",
    "details",
)
SUMMARY_COLUMNS = (
    "selection_status",
    "communication_allowed",
    "direction",
    "folds",
    "signal_count",
    "eligible_base_count",
    "hit_rate",
    "baseline_hit_rate",
    "lift",
    "benefit_sym_bps",
    "benefit_fwd_bps",
    "signals_per_week",
    "signals_per_month",
    "cluster_share",
    "interval_cv",
    "weeks_with_signal",
    "max_signals_in_fold_week",
)


@dataclass(frozen=True)
class RuleFold:
    """A leak-free expanding-train / one-quarter-test split."""

    name: str
    train_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    purged_tail_observations: int


def rule_config_sha256(path: Path | str = Path("configs/rule_selection.json")) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rule_config(path: Path | str = Path("configs/rule_selection.json")) -> dict[str, Any]:
    """Load the preregistered rule library and communication-policy gates."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Only rule-selection schema_version 1 is supported")
    evaluation = config.get("evaluation")
    library = config.get("rule_library")
    if not isinstance(evaluation, dict) or not isinstance(library, dict):
        raise ValueError("rule-selection config requires evaluation and rule_library objects")
    evaluation_fields = {
        "target_instrument_id",
        "horizon_observations",
        "min_training_observations",
        "test_period",
        "factor_lag_observations",
        "minimum_training_signals",
        "minimum_signals_per_week",
        "maximum_signals_per_week",
        "selection_minimum_lift",
        "promotion_minimum_lift",
        "minimum_training_benefit_bps",
        "no_signal_fallback",
    }
    missing = evaluation_fields - set(evaluation)
    if missing:
        raise ValueError(f"evaluation lacks: {', '.join(sorted(missing))}")
    if evaluation["horizon_observations"] not in HORIZONS:
        raise ValueError(f"horizon_observations must be one of {HORIZONS}")
    if evaluation["test_period"] != "quarter":
        raise ValueError("Only quarterly test_period is supported")
    for name in ("min_training_observations", "factor_lag_observations", "minimum_training_signals"):
        if not isinstance(evaluation[name], int) or evaluation[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "minimum_signals_per_week",
        "maximum_signals_per_week",
        "selection_minimum_lift",
        "promotion_minimum_lift",
        "minimum_training_benefit_bps",
    ):
        if not isinstance(evaluation[name], int | float):
            raise ValueError(f"{name} must be numeric")
    if evaluation["minimum_signals_per_week"] <= 0:
        raise ValueError("minimum_signals_per_week must be positive")
    if evaluation["maximum_signals_per_week"] < evaluation["minimum_signals_per_week"]:
        raise ValueError("maximum_signals_per_week must be at least minimum_signals_per_week")
    if evaluation["no_signal_fallback"] != "do_not_send":
        raise ValueError("Only no_signal_fallback=do_not_send is supported")
    library_fields = {
        "kind",
        "return_windows",
        "comparison_lookback_observations",
        "tail_quantiles",
        "tails",
        "directions",
    }
    missing = library_fields - set(library)
    if missing:
        raise ValueError(f"rule_library lacks: {', '.join(sorted(missing))}")
    if library["kind"] != "lagged_factor_return_quantile":
        raise ValueError("Only lagged_factor_return_quantile rules are supported")
    if not isinstance(library["comparison_lookback_observations"], int) or library[
        "comparison_lookback_observations"
    ] <= 1:
        raise ValueError("comparison_lookback_observations must be an integer above one")
    if not library["return_windows"] or not all(isinstance(value, int) and value > 0 for value in library["return_windows"]):
        raise ValueError("return_windows must contain positive integers")
    if not library["tail_quantiles"] or not all(
        isinstance(value, int | float) and 0 < value < 0.5 for value in library["tail_quantiles"]
    ):
        raise ValueError("tail_quantiles must be in (0, 0.5)")
    if set(library["tails"]) - {"lower", "upper"} or not library["tails"]:
        raise ValueError("tails must contain lower and/or upper")
    if set(library["directions"]) - {"favorable", "window_closing"} or not library["directions"]:
        raise ValueError("directions must contain favorable and/or window_closing")
    return config


def _read_manifest(snapshot_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{snapshot_dir} has no manifest.json and is not a consumable snapshot")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("series"), list) or not manifest["series"]:
        raise ValueError(f"{manifest_path} has no series")
    return manifest


def resolve_snapshot(snapshot_dir: Path | str | None) -> Path:
    """Use an explicit manifest-gated snapshot or the widest available one."""

    if snapshot_dir is not None:
        resolved = Path(snapshot_dir)
        _read_manifest(resolved)
        return resolved
    root = Path("data/raw/moex_universe")
    candidates: list[tuple[int, str, Path]] = []
    for manifest_path in root.glob("snapshot-*/manifest.json"):
        manifest = _read_manifest(manifest_path.parent)
        try:
            length = (pd.Timestamp(manifest["date_to"]) - pd.Timestamp(manifest["date_from"])).days
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{manifest_path} has invalid date range") from exc
        candidates.append((length, str(manifest.get("created_at", "")), manifest_path.parent))
    if not candidates:
        raise ValueError("No manifest-gated universe snapshot found; run make universe-data first")
    return max(candidates)[2]


def load_universe_prices(snapshot_dir: Path | str, *, target_instrument_id: str) -> pd.DataFrame:
    """Read all fixed series into a target-date-aligned price table.

    Missing factor closes stay missing. In particular, this loader does not
    forward-fill the documented USD/RUB gap or infer prices over market
    holidays.
    """

    directory = Path(snapshot_dir)
    manifest = _read_manifest(directory)
    records: dict[str, pd.Series] = {}
    for item in manifest["series"]:
        if not isinstance(item, dict) or not isinstance(item.get("instrument_id"), str) or not isinstance(item.get("path"), str):
            raise ValueError("Every manifest series needs instrument_id and path")
        path = directory / item["path"]
        raw = pd.read_csv(path)
        required = {"trade_date", "instrument_id", "close"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"{path} lacks: {', '.join(sorted(missing))}")
        if raw["instrument_id"].nunique() != 1 or raw["instrument_id"].iloc[0] != item["instrument_id"]:
            raise ValueError(f"{path} does not match manifest instrument_id {item['instrument_id']}")
        dates = pd.to_datetime(raw["trade_date"], errors="raise").dt.normalize()
        if dates.duplicated().any():
            raise ValueError(f"{path} has duplicate trade_date rows")
        close = pd.to_numeric(raw["close"], errors="coerce").where(lambda value: value.gt(0))
        records[item["instrument_id"]] = pd.Series(close.to_numpy(), index=dates, name=item["instrument_id"]).sort_index()
    if target_instrument_id not in records:
        raise ValueError(f"Snapshot has no target {target_instrument_id}")
    target = records[target_instrument_id].dropna()
    if target.empty:
        raise ValueError(f"Target {target_instrument_id} has no positive closes")
    prices = pd.DataFrame(index=target.index)
    prices.index.name = "trade_date"
    for instrument_id, close in records.items():
        prices[instrument_id] = close.reindex(prices.index)
    return prices


def _target_panel(prices: pd.DataFrame, target_instrument_id: str) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(prices.index, errors="raise"))
    known_at = (dates.dt.tz_localize(MSK) + pd.DateOffset(hours=23, minutes=59)).rename("known_at")
    return pd.DataFrame(
        {
            "known_at": known_at,
            "value_date": dates.dt.date,
            "series_id": f"MOEX:{target_instrument_id}",
            "price": prices[target_instrument_id].to_numpy(),
            "is_carried": False,
            "meta": [{} for _ in range(len(prices))],
        }
    )


def quarterly_folds(panel: pd.DataFrame, *, horizon: int, min_training_observations: int) -> tuple[RuleFold, ...]:
    """Construct expanding, horizon-purged quarters without future labels."""

    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    dates = pd.to_datetime(panel["value_date"], errors="raise")
    quarters = dates.dt.to_period("Q")
    folds: list[RuleFold] = []
    for quarter in sorted(quarters.unique()):
        positions = panel.index[quarters.eq(quarter)].tolist()
        start, end = positions[0], positions[-1]
        train_stop = start - horizon
        if train_stop < min_training_observations:
            continue
        test_positions = tuple(range(start, end - horizon + 1))
        if not test_positions:
            continue
        folds.append(
            RuleFold(
                name=str(quarter),
                train_positions=tuple(range(train_stop)),
                test_positions=test_positions,
                purged_tail_observations=horizon,
            )
        )
    return tuple(folds)


def rule_candidates(prices: pd.DataFrame, *, config: dict[str, Any]) -> pd.DataFrame:
    """Produce the whole preregistered rule library using only information at T.

    Each factor return is shifted by `factor_lag_observations`; the rolling
    quantile is shifted once more, so its comparison distribution does not
    contain the factor observation currently used by a signal.
    """

    evaluation = config["evaluation"]
    library = config["rule_library"]
    target = evaluation["target_instrument_id"]
    lag = int(evaluation["factor_lag_observations"])
    lookback = int(library["comparison_lookback_observations"])
    rows: list[dict[str, object]] = []
    for factor in sorted(column for column in prices.columns if column != target):
        factor_prices = pd.to_numeric(prices[factor], errors="coerce")
        for return_window in library["return_windows"]:
            observed_return = factor_prices.pct_change(int(return_window), fill_method=None).shift(lag)
            history = observed_return.shift(1).rolling(lookback, min_periods=lookback)
            for quantile in library["tail_quantiles"]:
                lower_cutoff = history.quantile(float(quantile))
                upper_cutoff = history.quantile(1 - float(quantile))
                for tail, cutoff in (("lower", lower_cutoff), ("upper", upper_cutoff)):
                    if tail not in library["tails"]:
                        continue
                    candidate = observed_return.le(cutoff) if tail == "lower" else observed_return.ge(cutoff)
                    strength = (cutoff - observed_return) if tail == "lower" else (observed_return - cutoff)
                    for direction in library["directions"]:
                        rule_id = (
                            f"{factor}__return_{return_window}__{tail}_{float(quantile):.2f}__{direction}"
                        )
                        for position in prices.index[candidate.fillna(False)]:
                            rows.append(
                                {
                                    "position": int(prices.index.get_loc(position)),
                                    "timestamp": pd.Timestamp(position) + pd.DateOffset(hours=23, minutes=59),
                                    "rule_id": rule_id,
                                    "factor_instrument_id": factor,
                                    "return_window": int(return_window),
                                    "tail_quantile": float(quantile),
                                    "tail": tail,
                                    "direction": direction,
                                    "strength": float(max(strength.loc[position], 0.0)),
                                }
                            )
    columns = (
        "position",
        "timestamp",
        "rule_id",
        "factor_instrument_id",
        "return_window",
        "tail_quantile",
        "tail",
        "direction",
        "strength",
    )
    return pd.DataFrame(rows, columns=columns)


def _weekly_cap(candidates: pd.DataFrame, panel: pd.DataFrame, *, maximum_signals_per_week: float) -> pd.DataFrame:
    """Keep the first observable candidates in each week; never rank hindsight."""

    if candidates.empty:
        return candidates.copy()
    capped = candidates.sort_values("position", kind="mergesort").copy()
    dates = pd.to_datetime(panel.loc[capped["position"], "value_date"], errors="raise").reset_index(drop=True)
    capped["week"] = dates.dt.to_period("W").astype(str).to_numpy()
    return capped.groupby("week", sort=False).head(int(maximum_signals_per_week)).drop(columns="week")


def _metrics_for_rule(
    candidates: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    base_positions: tuple[int, ...],
    direction: str,
    horizon: int,
    maximum_signals_per_week: float,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    eligible = candidates.loc[candidates["position"].isin(base_positions)]
    capped = _weekly_cap(eligible, panel, maximum_signals_per_week=maximum_signals_per_week)
    base_labels = labels.loc[labels["position"].isin(base_positions)]
    signal = base_labels.loc[base_labels["position"].isin(capped["position"])]
    hit_column = "hit_favorable" if direction == "favorable" else "hit_closing"
    hit_rate = float(signal[hit_column].mean()) if not signal.empty else None
    baseline_hit_rate = float(base_labels[hit_column].mean()) if not base_labels.empty else None
    lift = hit_rate / baseline_hit_rate if hit_rate is not None and baseline_hit_rate not in {None, 0} else None
    base_dates = pd.to_datetime(panel.loc[list(base_positions), "value_date"], errors="raise")
    weeks = max(1, int(base_dates.dt.to_period("W").nunique()))
    metrics = {
        "signal_count": int(len(signal)),
        "eligible_base_count": int(len(base_labels)),
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline_hit_rate,
        "lift": lift,
        "benefit_fwd_bps": float(signal["benefit_fwd_bps"].mean()) if not signal.empty else None,
        "signals_per_week": float(len(signal) / weeks),
    }
    return capped, metrics


def _select_rule(
    candidates: pd.DataFrame,
    *,
    fold: RuleFold,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    evaluation: dict[str, Any],
) -> tuple[dict[str, object] | None, str, pd.DataFrame]:
    """Select one rule on the purged train prefix and record every gate."""

    rows: list[dict[str, object]] = []
    rule_columns = [
        "rule_id",
        "factor_instrument_id",
        "return_window",
        "tail_quantile",
        "tail",
        "direction",
    ]
    for rule_values, rule_candidates_frame in candidates.groupby(rule_columns, sort=True):
        direction = str(rule_values[-1])
        _, metrics = _metrics_for_rule(
            rule_candidates_frame,
            panel=panel,
            labels=labels,
            base_positions=fold.train_positions,
            direction=direction,
            horizon=int(evaluation["horizon_observations"]),
            maximum_signals_per_week=float(evaluation["maximum_signals_per_week"]),
        )
        rows.append({**dict(zip(rule_columns, rule_values, strict=True)), **metrics})
    if not rows:
        return None, "no_candidates", pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    leaderboard = pd.DataFrame(rows)
    leaderboard["meets_minimum_signals"] = leaderboard["signal_count"].ge(
        int(evaluation["minimum_training_signals"])
    )
    leaderboard["frequency_in_policy"] = leaderboard["signals_per_week"].between(
        float(evaluation["minimum_signals_per_week"]), float(evaluation["maximum_signals_per_week"])
    )
    leaderboard["meets_selection_lift"] = leaderboard["lift"].ge(float(evaluation["selection_minimum_lift"]))
    leaderboard["meets_benefit"] = leaderboard["benefit_fwd_bps"].gt(
        float(evaluation["minimum_training_benefit_bps"])
    )
    leaderboard["eligible_for_selection"] = (
        leaderboard["meets_minimum_signals"]
        & leaderboard["frequency_in_policy"]
        & leaderboard["meets_selection_lift"]
        & leaderboard["meets_benefit"]
    )
    leaderboard["fold"] = fold.name
    leaderboard = leaderboard.loc[:, LEADERBOARD_COLUMNS]
    eligible = leaderboard.loc[leaderboard["eligible_for_selection"]].copy()
    if eligible.empty:
        return None, "no_rule_meets_train_gate", leaderboard
    eligible["_lift_rank"] = eligible["lift"].fillna(float("-inf"))
    eligible["_benefit_rank"] = eligible["benefit_fwd_bps"].fillna(float("-inf"))
    selected = eligible.sort_values(
        ["_lift_rank", "_benefit_rank", "signal_count", "rule_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0].to_dict()
    promoted = float(selected["lift"]) >= float(evaluation["promotion_minimum_lift"])
    return selected, "promoted" if promoted else "research_only", leaderboard


def _selection_row(
    *,
    fold: RuleFold,
    panel: pd.DataFrame,
    selected: dict[str, object] | None,
    selection_status: str,
    candidate_rules: int,
    test_metrics: dict[str, float | int | None] | None,
    test_candidate_count: int,
    test_dispatched_count: int,
    test_frequency_in_policy: bool | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "fold": fold.name,
        "test_start": str(panel.loc[fold.test_positions[0], "value_date"]),
        "test_end": str(panel.loc[fold.test_positions[-1], "value_date"]),
        "training_observations": len(fold.train_positions),
        "purged_tail_observations": fold.purged_tail_observations,
        "candidate_rules": candidate_rules,
        "selection_status": selection_status,
        "test_candidate_signal_count": test_candidate_count,
        "test_dispatched_signal_count": test_dispatched_count,
        "test_frequency_in_policy": test_frequency_in_policy,
    }
    if selected is not None:
        for key in (
            "rule_id",
            "factor_instrument_id",
            "return_window",
            "tail_quantile",
            "tail",
            "direction",
            "signal_count",
            "signals_per_week",
            "hit_rate",
            "baseline_hit_rate",
            "lift",
            "benefit_fwd_bps",
        ):
            prefix = {
                "rule_id": "selected_rule_id",
                "signal_count": "train_signal_count",
                "signals_per_week": "train_signals_per_week",
                "hit_rate": "train_hit_rate",
                "baseline_hit_rate": "train_baseline_hit_rate",
                "lift": "train_lift",
                "benefit_fwd_bps": "train_benefit_fwd_bps",
            }.get(key, key)
            row[prefix] = selected[key]
    if test_metrics is not None:
        for key in ("signals_per_week", "hit_rate", "baseline_hit_rate", "lift", "benefit_fwd_bps"):
            row[f"test_{key}"] = test_metrics[key]
    return row


def _summary_rows(
    signals: pd.DataFrame,
    *,
    folds: tuple[RuleFold, ...],
    fold_status: dict[str, str],
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Report quality and clustering for the actual chronological signal flow."""

    if signals.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    fold_by_name = {fold.name: fold for fold in folds}
    date_to_position = {str(value): position for position, value in enumerate(panel["value_date"].astype(str))}
    rows: list[dict[str, object]] = []
    for group_values, group in signals.groupby(["selection_status", "communication_allowed", "direction"], sort=True):
        status, communication_allowed, direction = group_values
        group_folds = sorted(group["fold"].unique())
        base_positions = tuple(position for fold in group_folds for position in fold_by_name[fold].test_positions)
        positions = [date_to_position[value] for value in group["value_date"] if value in date_to_position]
        metrics = evaluate_positions(
            panel,
            positions,
            direction=str(direction),
            horizon=horizon,
            base_positions=base_positions,
            labels=labels,
        )
        timestamps = pd.to_datetime(group["timestamp"], errors="raise")
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_localize(None)
        weeks = timestamps.dt.to_period("W")
        per_fold_week = group.assign(_week=weeks).groupby(["fold", "_week"], sort=False).size()
        rows.append(
            {
                "selection_status": status,
                "communication_allowed": bool(communication_allowed),
                "direction": direction,
                "folds": len(group_folds),
                **metrics,
                "weeks_with_signal": int(weeks.nunique()),
                "max_signals_in_fold_week": int(per_fold_week.max()),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def run_rule_selection(
    *,
    snapshot_dir: Path | str | None = None,
    config_path: Path | str = Path("configs/rule_selection.json"),
    artifact_dir: Path | str = Path("artifacts/rule_selection"),
) -> dict[str, Any]:
    """Select a rule each quarter, test it OOT, and write auditable artifacts."""

    config = load_rule_config(config_path)
    evaluation = config["evaluation"]
    snapshot = resolve_snapshot(snapshot_dir)
    prices = load_universe_prices(snapshot, target_instrument_id=str(evaluation["target_instrument_id"]))
    panel = _target_panel(prices, str(evaluation["target_instrument_id"])).reset_index(drop=True)
    horizon = int(evaluation["horizon_observations"])
    labels = label_observations(panel, horizon)
    folds = quarterly_folds(
        panel,
        horizon=horizon,
        min_training_observations=int(evaluation["min_training_observations"]),
    )
    candidates = rule_candidates(prices, config=config)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    leaderboard_frames: list[pd.DataFrame] = []
    fold_status: dict[str, str] = {}
    for fold in folds:
        selected, status, leaderboard = _select_rule(
            candidates,
            fold=fold,
            panel=panel,
            labels=labels,
            evaluation=evaluation,
        )
        leaderboard_frames.append(leaderboard)
        fold_status[fold.name] = status
        test_metrics: dict[str, float | int | None] | None = None
        test_candidates = pd.DataFrame(columns=candidates.columns)
        dispatched = 0
        frequency_ok: bool | None = None
        if selected is not None:
            selected_candidates = candidates.loc[candidates["rule_id"].eq(selected["rule_id"])].copy()
            test_candidates, test_metrics = _metrics_for_rule(
                selected_candidates,
                panel=panel,
                labels=labels,
                base_positions=fold.test_positions,
                direction=str(selected["direction"]),
                horizon=horizon,
                maximum_signals_per_week=float(evaluation["maximum_signals_per_week"]),
            )
            frequency = test_metrics["signals_per_week"]
            frequency_ok = (
                frequency is not None
                and float(evaluation["minimum_signals_per_week"])
                <= float(frequency)
                <= float(evaluation["maximum_signals_per_week"])
            )
            communication_allowed = status == "promoted"
            dispatched = len(test_candidates) if communication_allowed else 0
            for row in test_candidates.itertuples(index=False):
                signal_rows.append(
                    {
                        "fold": fold.name,
                        "selection_status": status,
                        "timestamp": row.timestamp,
                        "value_date": pd.Timestamp(row.timestamp).date().isoformat(),
                        "rule_id": row.rule_id,
                        "factor_instrument_id": row.factor_instrument_id,
                        "direction": row.direction,
                        "strength": row.strength,
                        "communication_allowed": communication_allowed,
                        "details": json.dumps(
                            {
                                "selection_status": status,
                                "train_lift": selected["lift"],
                                "train_signals_per_week": selected["signals_per_week"],
                                "weekly_cap": evaluation["maximum_signals_per_week"],
                            },
                            sort_keys=True,
                        ),
                    }
                )
        fold_rows.append(
            _selection_row(
                fold=fold,
                panel=panel,
                selected=selected,
                selection_status=status,
                candidate_rules=len(leaderboard),
                test_metrics=test_metrics,
                test_candidate_count=len(test_candidates),
                test_dispatched_count=dispatched,
                test_frequency_in_policy=frequency_ok,
            )
        )
    folds_frame = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    signals_frame = pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS)
    leaderboard_frame = (
        pd.concat(leaderboard_frames, ignore_index=True)
        if leaderboard_frames
        else pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    )
    summary_frame = _summary_rows(
        signals_frame,
        folds=folds,
        fold_status=fold_status,
        panel=panel,
        labels=labels,
        horizon=horizon,
    )
    folds_frame.to_csv(output / "folds.csv", index=False)
    signals_frame.to_csv(output / "signals.csv", index=False)
    leaderboard_frame.to_csv(output / "leaderboard.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    factor_coverage = {
        instrument_id: {
            "available_observations": int(prices[instrument_id].notna().sum()),
            "coverage": float(prices[instrument_id].notna().mean()),
        }
        for instrument_id in prices.columns
        if instrument_id != evaluation["target_instrument_id"]
    }
    meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "snapshot_dir": str(snapshot),
        "config_path": str(config_path),
        "config_sha256": rule_config_sha256(config_path),
        "target_instrument_id": evaluation["target_instrument_id"],
        "horizon_observations": horizon,
        "folds": len(folds),
        "candidate_rules": int(candidates["rule_id"].nunique()),
        "factor_instruments": sorted(factor_coverage),
        "factor_coverage": factor_coverage,
        "signals_written": len(signals_frame),
        "dispatched_signals": int(signals_frame["communication_allowed"].sum()) if not signals_frame.empty else 0,
        "method": "expanding quarterly nested rule selection; h-observation train purge; one-observation factor lag; chronological weekly cap; do-not-send fallback",
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="manifest-gated universe snapshot; widest local snapshot is default")
    parser.add_argument("--config", type=Path, default=Path("configs/rule_selection.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/rule_selection"))
    args = parser.parse_args(argv)
    meta = run_rule_selection(snapshot_dir=args.snapshot, config_path=args.config, artifact_dir=args.artifact_dir)
    print(
        f"Wrote {meta['folds']} folds and {meta['signals_written']} candidate test signals "
        f"({meta['dispatched_signals']} communication-eligible) to {args.artifact_dir}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RuleFold",
    "load_rule_config",
    "load_universe_prices",
    "quarterly_folds",
    "resolve_snapshot",
    "rule_candidates",
    "run_rule_selection",
]
