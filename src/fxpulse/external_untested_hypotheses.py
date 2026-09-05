"""Leakage-aware tests for U10 (futures curve) and U13 (external FX pretraining)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from fxpulse.training_history_experiment import add_training_labels
from fxpulse.untested_hypotheses_experiment import (
    BASELINE,
    _fit_catboost_ensemble,
    _fold_boundaries,
    attach_asof,
    bootstrap_deltas,
    compare_to_baseline,
    select_standard_signals,
    summarize,
)


def load_config(path: Path | str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("external experiment must be preregistered")
    if config.get("corridors") != ["AMD", "KGS", "KZT", "TJS", "UZS"]:
        raise ValueError("the fixed five-corridor universe is required")
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def futures_curve_features(
    futures: pd.DataFrame,
    hourly_spot: pd.DataFrame,
    *,
    minimum_days: int,
    maximum_near_days: int,
    maximum_far_days: int,
) -> pd.DataFrame:
    """Build close-of-session curve features, then expose them one session later."""
    contracts = futures.copy()
    contracts["timestamp"] = pd.to_datetime(contracts["begin"], errors="raise").dt.normalize()
    contracts["expiry"] = pd.to_datetime(contracts["expiry"], errors="raise")
    for column in ("close", "volume"):
        contracts[column] = pd.to_numeric(contracts[column], errors="coerce")
    contracts["days_to_expiry"] = (contracts["expiry"] - contracts["timestamp"]).dt.days

    spot = hourly_spot.loc[hourly_spot["secid"].eq("CNYRUB_TOM")].copy()
    spot["dt_msk"] = pd.to_datetime(spot["dt_msk"], errors="raise")
    spot["timestamp"] = spot["dt_msk"].dt.normalize()
    spot = (
        spot.sort_values("dt_msk", kind="mergesort")
        .groupby("timestamp", as_index=False)
        .tail(1)[["timestamp", "close"]]
        .rename(columns={"close": "spot_close"})
    )
    spot["spot_close"] = pd.to_numeric(spot["spot_close"], errors="coerce")
    contracts = contracts.merge(spot, on="timestamp", how="left", validate="many_to_one")
    contracts = contracts.loc[
        contracts["close"].gt(0)
        & contracts["spot_close"].gt(0)
        & contracts["days_to_expiry"].ge(minimum_days)
    ].copy()
    contracts["basis"] = contracts["close"] / contracts["spot_close"] - 1

    rows: list[dict[str, object]] = []
    for timestamp, day in contracts.groupby("timestamp", sort=True):
        near_pool = day.loc[day["days_to_expiry"].le(maximum_near_days)]
        if near_pool.empty:
            continue
        # The completed session's volume is known when these features are used
        # on the next day.  Choosing the liquid contract avoids stale quotes.
        near = near_pool.sort_values(["volume", "expiry"], ascending=[False, True]).iloc[0]
        far_pool = day.loc[
            day["expiry"].gt(near["expiry"])
            & day["days_to_expiry"].le(maximum_far_days)
        ]
        far = (
            far_pool.sort_values(["volume", "expiry"], ascending=[False, True]).iloc[0]
            if not far_pool.empty
            else None
        )
        near_days = float(near["days_to_expiry"])
        row: dict[str, object] = {
            "timestamp": pd.Timestamp(timestamp),
            "futures__near_basis": float(near["basis"]),
            "futures__near_basis_annualized": float(near["basis"]) * 365 / near_days,
            "futures__near_days_to_expiry": near_days,
            "futures__near_volume_log": float(np.log1p(near["volume"])),
        }
        if far is not None:
            far_days = float(far["days_to_expiry"])
            expiry_gap = max(far_days - near_days, 1.0)
            row.update(
                {
                    "futures__far_basis": float(far["basis"]),
                    "futures__curve_slope_annualized": (
                        float(far["basis"]) - float(near["basis"])
                    ) * 365 / expiry_gap,
                    "futures__far_days_to_expiry": far_days,
                }
            )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("timestamp", kind="mergesort")
    basis = result["futures__near_basis"]
    result["futures__near_basis_change_1"] = basis.diff()
    result["futures__near_basis_change_5"] = basis.diff(5)
    rolling_mean = basis.rolling(20, min_periods=20).mean()
    rolling_std = basis.rolling(20, min_periods=20).std().replace(0, np.nan)
    result["futures__near_basis_zscore_20"] = (basis - rolling_mean) / rolling_std
    feature_columns = [column for column in result if column.startswith("futures__")]
    result[feature_columns] = result[feature_columns].shift(1)
    return result.replace([np.inf, -np.inf], np.nan)


def _add_evaluation_fields(usable: pd.DataFrame, *, horizon: int, tolerance_bps: float) -> pd.DataFrame:
    outcome = f"outcome__regret_{horizon}"
    benefit = f"outcome__benefit_{horizon}"
    data = usable.copy()
    data["target"] = data[outcome].le(tolerance_bps).astype(int)
    data["regret_bps"] = data[outcome]
    data["benefit_bps"] = data[benefit]
    data["week"] = data["timestamp"].dt.to_period("W").astype(str)
    weekly = data.groupby("week", sort=False)["target"].mean()
    data["matched_week_hit_rate"] = data["week"].map(weekly).astype(float)
    return data


def _purged_train(usable: pd.DataFrame, start: pd.Timestamp, horizon: int, years: int) -> pd.DataFrame:
    past = usable.loc[usable["timestamp"].lt(start)].sort_values("timestamp", kind="mergesort")
    past = past.iloc[:-horizon] if len(past) > horizon else past.iloc[0:0]
    return past.loc[past["timestamp"].ge(start - pd.DateOffset(years=years))].copy()


def build_u10_scores(frame: pd.DataFrame, curve: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config["u10"]
    data = attach_asof(frame, curve, carry_days=int(settings["alignment_carry_days"]))
    base_columns = [column for column in data if column.startswith(tuple(config["feature_prefixes"]))]
    futures_columns = [column for column in data if column.startswith("futures__")]
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for corridor in config["corridors"]:
        corridor_data = data.loc[data["corridor"].eq(corridor)].copy()
        for horizon_value in config["horizons"]:
            horizon = int(horizon_value)
            outcome = f"outcome__regret_{horizon}"
            label = f"training_target_{horizon}"
            usable = corridor_data.loc[corridor_data[outcome].notna() & corridor_data[label].notna()].copy()
            usable = _add_evaluation_fields(
                usable, horizon=horizon, tolerance_bps=float(config["evaluation_tolerance_bps"])
            )
            for fold_index, (start, end, fold_name) in enumerate(_fold_boundaries(config)):
                train = _purged_train(
                    usable, start, horizon, int(config["rolling_training_years"])
                )
                test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                complete_train = int(train["futures__near_basis"].notna().sum())
                if (
                    len(train) < int(config["minimum_training_observations"])
                    or complete_train < int(settings["minimum_complete_training_observations"])
                    or len(test) < 20
                    or train[label].nunique() < 2
                ):
                    continue
                score = _fit_catboost_ensemble(
                    train,
                    test,
                    [*base_columns, *futures_columns],
                    label,
                    config=config,
                    seed_offset=10_000 + 100 * fold_index + 10 * horizon + list(config["corridors"]).index(corridor),
                )
                export = test[
                    ["timestamp", "corridor", "week", "target", "regret_bps", "benefit_bps", "matched_week_hit_rate"]
                ].copy()
                export["variant"] = "u10_cny_futures_curve"
                export["horizon"] = horizon
                export["test_fold"] = fold_name
                export["score"] = score
                parts.append(export)
                diagnostics.append(
                    {
                        "corridor": corridor,
                        "horizon": horizon,
                        "test_fold": fold_name,
                        "train_rows": len(train),
                        "train_futures_rows": complete_train,
                        "test_rows": len(test),
                        "test_futures_share": float(test["futures__near_basis"].notna().mean()),
                    }
                )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(diagnostics)


def _shape_rows(frame: pd.DataFrame, *, group_column: str, date_column: str, price_column: str, window: int) -> pd.DataFrame:
    pieces = []
    for name, group in frame.groupby(group_column, sort=True):
        group = group.sort_values(date_column, kind="mergesort").copy()
        returns = np.log(pd.to_numeric(group[price_column], errors="coerce")).diff()
        lags = pd.concat([returns.shift(step) for step in range(window)], axis=1)
        lags.columns = [f"shape_{step}" for step in range(window)]
        mean = lags.mean(axis=1)
        scale = lags.std(axis=1).replace(0, np.nan)
        normalized = lags.sub(mean, axis=0).div(scale, axis=0)
        normalized[date_column] = group[date_column].to_numpy()
        normalized[group_column] = name
        if "_row_id" in group:
            normalized["_row_id"] = group["_row_id"].to_numpy()
        pieces.append(normalized)
    return pd.concat(pieces, ignore_index=True)


def _pca_coordinates(pca: PCA, shapes: pd.DataFrame, shape_columns: list[str], prefix: str) -> pd.DataFrame:
    result = pd.DataFrame({"_row_id": shapes["_row_id"]})
    matrix = shapes[shape_columns].to_numpy(dtype=float)
    valid = np.isfinite(matrix).all(axis=1)
    coordinates = np.full((len(shapes), pca.n_components_), np.nan)
    coordinates[valid] = pca.transform(pd.DataFrame(matrix[valid], columns=shape_columns))
    for index in range(pca.n_components_):
        result[f"{prefix}{index + 1}"] = coordinates[:, index]
    return result


def build_u13_scores(frame: pd.DataFrame, donors: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config["u13"]
    data = frame.copy().reset_index(drop=True)
    data["_row_id"] = np.arange(len(data))
    local_shapes = _shape_rows(
        data,
        group_column="corridor",
        date_column="timestamp",
        price_column="price",
        window=int(settings["shape_window"]),
    )
    donor_data = donors.copy()
    donor_data["date"] = pd.to_datetime(donor_data["date"], errors="raise")
    donor_shapes = _shape_rows(
        donor_data,
        group_column="area",
        date_column="date",
        price_column="usd_local",
        window=int(settings["shape_window"]),
    )
    shape_columns = [column for column in local_shapes if column.startswith("shape_")]
    base_columns = [column for column in data if column.startswith(tuple(config["feature_prefixes"]))]
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []

    for fold_index, (start, end, fold_name) in enumerate(_fold_boundaries(config)):
        lower = start - pd.DateOffset(years=int(config["rolling_training_years"]))
        local_fit = local_shapes.loc[
            local_shapes["timestamp"].between(lower, start, inclusive="left")
        ].dropna(subset=shape_columns)
        donor_cutoff = start - pd.offsets.Day(int(settings["donor_publication_lag_days"]))
        donor_fit = donor_shapes.loc[
            donor_shapes["date"].between(lower, donor_cutoff, inclusive="left")
        ].dropna(subset=shape_columns)
        donor_fit = donor_fit.loc[
            donor_fit.groupby("area", sort=False).cumcount().mod(int(settings["donor_subsample_step"])).eq(0)
        ]
        components = int(settings["pca_components"])
        if len(local_fit) < 10 * components or len(donor_fit) < 10 * components:
            continue
        local_pca = PCA(n_components=components, random_state=int(config["random_seed"]))
        donor_pca = PCA(n_components=components, random_state=int(config["random_seed"]))
        local_pca.fit(local_fit[shape_columns])
        donor_pca.fit(donor_fit[shape_columns])
        local_coordinates = _pca_coordinates(local_pca, local_shapes, shape_columns, "u13_local__pc")
        donor_coordinates = _pca_coordinates(donor_pca, local_shapes, shape_columns, "u13_donor__pc")
        fold_data = data.merge(local_coordinates, on="_row_id", how="left", validate="one_to_one")
        fold_data = fold_data.merge(donor_coordinates, on="_row_id", how="left", validate="one_to_one")

        for corridor in config["corridors"]:
            corridor_data = fold_data.loc[fold_data["corridor"].eq(corridor)].copy()
            for horizon_value in config["horizons"]:
                horizon = int(horizon_value)
                outcome = f"outcome__regret_{horizon}"
                label = f"training_target_{horizon}"
                usable = corridor_data.loc[corridor_data[outcome].notna() & corridor_data[label].notna()].copy()
                usable = _add_evaluation_fields(
                    usable, horizon=horizon, tolerance_bps=float(config["evaluation_tolerance_bps"])
                )
                train = _purged_train(
                    usable, start, horizon, int(config["rolling_training_years"])
                )
                test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                if (
                    len(train) < int(config["minimum_training_observations"])
                    or len(test) < 20
                    or train[label].nunique() < 2
                ):
                    continue
                for variant, prefix in (
                    ("u13_local_pca_control", "u13_local__"),
                    ("u13_bis_pretrained_pca", "u13_donor__"),
                ):
                    extra = [column for column in fold_data if column.startswith(prefix)]
                    score = _fit_catboost_ensemble(
                        train,
                        test,
                        [*base_columns, *extra],
                        label,
                        config=config,
                        seed_offset=20_000 + 100 * fold_index + 10 * horizon + list(config["corridors"]).index(corridor),
                    )
                    export = test[
                        ["timestamp", "corridor", "week", "target", "regret_bps", "benefit_bps", "matched_week_hit_rate"]
                    ].copy()
                    export["variant"] = variant
                    export["horizon"] = horizon
                    export["test_fold"] = fold_name
                    export["score"] = score
                    parts.append(export)
        diagnostics.append(
            {
                "test_fold": fold_name,
                "local_pretraining_rows": len(local_fit),
                "donor_pretraining_rows": len(donor_fit),
                "donor_cutoff": donor_cutoff,
                "local_explained_variance": float(local_pca.explained_variance_ratio_.sum()),
                "donor_explained_variance": float(donor_pca.explained_variance_ratio_.sum()),
            }
        )
        print(f"completed U13 {fold_name}", flush=True)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(diagnostics)


def evaluate(
    candidates: pd.DataFrame,
    baseline_scores: pd.DataFrame,
    *,
    config: dict[str, Any],
    output: Path,
    diagnostics: pd.DataFrame,
) -> dict[str, object]:
    key_columns = ["timestamp", "corridor", "horizon"]
    variants = sorted(candidates["variant"].unique())
    common = None
    for variant in variants:
        keys = candidates.loc[candidates["variant"].eq(variant), key_columns].drop_duplicates()
        common = keys if common is None else common.merge(keys, on=key_columns, how="inner")
    assert common is not None
    candidates = candidates.merge(common, on=key_columns, how="inner", validate="many_to_one")
    baseline = baseline_scores.loc[baseline_scores["variant"].eq(BASELINE)].copy()
    baseline = baseline.merge(common, on=key_columns, how="inner", validate="one_to_one")
    scores = pd.concat([baseline, candidates], ignore_index=True, sort=False)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], errors="raise")
    signals = select_standard_signals(scores, config)
    corridor, aggregate, detail = summarize(scores, signals)
    comparison, gates = compare_to_baseline(corridor, aggregate, detail, config)
    bootstrap = bootstrap_deltas(signals, config=config)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "scores.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    corridor.to_csv(output / "corridor_summary.csv", index=False)
    aggregate.to_csv(output / "aggregate_summary.csv", index=False)
    detail.to_csv(output / "fold_and_year_summary.csv", index=False)
    comparison.to_csv(output / "paired_comparison.csv", index=False)
    gates.to_csv(output / "success_gates.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_h5.csv", index=False)
    diagnostics.to_csv(output / "data_diagnostics.csv", index=False)
    return {
        "variants": variants,
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "first_test": scores["timestamp"].min().date().isoformat(),
        "last_test": scores["timestamp"].max().date().isoformat(),
    }


def run(*, config_path: Path | str, artifact_dir: Path | str) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_config(config_path)
    frame = pd.read_csv(config["input"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = add_training_labels(frame, config).sort_values(["corridor", "timestamp"], kind="mergesort")
    baseline_scores = pd.read_csv(config["baseline_scores"])
    baseline_scores["timestamp"] = pd.to_datetime(baseline_scores["timestamp"], errors="raise")
    output = Path(artifact_dir)

    curve = futures_curve_features(
        pd.read_csv(config["futures_input"]),
        pd.read_csv(config["hourly_input"]),
        minimum_days=int(config["u10"]["minimum_days_to_expiry"]),
        maximum_near_days=int(config["u10"]["maximum_near_days_to_expiry"]),
        maximum_far_days=int(config["u10"]["maximum_far_days_to_expiry"]),
    )
    curve.to_csv("data/processed/moex_cny_futures_curve_features.csv", index=False)
    u10_scores, u10_diagnostics = build_u10_scores(frame, curve, config)
    u10_meta = evaluate(
        u10_scores,
        baseline_scores,
        config=config,
        output=output / "u10_futures_curve",
        diagnostics=u10_diagnostics,
    )

    u13_scores, u13_diagnostics = build_u13_scores(
        frame, pd.read_csv(config["bis_donors_input"]), config
    )
    u13_meta = evaluate(
        u13_scores,
        baseline_scores,
        config=config,
        output=output / "u13_external_fx_pretraining",
        diagnostics=u13_diagnostics,
    )
    meta = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "input_sha256": _sha256(Path(config["input"])),
        "futures_input_sha256": _sha256(Path(config["futures_input"])),
        "bis_donors_input_sha256": _sha256(Path(config["bis_donors_input"])),
        "u10": u10_meta,
        "u13": u13_meta,
        "warning": "Exploratory on reviewed history; promotion requires genuinely untouched future data.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_untested_hypotheses_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/untested_hypotheses/external"))
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
