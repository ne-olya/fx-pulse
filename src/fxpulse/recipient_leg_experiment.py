"""Ablation of independently sourced USD/local recipient-currency legs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import _apply_policy, _feature_importance, _make_model


IDENTITY_KEYS = ("corridor", "model", "feature_set", "horizon")


def load_config(path: Path | str = Path("configs/recipient_leg_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("recipient leg config must be preregistered schema_version 1")
    return config


def build_independent_features(
    feature_frame: pd.DataFrame,
    recipient_rates: pd.DataFrame,
    *,
    corridor: str,
    bank: str,
    lag: int,
) -> pd.DataFrame:
    base = feature_frame.loc[feature_frame["corridor"].eq(corridor)].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="raise")
    rates = recipient_rates.loc[
        recipient_rates["bank"].eq(bank) & recipient_rates["local_ccy"].eq(corridor)
    ].copy()
    rates["timestamp"] = pd.to_datetime(rates["requested_date"], errors="raise")
    rates["per_unit"] = pd.to_numeric(rates["local_per_nominal"], errors="raise") / pd.to_numeric(
        rates["nominal"], errors="raise"
    )
    pivot = rates.pivot(index="timestamp", columns="quote_ccy", values="per_unit").sort_index()
    if not {"USD", "RUB"}.issubset(pivot.columns):
        raise ValueError(f"{bank} does not contain independent USD and RUB legs")
    independent = pd.DataFrame(index=pivot.index)
    independent["independent__local_per_usd"] = pivot["USD"]
    independent["independent__local_per_rub"] = pivot["RUB"]
    independent["independent__cross_rub_per_local"] = 1 / pivot["RUB"]
    independent["independent__implied_usd_rub"] = pivot["USD"] / pivot["RUB"]
    for name, price in (
        ("usd_leg", pivot["USD"]),
        ("rub_cross", independent["independent__cross_rub_per_local"]),
        ("implied_usd_rub", independent["independent__implied_usd_rub"]),
    ):
        ret1 = price.pct_change(fill_method=None)
        for window in (1, 3, 5, 10):
            independent[f"independent__{name}_return_{window}"] = price.pct_change(window, fill_method=None)
        independent[f"independent__{name}_volatility_20"] = ret1.rolling(20, min_periods=20).std()
    carried = rates.groupby("timestamp", sort=False)["is_carried"].max().astype(float)
    independent["independent__carried"] = carried
    independent = independent.shift(lag)
    merged = base.merge(independent, left_on="timestamp", right_index=True, how="left", validate="one_to_one")
    merged["independent__cross_gap_bps"] = (
        merged["independent__cross_rub_per_local"] / merged["price"] - 1
    ) * 10_000
    merged["independent__missing"] = merged["independent__local_per_usd"].isna().astype(float)
    return merged.replace([np.inf, -np.inf], np.nan)


def _purged_train(frame: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
    past = frame.loc[frame["timestamp"] < start].copy()
    return past.iloc[:-horizon] if len(past) > horizon else past.iloc[0:0]


def evaluate(datasets: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy = config["adaptive_policy"]
    folds: list[dict[str, object]] = []
    signals: list[pd.DataFrame] = []
    importances: list[dict[str, object]] = []
    for corridor, settings in config["corridors"].items():
        source = datasets[corridor]
        existing = [
            column
            for column in source
            if column.startswith(("base__", "leg__", "regime__", "market__", "indicator__"))
        ]
        independent = [column for column in source if column.startswith("independent__")]
        for feature_set in config["feature_sets"]:
            columns = existing if feature_set == "existing" else [*existing, *independent]
            for model_name in config["models"]:
                for horizon in config["horizons"]:
                    outcome = f"outcome__regret_{horizon}"
                    benefit = f"outcome__benefit_{horizon}"
                    usable = source.loc[source[outcome].notna()].copy()
                    usable["target"] = usable[outcome].le(float(config["tolerance_bps"])).astype(int)
                    usable["regret_bps"] = usable[outcome]
                    usable["benefit_bps"] = usable[benefit]
                    for year in settings["test_years"]:
                        start = pd.Timestamp(year=int(year), month=1, day=1)
                        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                        train = _purged_train(usable, start, int(horizon))
                        test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                        available_leg = int(train["independent__local_per_usd"].notna().sum())
                        if (
                            len(train) < int(config["minimum_training_observations"])
                            or available_leg < int(config["minimum_training_observations"])
                            or len(test) < 20
                            or train["target"].nunique() < 2
                        ):
                            continue
                        model = _make_model(
                            model_name,
                            iterations=int(config["model_iterations"]),
                            seed=int(config["random_seed"]) + int(year),
                        )
                        model.fit(train[columns], train["target"])
                        test["score"] = model.predict_proba(test[columns])[:, 1]
                        candidates = adaptive_candidates(
                            test,
                            share=float(policy["top_score_share"]),
                            lookback=int(policy["lookback_observations"]),
                            minimum_history=int(policy["minimum_history_observations"]),
                        )
                        selected = _apply_policy(
                            candidates,
                            cooldown_days=int(policy["cooldown_days"]),
                            weekly_cap=int(policy["weekly_cap"]),
                        )
                        test["week"] = test["timestamp"].dt.to_period("W").astype(str)
                        selected["week"] = selected["timestamp"].dt.to_period("W").astype(str)
                        weekly_rate = test.groupby("week")["target"].mean()
                        selected["matched_week_hit_rate"] = selected["week"].map(weekly_rate)
                        count = len(selected)
                        base_rate = float(test["target"].mean())
                        hit_rate = float(selected["target"].mean()) if count else np.nan
                        duration = max(float((test["timestamp"].max() - test["timestamp"].min()).days) / 7, 1 / 7)
                        identity = {
                            "corridor": corridor,
                            "model": model_name,
                            "feature_set": feature_set,
                            "horizon": int(horizon),
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
                                "matched_expected_hits": float(selected["matched_week_hit_rate"].sum()) if count else 0.0,
                                "regret_sum_bps": float(selected["regret_bps"].sum()) if count else 0.0,
                                "benefit_sum_bps": float(selected["benefit_bps"].sum()) if count else 0.0,
                                "duration_weeks": duration,
                                "independent_train_observations": available_leg,
                            }
                        )
                        if count:
                            exported = selected[
                                [
                                    "timestamp",
                                    "target",
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
                            signals.append(exported)
                        for feature, importance, signed in _feature_importance(model, columns):
                            importances.append(
                                {
                                    **identity,
                                    "test_year": int(year),
                                    "feature": feature,
                                    "importance": importance,
                                    "signed_effect": signed,
                                }
                            )
    return (
        pd.DataFrame(folds),
        pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(),
        pd.DataFrame(importances),
    )


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
        duration = float(group["duration_weeks"].sum())
        fold_lifts = group["lift"].dropna()
        rows.append(
            {
                **identity,
                "folds": len(group),
                "signals": count,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_random_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                "lift_vs_matched_random": hit_rate / matched_rate if count and matched_rate > 0 else np.nan,
                "regret_mean_bps": float(group["regret_sum_bps"].sum()) / count if count else np.nan,
                "benefit_mean_bps": float(group["benefit_sum_bps"].sum()) / count if count else np.nan,
                "signals_per_week": count / duration if duration else 0.0,
                "worst_fold_lift": float(fold_lifts.min()) if len(fold_lifts) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    config_path: Path | str = Path("configs/recipient_leg_experiment.json"),
    feature_path: Path | str = Path("data/processed/cbr_corridor_features.csv"),
    rate_path: Path | str = Path("data/raw/recipient_bank_daily.csv"),
    processed_path: Path | str = Path("data/processed/recipient_leg_features.csv"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/recipient_legs"),
) -> dict[str, object]:
    config = load_config(config_path)
    features = pd.read_csv(feature_path)
    rates = pd.read_csv(rate_path)
    datasets = {
        corridor: build_independent_features(
            features,
            rates,
            corridor=corridor,
            bank=settings["bank"],
            lag=int(config["recipient_rate_lag_observations"]),
        )
        for corridor, settings in config["corridors"].items()
    }
    processed = Path(processed_path)
    processed.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(datasets.values(), ignore_index=True).to_csv(processed, index=False)
    folds, signals, importance = evaluate(datasets, config)
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
        "rate_sha256": hashlib.sha256(Path(rate_path).read_bytes()).hexdigest(),
        "processed_path": str(processed),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory ablation; national official rates are not executable client quotes.",
    }
    (artifacts / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/recipient_leg_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/recipient_legs"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
