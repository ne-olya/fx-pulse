"""Test public-holiday features and cache the generated historical calendar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import _add_corridor_dummies, _make_model, _purged_train
from fxpulse.temporal_sequence_experiment import _evaluate_fold, summarize


BASIC_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")


def load_config(path: Path | str = Path("configs/holiday_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("holiday config must be preregistered")
    if config.get("feature_sets") != ["basic_calendar", "calendar_plus_holidays"]:
        raise ValueError("holiday feature sets differ from the implementation")
    return config


def build_holiday_table(years: list[int], country_codes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code in country_codes:
        calendar = holidays.country_holidays(code, years=years, observed=True)
        rows.extend(
            {
                "date": pd.Timestamp(date),
                "country_code": code,
                "name": name,
                "source": "python-holidays",
                "source_version": holidays.__version__,
            }
            for date, name in sorted(calendar.items())
        )
    return pd.DataFrame(rows).sort_values(["country_code", "date"], kind="mergesort").reset_index(drop=True)


def _holiday_distances(dates: pd.Series, holiday_dates: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = dates.to_numpy(dtype="datetime64[D]")
    known = np.sort(holiday_dates.to_numpy(dtype="datetime64[D]"))
    right = np.searchsorted(known, current, side="left")
    next_index = np.minimum(right, len(known) - 1)
    previous_index = np.maximum(np.searchsorted(known, current, side="right") - 1, 0)
    until = (known[next_index] - current).astype("timedelta64[D]").astype(int)
    since = (current - known[previous_index]).astype("timedelta64[D]").astype(int)
    today = np.isin(current, known)
    return today, until, since


def add_holiday_features(frame: pd.DataFrame, table: pd.DataFrame, country_codes: dict[str, str]) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    table = table.copy()
    table["date"] = pd.to_datetime(table["date"], errors="raise")
    ru = table.loc[table["country_code"].eq("RU"), "date"]
    ru_today, ru_until, ru_since = _holiday_distances(data["timestamp"], ru)
    data["holiday__ru_today"] = ru_today.astype(float)
    data["holiday__ru_in_next_3_days"] = ((ru_until >= 1) & (ru_until <= 3)).astype(float)
    data["holiday__ru_in_previous_3_days"] = ((ru_since >= 1) & (ru_since <= 3)).astype(float)
    recipient_today = np.zeros(len(data), dtype=bool)
    recipient_until = np.full(len(data), 999, dtype=int)
    recipient_since = np.full(len(data), 999, dtype=int)
    for corridor, code in country_codes.items():
        mask = data["corridor"].eq(corridor).to_numpy()
        country_dates = table.loc[table["country_code"].eq(code), "date"]
        today, until, since = _holiday_distances(data.loc[mask, "timestamp"], country_dates)
        recipient_today[mask] = today
        recipient_until[mask] = until
        recipient_since[mask] = since
    data["holiday__recipient_today"] = recipient_today.astype(float)
    data["holiday__recipient_in_next_3_days"] = (
        (recipient_until >= 1) & (recipient_until <= 3)
    ).astype(float)
    data["holiday__recipient_in_previous_3_days"] = (
        (recipient_since >= 1) & (recipient_since <= 3)
    ).astype(float)
    data["holiday__today_mismatch"] = np.logical_xor(ru_today, recipient_today).astype(float)
    return data


def evaluate(data: pd.DataFrame, holiday_table: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_holiday_features(data, holiday_table, config["country_codes"])
    basic_columns = [column for column in frame if column.startswith(BASIC_PREFIXES)]
    holiday_columns = [column for column in frame if column.startswith("holiday__")]
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
                    columns = list(basic_columns)
                    if feature_set == "calendar_plus_holidays":
                        columns += holiday_columns
                    for year in config["test_years"]:
                        start = pd.Timestamp(year=int(year), month=1, day=1)
                        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                        train = _purged_train(usable, start, int(horizon))
                        test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                        model_columns = list(columns)
                        if scope == "pooled":
                            train, test, model_columns = _add_corridor_dummies(
                                train, test, model_columns, list(config["corridors"])
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
                        model.fit(train[model_columns], train["target"])
                        test["score"] = model.predict_proba(test[model_columns])[:, 1]
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
    config_path: Path | str = Path("configs/holiday_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/holidays"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    country_codes = ["RU", *sorted(set(config["country_codes"].values()))]
    table = build_holiday_table([int(year) for year in config["years"]], country_codes)
    cache_path = Path(config["holiday_cache"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, index=False)
    folds, signals = evaluate(pd.read_csv(input_path), table, config)
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
        "holiday_cache": str(cache_path),
        "holiday_rows": len(table),
        "holiday_library_version": holidays.__version__,
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory calendar generated by python-holidays; production use requires official calendar validation.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/holiday_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/holidays"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
