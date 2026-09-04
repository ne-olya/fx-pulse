"""Let explainable rules create candidates and ML only confirm them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import _add_corridor_dummies, _apply_policy, _make_model, _purged_train


IDENTITY_KEYS = ("scope", "model", "rule", "horizon")
FEATURE_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")
VOTE_COLUMNS = (
    "indicator__cheap_60",
    "indicator__near_min_20",
    "indicator__rebound_from_low",
    "indicator__rub_strength_3",
    "indicator__recipient_weakness_3",
    "indicator__volatility_falling",
    "indicator__value_and_calm",
)


def load_config(path: Path | str = Path("configs/meta_labeling_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("meta-labeling config must be preregistered schema_version 1")
    expected = {"all_days", "cheap_or_near_min", "indicator_vote_2"}
    if {rule["name"] for rule in config.get("rules", [])} != expected:
        raise ValueError("meta-labeling rules differ from the implementation")
    return config


def candidate_mask(frame: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "all_days":
        return pd.Series(True, index=frame.index)
    if rule == "cheap_or_near_min":
        return frame["indicator__cheap_60"].eq(1) | frame["indicator__near_min_20"].eq(1)
    if rule == "indicator_vote_2":
        return frame[list(VOTE_COLUMNS)].fillna(0).sum(axis=1).ge(2)
    raise ValueError(f"unknown candidate rule {rule}")


def _dispatch(test: pd.DataFrame, *, share: float, policy: dict[str, Any]) -> pd.DataFrame:
    candidate_parts = [
        adaptive_candidates(
            group,
            share=share,
            lookback=int(policy["lookback_observations"]),
            minimum_history=int(policy["minimum_history_observations"]),
        )
        for _, group in test.groupby("corridor", sort=False)
    ]
    candidates = pd.concat(candidate_parts, ignore_index=False) if candidate_parts else test.iloc[0:0]
    selected_parts = [
        _apply_policy(
            group,
            cooldown_days=int(policy["cooldown_days"]),
            weekly_cap=int(policy["weekly_cap"]),
        )
        for _, group in candidates.groupby("corridor", sort=False)
    ]
    return pd.concat(selected_parts, ignore_index=False) if selected_parts else candidates.iloc[0:0]


def evaluate(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    feature_columns = [column for column in frame if column.startswith(FEATURE_PREFIXES)]
    policy = config["adaptive_policy"]
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
                for rule_config in config["rules"]:
                    rule = str(rule_config["name"])
                    eligible = usable.loc[candidate_mask(usable, rule)].copy()
                    for year in config["test_years"]:
                        start = pd.Timestamp(year=int(year), month=1, day=1)
                        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                        purged = _purged_train(usable, start, int(horizon))
                        train = purged.loc[candidate_mask(purged, rule)].copy()
                        test_all = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                        test = eligible.loc[eligible["timestamp"].between(start, end, inclusive="left")].copy()
                        columns = list(feature_columns)
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
                        test_all["week"] = test_all["timestamp"].dt.to_period("W").astype(str)
                        weekly_rate = test_all.groupby(["corridor", "week"])["target"].mean()
                        test["matched_week_hit_rate"] = [
                            float(weekly_rate.loc[(corridor, week)])
                            for corridor, week in zip(test["corridor"], test["week"], strict=True)
                        ]
                        selected = _dispatch(
                            test,
                            share=float(rule_config["top_score_share"]),
                            policy=policy,
                        )
                        count = len(selected)
                        base_rate = float(test_all["target"].mean())
                        candidate_rate = float(test["target"].mean())
                        hit_rate = float(selected["target"].mean()) if count else np.nan
                        exposure = int(test_all["corridor"].nunique())
                        duration = max(float((test_all["timestamp"].max() - test_all["timestamp"].min()).days) / 7, 1 / 7) * exposure
                        identity = {
                            "scope": scope,
                            "model": str(config["model"]),
                            "rule": rule,
                            "horizon": int(horizon),
                        }
                        fold_rows.append(
                            {
                                **identity,
                                "test_year": int(year),
                                "test_count": len(test_all),
                                "test_hits": int(test_all["target"].sum()),
                                "candidate_count": len(test),
                                "candidate_hits": int(test["target"].sum()),
                                "signal_count": count,
                                "signal_hits": int(selected["target"].sum()) if count else 0,
                                "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                                "candidate_lift": candidate_rate / base_rate if base_rate > 0 else np.nan,
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


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(list(IDENTITY_KEYS), sort=True):
        identity = dict(zip(IDENTITY_KEYS, keys, strict=True))
        count = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        candidate_count = int(group["candidate_count"].sum())
        candidate_hits = int(group["candidate_hits"].sum())
        hit_rate = hits / count if count else np.nan
        base_rate = base_hits / base_count if base_count else np.nan
        candidate_rate = candidate_hits / candidate_count if candidate_count else np.nan
        matched_rate = float(group["matched_expected_hits"].sum()) / count if count else np.nan
        valid_lift = group["lift"].dropna()
        rows.append(
            {
                **identity,
                "signals": count,
                "candidates": candidate_count,
                "candidate_coverage": candidate_count / base_count if base_count else np.nan,
                "candidate_lift": candidate_rate / base_rate if base_rate > 0 else np.nan,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_random_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                "lift_vs_matched_random": hit_rate / matched_rate if count and matched_rate > 0 else np.nan,
                "regret_mean_bps": float(group["regret_sum_bps"].sum()) / count if count else np.nan,
                "benefit_mean_bps": float(group["benefit_sum_bps"].sum()) / count if count else np.nan,
                "signals_per_week": count / float(group["duration_weeks"].sum()),
                "worst_fold_lift": float(valid_lift.min()) if len(valid_lift) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    config_path: Path | str = Path("configs/meta_labeling_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/meta_labeling"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    folds, signals = evaluate(pd.read_csv(input_path), config)
    summary = summarize(folds)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory meta-label comparison; requires untouched confirmation.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/meta_labeling_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/meta_labeling"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
