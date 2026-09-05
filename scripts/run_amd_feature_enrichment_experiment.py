"""Evaluate causal market-feature ablations around the frozen AMD h=5 design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fxpulse.adaptive_threshold import adaptive_candidates  # noqa: E402
from fxpulse.feature_enrichment import FEATURE_GROUP_PREFIXES, build_enriched_features  # noqa: E402
from fxpulse.innovation_followup import causal_percentile  # noqa: E402
from fxpulse.next_hypotheses import _apply_policy, _make_model, _purged_train  # noqa: E402
from fxpulse.robust_innovation_experiment import add_registered_labels, load_config as load_robust_config  # noqa: E402


EXPECTED_FEATURE_SETS = {
    "baseline": [],
    "plus_liquidity": ["liquidity"],
    "plus_cross_currency": ["cross_currency"],
    "plus_rates": ["rates"],
    "plus_commodities": ["commodities"],
    "plus_equities": ["equities"],
    "plus_regimes": ["regimes"],
    "plus_events": ["events"],
    "all_enriched": ["liquidity", "cross_currency", "rates", "commodities", "equities", "regimes", "events"],
}
EXPECTED_CONSENSUS = ["official_mean", "agreement_weighted", "agreement_gate"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path, help="Frozen base h=5 feature panel.")
    parser.add_argument(
        "--moex-source",
        action="append",
        nargs=2,
        metavar=("SNAPSHOT", "UNIVERSE"),
        required=True,
        help="Manifest-gated MOEX snapshot and the exact registry used to build it; repeatable.",
    )
    parser.add_argument("--key-rate", type=Path, help="Optional official CBR key-rate CSV.")
    parser.add_argument("--uzbekistan-data", type=Path, help="Optional normalized Uzbekistan data directory.")
    parser.add_argument("--bank-quotes", type=Path, help="Optional real point-in-time RUB->UZS bank quotes.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/amd_feature_enrichment_experiment.json")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("feature-enrichment experiment must be preregistered")
    if config.get("feature_sets") != EXPECTED_FEATURE_SETS:
        raise ValueError("feature sets differ from preregistered implementation")
    if config.get("consensus_variants") != EXPECTED_CONSENSUS:
        raise ValueError("consensus variants differ from preregistered implementation")
    if set(config.get("corridors", ())) != {"AMD", "KGS", "KZT", "TJS", "UZS"}:
        raise ValueError("all five corridors are required")
    return config


def _columns(frame: pd.DataFrame, config: dict[str, Any], feature_set: str) -> list[str]:
    prefixes = tuple(config["base_feature_prefixes"])
    result = [column for column in frame if column.startswith(prefixes)]
    for group in config["feature_sets"][feature_set]:
        result.extend(column for column in frame if column.startswith(FEATURE_GROUP_PREFIXES[group]))
    return list(dict.fromkeys(result))


def _fit_scores(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizon = int(config["horizon"])
    robust = load_robust_config()
    robust["horizons"] = [horizon]
    data = add_registered_labels(frame, robust)
    label = f"label__tb_{horizon}"
    outcome = f"outcome__regret_{horizon}"
    benefit = f"outcome__benefit_{horizon}"
    score_rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    for feature_set in config["feature_sets"]:
        requested_columns = _columns(data, config, feature_set)
        for corridor in config["corridors"]:
            corridor_data = data.loc[data["corridor"].eq(corridor) & data[outcome].notna()].copy()
            corridor_data = corridor_data.sort_values("timestamp", kind="mergesort")
            for year in config["test_years"]:
                start = pd.Timestamp(year=int(year), month=1, day=1)
                end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                train = _purged_train(corridor_data, start, horizon)
                train = train.loc[
                    train["timestamp"].ge(start - pd.DateOffset(years=int(config["rolling_training_years"])))
                    & train[label].notna()
                ].copy()
                test = corridor_data.loc[corridor_data["timestamp"].between(start, end, inclusive="left")].copy()
                columns = [column for column in requested_columns if train[column].notna().any()]
                status = "ok"
                if (
                    len(train) < int(config["minimum_training_observations"])
                    or len(test) < 20
                    or train[label].nunique() < 2
                    or not columns
                ):
                    status = "skipped"
                diagnostics.append(
                    {
                        "feature_set": feature_set,
                        "corridor": corridor,
                        "test_year": int(year),
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "requested_features": len(requested_columns),
                        "fitted_features": len(columns),
                        "status": status,
                    }
                )
                if status != "ok":
                    continue
                seed = int(config["random_seed"]) + int(year) + horizon + 701
                model = _make_model("catboost", iterations=int(config["model_iterations"]), seed=seed)
                model.fit(train[columns], train[label].astype(int))
                scored = test[["timestamp", "corridor", outcome, benefit]].copy()
                scored = scored.rename(columns={outcome: "regret_bps", benefit: "benefit_bps"})
                scored["target"] = scored["regret_bps"].le(25).astype(int)
                scored["score"] = model.predict_proba(test[columns])[:, 1]
                scored["week"] = scored["timestamp"].dt.to_period("W").astype(str)
                weekly = scored.groupby("week", sort=False)["target"].mean()
                scored["matched_week_hit_rate"] = scored["week"].map(weekly).astype(float)
                scored["feature_set"] = feature_set
                scored["test_year"] = int(year)
                score_rows.append(scored)
                fitted = model.steps[-1][1]
                transformed = model.steps[0][1].get_feature_names_out(columns)
                for name, value in zip(transformed, fitted.feature_importances_, strict=True):
                    importances.append(
                        {
                            "feature_set": feature_set,
                            "corridor": corridor,
                            "test_year": int(year),
                            "feature": str(name),
                            "importance": float(value),
                        }
                    )
            print(f"scored {feature_set} / {corridor}", flush=True)
    if not score_rows:
        raise ValueError("no OOT scores were produced")
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(diagnostics), pd.DataFrame(importances)


def _consensus_frame(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = scores.sort_values(["feature_set", "corridor", "timestamp"], kind="mergesort").copy()
    data["own_causal_percentile"] = np.nan
    for (_, _), indices in data.groupby(["feature_set", "corridor"], sort=False).groups.items():
        ordered = data.loc[indices].sort_values("timestamp", kind="mergesort")
        data.loc[ordered.index, "own_causal_percentile"] = causal_percentile(
            ordered["score"],
            lookback=int(config["consensus_rank_lookback"]),
            minimum_history=int(config["consensus_rank_minimum_history"]),
        ).to_numpy()
    for column in (
        "other_rank_mean", "other_rank_min", "other_rank_max", "other_rank_std",
        "other_positive_share", "own_minus_other_mean",
    ):
        data[column] = np.nan
    for (_, _), indices in data.groupby(["feature_set", "timestamp"], sort=False).groups.items():
        ranks = data.loc[indices, "own_causal_percentile"]
        for index in indices:
            own = float(data.at[index, "own_causal_percentile"])
            others = ranks.drop(index).dropna().astype(float)
            if not np.isfinite(own) or others.empty:
                continue
            data.at[index, "other_rank_mean"] = float(others.mean())
            data.at[index, "other_rank_min"] = float(others.min())
            data.at[index, "other_rank_max"] = float(others.max())
            data.at[index, "other_rank_std"] = float(others.std(ddof=0))
            data.at[index, "other_positive_share"] = float(others.ge(0.5).mean())
            data.at[index, "own_minus_other_mean"] = own - float(others.mean())
    data["official_mean"] = 0.5 * data["own_causal_percentile"] + 0.5 * data["other_rank_mean"]
    data["agreement_weighted"] = (
        0.45 * data["own_causal_percentile"]
        + 0.35 * data["other_rank_mean"]
        + 0.20 * data["other_rank_min"]
    )
    data["agreement_gate"] = data["official_mean"]
    return data.dropna(subset=["official_mean"])


def _evaluate_consensus(
    consensus: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds: list[dict[str, Any]] = []
    signals: list[pd.DataFrame] = []
    policy = config["adaptive_policy"]
    gate = config["agreement_gate"]
    for (feature_set, corridor), group in consensus.groupby(["feature_set", "corridor"], sort=True):
        ordered = group.sort_values("timestamp", kind="mergesort")
        for variant in config["consensus_variants"]:
            scored = ordered.copy()
            scored["policy_score"] = scored[variant]
            candidates = adaptive_candidates(
                scored.rename(columns={"score": "raw_model_score", "policy_score": "score"}),
                share=float(policy["top_score_share"]),
                lookback=int(policy["lookback_observations"]),
                minimum_history=int(policy["minimum_history_observations"]),
            )
            if variant == "agreement_gate":
                candidates = candidates.loc[
                    candidates["other_positive_share"].ge(float(gate["minimum_other_positive_share"]))
                    & candidates["other_rank_std"].le(float(gate["maximum_other_rank_std"]))
                ]
            selected = _apply_policy(
                candidates,
                cooldown_days=int(policy["cooldown_days"]),
                weekly_cap=int(policy["weekly_cap"]),
            )
            if len(selected):
                exported = selected.copy()
                exported["consensus_variant"] = variant
                signals.append(exported)
            for year, test in scored.groupby("test_year", sort=True):
                year_selected = selected.loc[selected["test_year"].eq(year)]
                count = len(year_selected)
                base_rate = float(test["target"].mean())
                hit_rate = float(year_selected["target"].mean()) if count else np.nan
                duration = max(float((test["timestamp"].max() - test["timestamp"].min()).days) / 7, 1 / 7)
                folds.append(
                    {
                        "feature_set": feature_set,
                        "consensus_variant": variant,
                        "corridor": corridor,
                        "test_year": int(year),
                        "test_count": len(test),
                        "test_hits": int(test["target"].sum()),
                        "signal_count": count,
                        "signal_hits": int(year_selected["target"].sum()) if count else 0,
                        "matched_expected_hits": float(year_selected["matched_week_hit_rate"].sum()) if count else 0.0,
                        "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                        "duration_weeks": duration,
                    }
                )
    fold_frame = pd.DataFrame(folds)
    summary: list[dict[str, Any]] = []
    for keys, group in fold_frame.groupby(["feature_set", "consensus_variant", "corridor"], sort=True):
        feature_set, variant, corridor = keys
        count = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        matched_hits = float(group["matched_expected_hits"].sum())
        rate = hits / count if count else np.nan
        summary.append(
            {
                "feature_set": feature_set,
                "consensus_variant": variant,
                "corridor": corridor,
                "signals": count,
                "hits": hits,
                "hit_rate": rate,
                "baseline_hit_rate": base_hits / base_count if base_count else np.nan,
                "matched_week_hit_rate": matched_hits / count if count else np.nan,
                "raw_lift": rate / (base_hits / base_count) if count and base_hits else np.nan,
                "same_week_lift": rate / (matched_hits / count) if count and matched_hits else np.nan,
                "signals_per_week": count / float(group["duration_weeks"].sum()),
                "worst_year_lift": group["lift"].dropna().min(),
            }
        )
    return fold_frame, pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(), pd.DataFrame(summary)


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    base = pd.read_csv(args.features)
    key_rate = pd.read_csv(args.key_rate) if args.key_rate else None
    uzbekistan_data = None
    if args.uzbekistan_data:
        uzbekistan_data = {
            name: pd.read_csv(args.uzbekistan_data / f"{name}.csv")
            for name in ("fx", "policy", "inflation", "reserves", "remittances_proxy")
        }
    bank_quotes = pd.read_csv(args.bank_quotes) if args.bank_quotes else None
    sources = [(Path(snapshot), Path(universe)) for snapshot, universe in args.moex_source]
    enriched = build_enriched_features(
        base,
        moex_sources=sources,
        key_rate=key_rate,
        uzbekistan_data=uzbekistan_data,
        bank_quotes=bank_quotes,
    )
    scores, diagnostics, importances = _fit_scores(enriched.frame, config)
    consensus = _consensus_frame(scores, config)
    folds, signals, summary = _evaluate_consensus(consensus, config)

    gzip = {"method": "gzip", "mtime": 0}
    enriched.frame.to_csv(args.output / "features.csv.gz", index=False, compression=gzip)
    scores.to_csv(args.output / "raw_scores.csv.gz", index=False, compression=gzip)
    consensus.to_csv(args.output / "consensus_scores.csv.gz", index=False, compression=gzip)
    folds.to_csv(args.output / "folds.csv", index=False)
    signals.to_csv(args.output / "signals.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    diagnostics.to_csv(args.output / "training_diagnostics.csv", index=False)
    importances.to_csv(args.output / "feature_importance.csv.gz", index=False, compression=gzip)
    amd = summary.loc[summary["corridor"].eq("AMD")].sort_values(
        ["same_week_lift", "raw_lift"], ascending=False, kind="mergesort"
    )
    amd.to_csv(args.output / "amd_comparison.csv", index=False)
    meta = {
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "features": {"path": str(args.features), "sha256": _sha256(args.features)},
        "key_rate": ({"path": str(args.key_rate), "sha256": _sha256(args.key_rate)} if args.key_rate else None),
        "uzbekistan_data": (
            {
                "path": str(args.uzbekistan_data),
                "manifest_sha256": _sha256(args.uzbekistan_data / "manifest.json"),
            }
            if args.uzbekistan_data else None
        ),
        "bank_quotes": (
            {"path": str(args.bank_quotes), "sha256": _sha256(args.bank_quotes)}
            if args.bank_quotes else None
        ),
        "enrichment": enriched.provenance,
        "comparison_count": int(len(summary)),
        "warning": "Exploratory multiple-comparison run on previously inspected 2022-2026 OOT history; it cannot promote a new production winner.",
    }
    (args.output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(amd.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
