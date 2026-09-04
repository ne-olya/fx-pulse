"""Compare market-only, news-only and combined daily CatBoost models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import _apply_policy, _feature_importance, _make_model, _purged_train


EXPECTED_VARIANTS = ["market_only", "news_only", "market_plus_news"]


def load_config(path: Path | str = Path("configs/news_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("news experiment must be preregistered before results")
    if config.get("variants") != EXPECTED_VARIANTS:
        raise ValueError("news variants differ from the fixed comparison")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_frame(market: pd.DataFrame, news: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    market = market.copy()
    news = news.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], errors="raise").dt.normalize()
    news["feature_date"] = pd.to_datetime(news["feature_date"], errors="raise").dt.normalize()
    market["corridor"] = market["corridor"].astype(str).str.upper()
    news["corridor"] = news["corridor"].astype(str).str.upper()
    news_columns = [column for column in news if column.startswith(tuple(config["news_feature_prefixes"]))]
    if not news_columns:
        raise ValueError("news input contains no registered news feature")
    if news.duplicated(["feature_date", "corridor"]).any():
        raise ValueError("news input must have one row per feature date and corridor")
    joined = market.merge(
        news[["feature_date", "corridor", *news_columns]],
        left_on=["timestamp", "corridor"],
        right_on=["feature_date", "corridor"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="feature_date")
    allowed = set(config["corridors"])
    joined = joined.loc[joined["corridor"].isin(allowed)].copy()
    return joined.sort_values(["corridor", "timestamp"]).reset_index(drop=True)


def feature_sets(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[str]]:
    market = [column for column in frame if column.startswith(tuple(config["market_feature_prefixes"]))]
    news = [column for column in frame if column.startswith(tuple(config["news_feature_prefixes"]))]
    if not market or not news:
        raise ValueError("both market and news features are required")
    return {
        "market_only": market,
        "news_only": news,
        "market_plus_news": [*market, *news],
    }


def fit_oot_scores(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sets = feature_sets(frame, config)
    scores: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    for corridor in config["corridors"]:
        corridor_data = frame.loc[frame["corridor"].eq(corridor)].copy()
        for horizon in config["horizons"]:
            h = int(horizon)
            outcome = f"outcome__regret_{h}"
            benefit = f"outcome__benefit_{h}"
            usable = corridor_data.loc[corridor_data[outcome].notna()].copy()
            usable["target"] = usable[outcome].le(float(config["tolerance_bps"])).astype(int)
            for year in config["test_years"]:
                start = pd.Timestamp(year=int(year), month=1, day=1)
                end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                train = _purged_train(usable, start, h)
                test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                if (
                    len(train) < int(config["minimum_training_rows"])
                    or test.empty
                    or train["target"].nunique() < 2
                ):
                    continue
                for variant in config["variants"]:
                    columns = sets[variant]
                    model = _make_model(
                        str(config["model"]),
                        iterations=int(config["model_iterations"]),
                        seed=int(config["random_seed"]) + int(year) + h,
                    )
                    model.fit(train[columns], train["target"])
                    predicted = test[["timestamp", "corridor", "target", outcome, benefit]].copy()
                    predicted["score"] = model.predict_proba(test[columns])[:, 1]
                    predicted["variant"] = variant
                    predicted["horizon"] = h
                    predicted["test_year"] = int(year)
                    predicted = predicted.rename(columns={outcome: "regret_bps", benefit: "benefit_bps"})
                    scores.append(predicted)
                    fold_rows.append(
                        {
                            "corridor": corridor,
                            "horizon": h,
                            "test_year": int(year),
                            "variant": variant,
                            "train_rows": len(train),
                            "test_rows": len(test),
                            "test_base_rate": float(test["target"].mean()),
                            "train_end": train["timestamp"].max(),
                            "test_start": test["timestamp"].min(),
                        }
                    )
                    for name, value, signed in _feature_importance(model, columns):
                        importance_rows.append(
                            {
                                "corridor": corridor,
                                "horizon": h,
                                "test_year": int(year),
                                "variant": variant,
                                "feature": name,
                                "importance": value,
                                "signed_effect": signed,
                            }
                        )
    if not scores:
        raise RuntimeError("news experiment produced no OOT score")
    output_scores = pd.concat(scores, ignore_index=True)
    identity = ["corridor", "horizon", "test_year", "timestamp"]
    date_counts = output_scores.groupby(identity)["variant"].nunique()
    if not date_counts.eq(len(EXPECTED_VARIANTS)).all():
        raise RuntimeError("the three news variants were not evaluated on identical OOT dates")
    return output_scores, pd.DataFrame(fold_rows), pd.DataFrame(importance_rows)


def apply_policy(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    policy = config["policy"]
    selected: list[pd.DataFrame] = []
    for _, group in scores.groupby(["variant", "corridor", "horizon"], sort=False):
        ordered = group.sort_values("timestamp").copy()
        iso = ordered["timestamp"].dt.isocalendar()
        ordered["week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
        weekly_rate = ordered.groupby("week")["target"].mean()
        ordered["matched_week_hit_rate"] = ordered["week"].map(weekly_rate)
        candidates = adaptive_candidates(
            ordered,
            share=float(policy["top_score_share"]),
            lookback=int(policy["lookback_observations"]),
            minimum_history=int(policy["minimum_history_observations"]),
        )
        chosen = _apply_policy(
            candidates,
            cooldown_days=int(policy["cooldown_days"]),
            weekly_cap=int(policy["weekly_cap"]),
        )
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def summarize(scores: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["variant", "corridor", "horizon"]
    for identity, all_scores in scores.groupby(keys, sort=True):
        variant, corridor, horizon = identity
        chosen = signals.loc[
            signals["variant"].eq(variant)
            & signals["corridor"].eq(corridor)
            & signals["horizon"].eq(horizon)
        ]
        count = len(chosen)
        hit_rate = float(chosen["target"].mean()) if count else np.nan
        base_rate = float(all_scores["target"].mean())
        matched_rate = float(chosen["matched_week_hit_rate"].mean()) if count else np.nan
        duration_weeks = max((all_scores["timestamp"].max() - all_scores["timestamp"].min()).days / 7, 1 / 7)
        rows.append(
            {
                "variant": variant,
                "corridor": corridor,
                "horizon": int(horizon),
                "oot_rows": len(all_scores),
                "signals": count,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "same_week_hit_rate": matched_rate,
                "raw_lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                "same_week_lift": hit_rate / matched_rate if count and matched_rate > 0 else np.nan,
                "signals_per_week": count / duration_weeks,
                "regret_mean_bps": float(chosen["regret_bps"].mean()) if count else np.nan,
                "benefit_mean_bps": float(chosen["benefit_bps"].mean()) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(config_path: Path, artifact_dir: Path) -> None:
    config = load_config(config_path)
    market_path = Path(config["market_input"])
    news_path = Path(config["news_input"])
    frame = prepare_frame(pd.read_csv(market_path), pd.read_csv(news_path), config)
    scores, folds, importance = fit_oot_scores(frame, config)
    signals = apply_policy(scores, config)
    summary = summarize(scores, signals)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(artifact_dir / "oot_scores.csv.gz", index=False, compression="gzip")
    folds.to_csv(artifact_dir / "folds.csv", index=False)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
    signals.to_csv(artifact_dir / "signals.csv", index=False)
    summary.to_csv(artifact_dir / "summary.csv", index=False)
    metadata = {
        "config_sha256": _sha256(config_path),
        "market_sha256": _sha256(market_path),
        "news_sha256": _sha256(news_path),
        "joined_rows": len(frame),
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "warning": "Exploratory news test on previously observed FX history; future or bank hold-out is still required.",
    }
    (artifact_dir / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/news_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/news"))
    args = parser.parse_args()
    run(args.config, args.artifact_dir)


if __name__ == "__main__":
    main()
