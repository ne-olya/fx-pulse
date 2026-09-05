"""Build conservative daily news features from GDELT DOC timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_config(path: Path | str = Path("configs/gdelt_doc.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported GDELT DOC feature config")
    return config


def _past_zscore(values: pd.Series, window: int) -> pd.Series:
    history = values.shift(1).rolling(window, min_periods=max(5, window // 3))
    return (values - history.mean()) / history.std().replace(0, np.nan)


def build_features(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required = {"date", "series", "article_count", "all_article_count", "article_share"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"GDELT DOC daily input lacks {sorted(missing)}")
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data.duplicated(["date", "series"]).any():
        raise ValueError("GDELT DOC input must contain one row per date/series")
    for column in ("article_count", "all_article_count", "article_share"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    expected = set(config["queries"])
    if set(data["series"].unique()) != expected:
        absent = expected - set(data["series"].unique())
        raise ValueError(f"GDELT DOC input lacks configured series {sorted(absent)}")

    dates = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
    counts = data.pivot(index="date", columns="series", values="article_count").reindex(dates)
    shares = data.pivot(index="date", columns="series", values="article_share").reindex(dates)
    norms = data.pivot(index="date", columns="series", values="all_article_count").reindex(dates)
    source_observed = norms.notna().any(axis=1)
    lag = int(config["availability_lag_days"])
    windows = [int(value) for value in config["rolling_windows_days"]]
    shock_window = int(config["shock_window_days"])
    shared = ["russia", "sanctions", "currency", "energy"]
    outputs: list[pd.DataFrame] = []
    for corridor, recipient in config["corridor_series"].items():
        result = pd.DataFrame({"feature_date": dates + pd.Timedelta(lag, unit="D")})
        result["corridor"] = corridor
        result["news__source_missing"] = (~source_observed).astype(float).to_numpy()
        for window in windows:
            result[f"news__source_coverage_{window}d"] = (
                source_observed.astype(float).rolling(window, min_periods=1).mean().to_numpy()
            )
        configured = [(series, series) for series in shared]
        configured.append(("recipient", recipient))
        configured.extend(
            (short, series)
            for short, series in config.get("corridor_additional_series", {}).get(corridor, {}).items()
        )
        for short, series in configured:
            # A missing query row on an otherwise observed date means zero matches.
            # A date absent from every query is a GDELT outage and must stay unknown.
            count = counts[series].fillna(0.0).where(source_observed)
            share = shares[series]
            result[f"news__{short}_count"] = count.to_numpy()
            result[f"news__{short}_share"] = share.to_numpy()
            for window in windows:
                result[f"news__{short}_count_{window}d"] = count.rolling(
                    window, min_periods=1
                ).sum().to_numpy()
            result[f"news__{short}_count_zscore_{shock_window}d"] = _past_zscore(
                count, shock_window
            ).to_numpy()
            result[f"news__{short}_share_zscore_{shock_window}d"] = _past_zscore(
                share, shock_window
            ).to_numpy()

        recipient_count_shock = result[f"news__recipient_count_zscore_{shock_window}d"]
        recipient_share_shock = result[f"news__recipient_share_zscore_{shock_window}d"]
        russia_count_shock = result[f"news__russia_count_zscore_{shock_window}d"]
        russia_share_shock = result[f"news__russia_share_zscore_{shock_window}d"]
        result[f"news__cross_country_count_shock_gap_{shock_window}d"] = (
            russia_count_shock - recipient_count_shock
        )
        result[f"news__cross_country_share_shock_gap_{shock_window}d"] = (
            russia_share_shock - recipient_share_shock
        )
        result[f"news__cross_country_joint_positive_shock_{shock_window}d"] = (
            russia_count_shock.clip(lower=0) * recipient_count_shock.clip(lower=0)
        )
        if f"news__recipient_macro_count_zscore_{shock_window}d" in result:
            result[f"news__cross_macro_shock_gap_{shock_window}d"] = (
                result[f"news__currency_count_zscore_{shock_window}d"]
                - result[f"news__recipient_macro_count_zscore_{shock_window}d"]
            )
            comparison_window = max(windows)
            result[f"news__cross_recipient_macro_share_{comparison_window}d"] = (
                result[f"news__recipient_macro_count_{comparison_window}d"]
                / result[f"news__recipient_count_{comparison_window}d"].replace(0, np.nan)
            )
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gdelt_doc.json"))
    parser.add_argument("--input", type=Path, default=Path("data/raw/gdelt_doc_daily.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/gdelt_news_daily_features.csv")
    )
    args = parser.parse_args()
    features = build_features(pd.read_csv(args.input), load_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "rows": len(features),
                "features": len([column for column in features if column.startswith("news__")]),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
