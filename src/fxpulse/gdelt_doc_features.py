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
    lag = int(config["availability_lag_days"])
    windows = [int(value) for value in config["rolling_windows_days"]]
    shock_window = int(config["shock_window_days"])
    shared = ["russia", "sanctions", "currency", "energy"]
    outputs: list[pd.DataFrame] = []
    for corridor, recipient in config["corridor_series"].items():
        result = pd.DataFrame({"feature_date": dates + pd.Timedelta(lag, unit="D")})
        result["corridor"] = corridor
        for series in [*shared, recipient]:
            short = "recipient" if series == recipient else series
            count = counts[series].fillna(0.0)
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
