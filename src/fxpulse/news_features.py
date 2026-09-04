"""Build point-in-time hourly and daily features from GDELT aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COUNT_COLUMNS = (
    "article_count",
    "russia_article_count",
    "recipient_article_count",
    "bilateral_article_count",
    "source_count",
    "tone_count",
    "russia_tone_count",
    "recipient_tone_count",
    "negative_count",
    "conflict_count",
    "material_conflict_count",
    "cooperation_count",
    "coercion_count",
    "protest_count",
    "goldstein_count",
)
SUM_COLUMNS = (
    "tone_sum",
    "russia_tone_sum",
    "recipient_tone_sum",
    "goldstein_sum",
)
REQUIRED_COLUMNS = {"timestamp_utc", "corridor", *COUNT_COLUMNS, *SUM_COLUMNS}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def _past_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, max(3, window // 4))
    history = series.shift(1).rolling(window, min_periods=min_periods)
    mean = history.mean()
    std = history.std().replace(0, np.nan)
    return (series - mean) / std


def _past_ratio_to_median(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, max(3, window // 4))
    median = series.shift(1).rolling(window, min_periods=min_periods).median()
    return (series + 1.0) / (median + 1.0)


def _validate(raw: pd.DataFrame, corridors: Iterable[str]) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"GDELT aggregate lacks columns: {', '.join(sorted(missing))}")
    data = raw.copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="raise")
    data["corridor"] = data["corridor"].astype(str).str.upper()
    allowed = set(corridors)
    unknown = set(data["corridor"].unique()) - allowed
    if unknown:
        raise ValueError(f"unexpected corridors: {', '.join(sorted(unknown))}")
    if data.duplicated(["timestamp_utc", "corridor"]).any():
        raise ValueError("GDELT aggregate must contain one row per hour and corridor")
    for column in (*COUNT_COLUMNS, *SUM_COLUMNS):
        data[column] = pd.to_numeric(data[column], errors="raise").fillna(0.0)
    return data.sort_values(["corridor", "timestamp_utc"]).reset_index(drop=True)


def _complete_grid(data: pd.DataFrame, corridors: Iterable[str], frequency: str) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    start = data["timestamp_utc"].min().floor(frequency)
    end = data["timestamp_utc"].max().floor(frequency)
    index = pd.MultiIndex.from_product(
        [list(corridors), pd.date_range(start, end, freq=frequency, tz="UTC")],
        names=["corridor", "timestamp_utc"],
    )
    completed = data.set_index(["corridor", "timestamp_utc"]).reindex(index)
    for column in (*COUNT_COLUMNS, *SUM_COLUMNS):
        completed[column] = completed[column].fillna(0.0)
    return completed.reset_index()


def _add_common_features(
    data: pd.DataFrame,
    *,
    rolling_windows: Iterable[int],
    shock_windows: Iterable[int],
    suffix: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in data.groupby("corridor", sort=False):
        group = group.copy().sort_values("timestamp_utc")
        group["news__article_count"] = group["article_count"]
        group["news__russia_count"] = group["russia_article_count"]
        group["news__recipient_count"] = group["recipient_article_count"]
        group["news__bilateral_count"] = group["bilateral_article_count"]
        group["news__source_count"] = group["source_count"]
        group["news__avg_tone"] = _safe_ratio(group["tone_sum"], group["tone_count"])
        group["news__russia_avg_tone"] = _safe_ratio(
            group["russia_tone_sum"], group["russia_tone_count"]
        )
        group["news__recipient_avg_tone"] = _safe_ratio(
            group["recipient_tone_sum"], group["recipient_tone_count"]
        )
        group["news__negative_share"] = _safe_ratio(group["negative_count"], group["article_count"])
        group["news__conflict_share"] = _safe_ratio(group["conflict_count"], group["article_count"])
        group["news__material_conflict_share"] = _safe_ratio(
            group["material_conflict_count"], group["article_count"]
        )
        group["news__cooperation_share"] = _safe_ratio(
            group["cooperation_count"], group["article_count"]
        )
        group["news__coercion_share"] = _safe_ratio(group["coercion_count"], group["article_count"])
        group["news__protest_share"] = _safe_ratio(group["protest_count"], group["article_count"])
        group["news__avg_goldstein"] = _safe_ratio(group["goldstein_sum"], group["goldstein_count"])

        for window in rolling_windows:
            for source, short_name in (
                ("article_count", "volume"),
                ("russia_article_count", "russia_volume"),
                ("recipient_article_count", "recipient_volume"),
                ("bilateral_article_count", "bilateral_volume"),
                ("negative_count", "negative_volume"),
                ("conflict_count", "conflict_volume"),
                ("coercion_count", "coercion_volume"),
            ):
                group[f"news__{short_name}_{window}{suffix}"] = group[source].rolling(
                    window, min_periods=1
                ).sum()

        for window in shock_windows:
            for source, short_name in (
                ("article_count", "volume"),
                ("negative_count", "negative"),
                ("conflict_count", "conflict"),
                ("recipient_article_count", "recipient_volume"),
            ):
                group[f"news__{short_name}_zscore_{window}{suffix}"] = _past_zscore(
                    group[source], window
                )
                group[f"news__{short_name}_ratio_{window}{suffix}"] = _past_ratio_to_median(
                    group[source], window
                )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def build_hourly_features(raw: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    """Return features available after each completed UTC news hour."""

    corridors = [str(item) for item in config["corridors"]]
    data = _complete_grid(_validate(raw, corridors), corridors, "h")
    data = _add_common_features(
        data,
        rolling_windows=[int(value) for value in config["hourly_windows"]],
        shock_windows=[int(value) for value in config["shock_baselines_hours"]],
        suffix="h",
    )
    delay = int(config["availability_delay_minutes"])
    data["available_at_utc"] = data["timestamp_utc"] + pd.to_timedelta(60 + delay, unit="m")
    return data


def _daily_decision_date(
    timestamps: pd.Series,
    *,
    timezone: str,
    decision_time: str,
    delay_minutes: int,
) -> pd.Series:
    local = (timestamps + pd.to_timedelta(60 + delay_minutes, unit="m")).dt.tz_convert(timezone)
    hour, minute = (int(part) for part in decision_time.split(":"))
    cutoff_minutes = hour * 60 + minute
    after_cutoff = (local.dt.hour * 60 + local.dt.minute) > cutoff_minutes
    dates = local.dt.normalize() + pd.to_timedelta(after_cutoff.astype(int), unit="D")
    return dates.dt.tz_localize(None)


def build_daily_features(raw: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    """Aggregate each hour into the next observable Moscow decision window."""

    corridors = [str(item) for item in config["corridors"]]
    data = _validate(raw, corridors)
    data["feature_date"] = _daily_decision_date(
        data["timestamp_utc"],
        timezone=str(config["decision_timezone"]),
        decision_time=str(config["daily_decision_time"]),
        delay_minutes=int(config["availability_delay_minutes"]),
    )
    daily = (
        data.groupby(["corridor", "feature_date"], as_index=False)[[*COUNT_COLUMNS, *SUM_COLUMNS]]
        .sum()
        .rename(columns={"feature_date": "timestamp_utc"})
    )
    daily["timestamp_utc"] = pd.to_datetime(daily["timestamp_utc"], utc=True)
    daily = _complete_grid(daily, corridors, "D")
    daily = _add_common_features(
        daily,
        rolling_windows=[int(value) for value in config["daily_windows"]],
        shock_windows=[int(value) for value in config["shock_baselines_days"]],
        suffix="d",
    )
    daily["feature_date"] = daily["timestamp_utc"].dt.tz_localize(None)
    daily["available_at_msk"] = (
        daily["feature_date"].dt.strftime("%Y-%m-%d") + " " + str(config["daily_decision_time"])
    )
    return daily.drop(columns=["timestamp_utc"])


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported news feature config")
    return config


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("news__")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/gdelt_hourly_news.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/news_features.json"))
    parser.add_argument(
        "--hourly-output", type=Path, default=Path("data/processed/gdelt_news_hourly_features.csv")
    )
    parser.add_argument(
        "--daily-output", type=Path, default=Path("data/processed/gdelt_news_daily_features.csv")
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    raw = pd.read_csv(args.input)
    hourly = build_hourly_features(raw, config)
    daily = build_daily_features(raw, config)
    args.hourly_output.parent.mkdir(parents=True, exist_ok=True)
    args.daily_output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(args.hourly_output, index=False)
    daily.to_csv(args.daily_output, index=False)
    print(
        json.dumps(
            {
                "hourly_rows": len(hourly),
                "daily_rows": len(daily),
                "hourly_features": len(_feature_columns(hourly)),
                "daily_features": len(_feature_columns(daily)),
                "hourly_output": str(args.hourly_output),
                "daily_output": str(args.daily_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
