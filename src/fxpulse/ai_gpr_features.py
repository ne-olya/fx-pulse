"""Build conservative daily features from the LLM-derived AI-GPR index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_config(path: Path | str = Path("configs/ai_gpr.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported AI-GPR feature config")
    return config


def _past_zscore(values: pd.Series, window: int) -> pd.Series:
    history = values.shift(1).rolling(window, min_periods=max(5, window // 3))
    return (values - history.mean()) / history.std().replace(0, np.nan)


def _daily_features(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = raw.copy()
    data["feature_date"] = (
        pd.to_datetime(data["Date"], errors="raise")
        + pd.Timedelta(int(config["daily_availability_lag_days"]), unit="D")
    )
    result = data[["feature_date"]].copy()
    for short, source in config["daily_columns"].items():
        values = pd.to_numeric(data[source], errors="coerce")
        logged = np.log1p(values.clip(lower=0))
        result[f"aigpr__daily_{short}_level"] = values
        result[f"aigpr__daily_{short}_log_change_1d"] = logged.diff(1)
        result[f"aigpr__daily_{short}_log_change_7d"] = logged.diff(7)
        result[f"aigpr__daily_{short}_mean_7d"] = values.rolling(7, min_periods=3).mean()
        result[f"aigpr__daily_{short}_zscore_30d"] = _past_zscore(values, 30)
        result[f"aigpr__daily_{short}_zscore_90d"] = _past_zscore(values, 90)
    return result.loc[result["feature_date"].ge(pd.Timestamp(config["date_from"]))].copy()


def _monthly_available_dates(values: pd.Series, delay_days: int) -> pd.Series:
    month = pd.to_datetime(values, errors="raise")
    return month + pd.offsets.MonthEnd(1) + pd.Timedelta(delay_days, unit="D")


def _add_monthly_series(result: pd.DataFrame, name: str, values: pd.Series) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    result[f"{name}_level"] = numeric
    result[f"{name}_change_1m"] = numeric.diff(1)
    result[f"{name}_zscore_12m"] = _past_zscore(numeric, 12)


def _monthly_features(
    country_raw: pd.DataFrame,
    bilateral_raw: pd.DataFrame,
    config: dict[str, Any],
    corridor: str,
) -> pd.DataFrame:
    country = country_raw.copy()
    bilateral = bilateral_raw.copy()
    delay = int(config["monthly_availability_day_after_month_end"])
    country["feature_date"] = _monthly_available_dates(country["Date"], delay)
    bilateral["feature_date"] = _monthly_available_dates(bilateral["Date"], delay)
    if not country["feature_date"].equals(bilateral["feature_date"]):
        raise ValueError("AI-GPR country and bilateral monthly dates differ")
    recipient = str(config["corridor_countries"][corridor])
    result = country[["feature_date"]].copy()
    for role in ("all", "initiator", "respondent", "spillover"):
        _add_monthly_series(result, f"aigpr__country_russia_{role}", country[f"Russia_{role}"])
        _add_monthly_series(
            result,
            f"aigpr__country_recipient_{role}",
            country[f"{recipient}_{role}"],
        )
    directions = [
        column
        for column in (f"Russia|{recipient}", f"{recipient}|Russia")
        if column in bilateral
    ]
    if not directions:
        raise ValueError(f"AI-GPR bilateral input lacks Russia/{recipient}")
    bilateral_total = bilateral[directions].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    _add_monthly_series(result, "aigpr__bilateral_russia_recipient", bilateral_total)
    result["aigpr__cross_country_all_gap"] = (
        pd.to_numeric(country["Russia_all"], errors="coerce")
        - pd.to_numeric(country[f"{recipient}_all"], errors="coerce")
    )
    return result.loc[result["feature_date"].ge(pd.Timestamp(config["date_from"]))].copy()


def build_features(
    daily_raw: pd.DataFrame,
    country_raw: pd.DataFrame,
    bilateral_raw: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    daily = _daily_features(daily_raw, config).sort_values("feature_date")
    outputs: list[pd.DataFrame] = []
    for corridor in config["corridor_countries"]:
        monthly = _monthly_features(country_raw, bilateral_raw, config, corridor).sort_values(
            "feature_date"
        )
        merged = pd.merge_asof(daily, monthly, on="feature_date", direction="backward")
        merged.insert(1, "corridor", corridor)
        outputs.append(merged)
    return pd.concat(outputs, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ai_gpr.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    features = build_features(
        pd.read_csv(config["raw_files"]["daily"]),
        pd.read_csv(config["raw_files"]["country_monthly"]),
        pd.read_csv(config["raw_files"]["bilateral_monthly"]),
        config,
    )
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(
        json.dumps(
            {
                "rows": len(features),
                "features": len([column for column in features if column.startswith("aigpr__")]),
                "date_from": str(features["feature_date"].min().date()),
                "date_to": str(features["feature_date"].max().date()),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
