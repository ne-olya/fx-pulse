"""Deterministic checks for the raw FX Pulse data contracts."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd


def _number(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}".replace(",", " ")


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _weekday_gaps(dates: pd.Series) -> tuple[int, str]:
    days = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise").drop_duplicates().sort_values())
    if days.empty:
        return 0, "—"
    expected = pd.bdate_range(days.min(), days.max())
    missing = expected.difference(days)
    sample = ", ".join(day.date().isoformat() for day in missing[:5]) or "—"
    return len(missing), sample


def _outlier_count(prices: pd.Series) -> int:
    returns = prices.pct_change().dropna()
    sigma = returns.std(ddof=1)
    if pd.isna(sigma) or sigma == 0:
        return 0
    return int(returns.abs().gt(5 * sigma).sum())


def _cbr_section(raw_dir: Path) -> tuple[str, dict[str, int]]:
    path = raw_dir / "cbr_daily.csv"
    frame = pd.read_csv(path)
    required = {"rate_date", "ccy", "nominal", "rate_rub", "fetched_at", "source_url"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    frame["rate_date"] = pd.to_datetime(frame["rate_date"], errors="raise")
    frame["nominal"] = pd.to_numeric(frame["nominal"], errors="coerce")
    frame["rate_rub"] = pd.to_numeric(frame["rate_rub"], errors="coerce")
    frame["price"] = frame["rate_rub"] / frame["nominal"]
    rows: list[tuple[object, ...]] = []
    for ccy, group in frame.groupby("ccy", sort=True):
        group = group.sort_values("rate_date")
        gaps, gap_sample = _weekday_gaps(group["rate_date"])
        invalid = group["price"].isna() | group["price"].le(0)
        rows.append(
            (
                ccy,
                len(group),
                int(group.duplicated("rate_date").sum()),
                gaps,
                gap_sample,
                int(group["nominal"].nunique(dropna=True)),
                int(invalid.sum()),
                _outlier_count(group.loc[~invalid, "price"]),
            )
        )

    pivot = frame.pivot_table(index="rate_date", columns="ccy", values="price", aggfunc="first")
    cross_rows: list[tuple[object, ...]] = []
    if "USD" in pivot:
        for ccy in ("TJS", "UZS", "KGS", "KZT", "AMD"):
            if ccy in pivot:
                reconstructed = (pivot["USD"] / pivot[ccy]).dropna()
                cross_rows.append(
                    (
                        ccy,
                        len(reconstructed),
                        _number(reconstructed.median(), 6),
                        _number(reconstructed.min(), 6),
                        _number(reconstructed.max(), 6),
                    )
                )

    duplicates = int(frame.duplicated(["rate_date", "ccy"]).sum())
    section = "\n".join(
        [
            "## ЦБ РФ, дневные курсы",
            "",
            "Номиналы нормированы в `price = rate_rub / nominal`; исходные `nominal` и `rate_rub` сохранены в raw.",
            "",
            _markdown_table(
                ("Валюта", "Строк", "Дубли", "Пропущ. будни*", "Пример", "Номиналов", "Невалидных", ">5σ"),
                rows,
            ),
            "",
            "* Это прокси по будням, а не производственный календарь РФ: праздничные дни требуют отдельной календарной сверки.",
            "",
            "Восстановленная нога `USD_RUB / XXX_RUB` (диагностика доступности кросса; сравнение с источником нацбанка будет добавлено после его загрузки):",
            "",
            _markdown_table(("Нога", "Общих дат", "Медиана", "Мин.", "Макс."), cross_rows),
        ]
    )
    return section, {"cbr_rows": len(frame), "cbr_duplicates": duplicates}


def _moex_daily_section(raw_dir: Path) -> tuple[str, dict[str, int]]:
    path = raw_dir / "moex_daily.csv"
    frame = pd.read_csv(path)
    required = {"trade_date", "secid", "close", "waprice", "num_trades", "fetched_at", "source_url"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise")
    for column in ("close", "waprice", "num_trades"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rows: list[tuple[object, ...]] = []
    for secid, group in frame.groupby("secid", sort=True):
        group = group.sort_values("trade_date")
        gaps, gap_sample = _weekday_gaps(group["trade_date"])
        invalid = group["close"].isna() | group["close"].le(0)
        comparable = group["close"].gt(0) & group["waprice"].gt(0)
        close_vs_wap = ((group.loc[comparable, "close"] / group.loc[comparable, "waprice"] - 1) * 10_000).abs()
        rows.append(
            (
                secid,
                len(group),
                int(group.duplicated("trade_date").sum()),
                gaps,
                gap_sample,
                int(invalid.sum()),
                _number(close_vs_wap.median(), 2),
                _number(close_vs_wap.quantile(0.95), 2),
                _outlier_count(group.loc[~invalid, "close"]),
            )
        )
    duplicates = int(frame.duplicated(["trade_date", "secid"]).sum())
    section = "\n".join(
        [
            "## MOEX, дневные закрытия",
            "",
            _markdown_table(
                ("Инструмент", "Строк", "Дубли", "Пропущ. будни*", "Пример", "close ≤ 0 / пуст.", "med abs(close/WAP−1), бп", "p95, бп", ">5σ"),
                rows,
            ),
            "",
            "Нулевые или отсутствующие `close` остаются в raw для аудита, но исключаются из `load_panel` без forward-fill. Это предотвращает ложный минимум и ложную плоскую динамику.",
        ]
    )
    return section, {"moex_daily_rows": len(frame), "moex_daily_duplicates": duplicates}


def _moex_candles_section(raw_dir: Path) -> tuple[str, dict[str, int]]:
    path = raw_dir / "moex_candles.csv"
    frame = pd.read_csv(path)
    required = {"dt_msk", "secid", "close", "fetched_at", "source_url"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    frame["dt_msk"] = pd.to_datetime(frame["dt_msk"], errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    rows: list[tuple[object, ...]] = []
    for secid, group in frame.groupby("secid", sort=True):
        invalid = group["close"].isna() | group["close"].le(0)
        rows.append(
            (
                secid,
                len(group),
                int(group.duplicated("dt_msk").sum()),
                int(invalid.sum()),
                group["dt_msk"].min(),
                group["dt_msk"].max(),
            )
        )
    duplicates = int(frame.duplicated(["dt_msk", "secid"]).sum())
    section = "\n".join(
        [
            "## MOEX, 10-минутные свечи",
            "",
            _markdown_table(("Инструмент", "Строк", "Дубли", "close ≤ 0 / пуст.", "Начало", "Конец"), rows),
            "",
            "`dt_msk` — время закрытия свечи (`END` из ISS), поэтому наблюдение не доступно внутри самой свечи.",
        ]
    )
    return section, {"moex_candle_rows": len(frame), "moex_candle_duplicates": duplicates}


def _moex_hourly_section(raw_dir: Path) -> tuple[str, dict[str, int]]:
    path = raw_dir / "moex_cny_60m.csv"
    if not path.exists():
        return "", {"moex_hourly_rows": 0, "moex_hourly_duplicates": 0}
    frame = pd.read_csv(path)
    required = {"dt_msk", "secid", "open", "high", "low", "close", "fetched_at", "source_url"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    frame["dt_msk"] = pd.to_datetime(frame["dt_msk"], errors="raise")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[tuple[object, ...]] = []
    for secid, group in frame.groupby("secid", sort=True):
        invalid = group[["open", "high", "low", "close"]].isna().any(axis=1) | group["close"].le(0)
        per_day = group.groupby(group["dt_msk"].dt.date).size()
        rows.append(
            (
                secid,
                len(group),
                int(group.duplicated("dt_msk").sum()),
                int(invalid.sum()),
                group["dt_msk"].min(),
                group["dt_msk"].max(),
                _number(per_day.median(), 1),
            )
        )
    duplicates = int(frame.duplicated(["dt_msk", "secid"]).sum())
    section = "\n".join(
        [
            "## MOEX, часовые свечи эксперимента",
            "",
            _markdown_table(
                ("Инструмент", "Строк", "Дубли", "Невалидных", "Начало", "Конец", "Медиана свечей/день"),
                rows,
            ),
            "",
            "Одна строка известна только после времени `dt_msk`, то есть после закрытия соответствующей свечи.",
        ]
    )
    return section, {"moex_hourly_rows": len(frame), "moex_hourly_duplicates": duplicates}


def build_report(raw_dir: Path | str = Path("data/raw")) -> str:
    """Return the Markdown quality report for the currently downloaded raw data."""

    raw_path = Path(raw_dir)
    cbr, cbr_summary = _cbr_section(raw_path)
    daily, daily_summary = _moex_daily_section(raw_path)
    candles, candles_summary = _moex_candles_section(raw_path)
    hourly, hourly_summary = _moex_hourly_section(raw_path)
    boundaries = ("2022-01-01", "2024-06-13", "2024-12-27")
    present = pd.read_csv(raw_path / "cbr_daily.csv", usecols=["rate_date"])["rate_date"].astype(str)
    boundary_rows = [(boundary, "да" if boundary in set(present) else "нет") for boundary in boundaries]
    summary = {**cbr_summary, **daily_summary, **candles_summary, **hourly_summary}
    return "\n".join(
        [
            "# Качество данных",
            "",
            f"Сформировано: {dt.datetime.now(dt.UTC).isoformat()}. Проверены raw-файлы в `{raw_path}`.",
            "",
            "## Итог",
            "",
            f"- ЦБ: {summary['cbr_rows']} строк, дубликатов ключа `(rate_date, ccy)` — {summary['cbr_duplicates']}.",
            f"- MOEX дневной: {summary['moex_daily_rows']} строк, дубликатов `(trade_date, secid)` — {summary['moex_daily_duplicates']}.",
            f"- MOEX 10 минут: {summary['moex_candle_rows']} строк, дубликатов `(dt_msk, secid)` — {summary['moex_candle_duplicates']}.",
            f"- MOEX 1 час: {summary['moex_hourly_rows']} строк, дубликатов `(dt_msk, secid)` — {summary['moex_hourly_duplicates']}.",
            "- Исправление в аналитическом слое: технические нулевые MOEX `close` не переносятся вперёд и не становятся ценой; они остаются в raw и исключаются из панели с предупреждением.",
            "- Известное ограничение: пока не загружены ноги нацбанков и производственные календари, нельзя завершить их сверку и отличить официальный перенос от праздника страны-получателя.",
            "",
            cbr,
            "",
            daily,
            "",
            candles,
            "",
            hourly,
            "",
            "## Границы режимов",
            "",
            _markdown_table(("Дата", "Есть в ряде ЦБ"), boundary_rows),
            "",
            "Границы будут переданы как явные разрезы в walk-forward. Их присутствие в raw не доказывает отсутствие структурного сдвига и не заменяет режимный анализ.",
        ]
    ) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("docs/data-quality.md"))
    args = parser.parse_args(argv)
    report = build_report(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
