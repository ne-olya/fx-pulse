"""Evaluate the preregistered rule grid on the five product corridors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.grid import grid_sha256, load_grid, spec_id
from fxpulse.indicators import evaluate
from fxpulse.labeling import label_observations
from fxpulse.panel import load_panel


CORRIDORS = ("UZS", "TJS", "KGS", "AMD", "KZT")
FAMILY_LABELS = {
    "level_percentile": "Уровень: нижние проценты диапазона",
    "momentum_streak": "Моментум: N дней подряд вниз",
    "filter_rule": "Сильное однодневное снижение",
    "reversal_from_low": "Разворот вверх от минимума",
    "seasonality": "Сезонность",
    "volatility_regime": "Волатильность и режим",
    "model_consensus": "Скор модели с consensus",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _week_key(values: pd.Series) -> pd.Series:
    iso = pd.to_datetime(values, errors="raise").dt.isocalendar()
    return iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)


def _duration_weeks(labels: pd.DataFrame) -> float:
    """Match the established corridor reports: sum elapsed time per year."""

    duration = 0.0
    for _, part in labels.groupby(pd.to_datetime(labels["value_date"]).dt.year, sort=True):
        dates = pd.to_datetime(part["value_date"], errors="raise")
        duration += max(float((dates.max() - dates.min()).days) / 7, 1 / 7)
    return duration


def _metrics(labels: pd.DataFrame, positions: set[int]) -> dict[str, float | int | None]:
    chosen = labels.loc[labels["position"].isin(positions)].copy()
    signals = len(chosen)
    hits = int(chosen["hit_favorable"].sum()) if signals else 0
    baseline = float(labels["hit_favorable"].mean()) if len(labels) else np.nan
    hit_rate = hits / signals if signals else np.nan
    expected = float(chosen["matched_week_hit_rate"].sum()) if signals else 0.0
    duration = _duration_weeks(labels)
    return {
        "eligible_days": len(labels),
        "signals": signals,
        "hits": hits,
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline,
        "raw_lift": hit_rate / baseline if signals and baseline else np.nan,
        "same_week_lift": hits / expected if expected else np.nan,
        "signals_per_week": signals / duration if duration else np.nan,
        "mean_regret_bps": float(chosen["future_regret_bps"].mean()) if signals else np.nan,
    }


def _verdict(rows: pd.DataFrame) -> tuple[str, str]:
    usable = rows.loc[rows["signals"].ge(20) & rows["same_week_lift"].notna()]
    corridors = int(usable["corridor"].nunique())
    if corridors < 3:
        return "exclude_trigger", "Слишком мало сопоставимых срабатываний для межкоридорного вывода."
    raw_losses = int(usable["raw_lift"].lt(1.0).sum())
    matched_losses = int(usable["same_week_lift"].lt(1.0).sum())
    joint_wins = int(
        (usable["raw_lift"].gt(1.0) & usable["same_week_lift"].gt(1.0)).sum()
    )
    if raw_losses >= 4:
        return (
            "conditional_timing_only",
            "Не самостоятельный триггер: raw lift ниже 1 минимум на четырёх коридорах; условный same-week эффект можно использовать только как признак модели.",
        )
    if matched_losses >= 4:
        return "exclude_trigger", "Строгий lift ниже 1 как минимум на четырёх коридорах из пяти."
    if joint_wins >= 4:
        return "keep_candidate", "Оба lift выше 1 на большинстве коридоров; нужна отдельная подтверждающая выборка."
    return "message_context_only", "Результат неоднороден по коридорам; не использовать как самостоятельный триггер."


def evaluate_rule_grid(
    *,
    raw_dir: Path | str,
    grid_path: Path | str,
    date_from: str,
    date_to: str,
    horizon: int,
    tolerance_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = tuple(spec for spec in load_grid(grid_path) if spec.name != "uzs_frozen_consensus")
    detail_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for corridor in CORRIDORS:
        panel = load_panel(f"CBR:{corridor}", raw_dir=raw_dir).reset_index(drop=True)
        labels = label_observations(panel, horizon, tolerance_bps=tolerance_bps)
        dates = pd.to_datetime(labels["value_date"], errors="raise")
        labels = labels.loc[dates.between(pd.Timestamp(date_from), pd.Timestamp(date_to))].copy()
        labels["week"] = _week_key(labels["value_date"])
        week_rate = labels.groupby("week", sort=False)["hit_favorable"].mean()
        labels["matched_week_hit_rate"] = labels["week"].map(week_rate).astype(float)
        eligible = set(labels["position"].astype(int))
        fired: dict[str, set[int]] = {spec_id(spec): set() for spec in specs}
        by_family: dict[str, set[int]] = {name: set() for name in FAMILY_LABELS if name != "model_consensus"}
        for position in sorted(eligible):
            snapshot = panel.iloc[: position + 1].copy()
            snapshot.attrs["fxpulse_sorted_by_known_at"] = True
            for spec in specs:
                output = evaluate(spec.name, snapshot, **dict(spec.params))
                if output.fired:
                    fired[spec_id(spec)].add(position)
                    by_family[spec.name].add(position)
        for spec in specs:
            detail_rows.append(
                {
                    "row_type": "registered_config",
                    "indicator": spec.name,
                    "indicator_label": FAMILY_LABELS[spec.name],
                    "config_id": spec_id(spec),
                    "params": json.dumps(dict(spec.params), sort_keys=True, separators=(",", ":")),
                    "corridor": corridor,
                    **_metrics(labels, fired[spec_id(spec)]),
                }
            )
        for family, positions in by_family.items():
            family_rows.append(
                {
                    "row_type": "family_union",
                    "indicator": family,
                    "indicator_label": FAMILY_LABELS[family],
                    "config_id": "preregistered_family_union",
                    "params": "all preregistered grid configurations; union of dates",
                    "corridor": corridor,
                    **_metrics(labels, positions),
                }
            )
        print(f"completed rule grid for {corridor}", flush=True)
    details = pd.DataFrame(detail_rows)
    families = pd.DataFrame(family_rows)
    verdicts = {name: _verdict(group) for name, group in families.groupby("indicator", sort=False)}
    families["verdict"] = families["indicator"].map(lambda name: verdicts[name][0])
    families["verdict_reason"] = families["indicator"].map(lambda name: verdicts[name][1])
    return families, details


def _model_rows(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for corridor in CORRIDORS:
        values = payload["rows"][corridor]
        rows.append(
            {
                "row_type": "frozen_model_reference",
                "indicator": "model_consensus",
                "indicator_label": FAMILY_LABELS["model_consensus"],
                "config_id": values["candidate"],
                "params": "frozen published candidate; not retuned in this run",
                "corridor": corridor,
                "eligible_days": np.nan,
                "signals": values["signals"],
                "hits": round(values["signals"] * values["hit_rate"]),
                "hit_rate": values["hit_rate"],
                "baseline_hit_rate": values["hit_rate"] / values["raw_lift"],
                "raw_lift": values["raw_lift"],
                "same_week_lift": values["same_week_lift"],
                "signals_per_week": values["signals_per_week"],
                "mean_regret_bps": np.nan,
                "verdict": "keep_research_candidate",
                "verdict_reason": "Единственная строка, где raw и same-week lift выше 1 на всех пяти коридорах при селективной частоте; нужен новый закрытый holdout.",
            }
        )
    return pd.DataFrame(rows)


def _format_cell(row: pd.Series) -> str:
    lift = row["same_week_lift"]
    frequency = row["signals_per_week"]
    return "—" if pd.isna(lift) else f"{float(lift):.2f} / {float(frequency):.2f}"


def _render_markdown(matrix: pd.DataFrame, details: pd.DataFrame, metadata: dict[str, Any]) -> str:
    lines = [
        "# Матрица «индикатор × валютный коридор»",
        "",
        f"Дата расчёта: **{metadata['created_on']}**. Статус: **исследовательский OOT-бэктест, не production-оценка**.",
        "",
        "## Короткий вывод",
        "",
        "Базовые правила не выбирались по этим результатам: все параметры взяты из заранее существовавшего `configs/grid.json`, а строка семейства — это объединение дат всех его конфигураций. Основной показатель в ячейке — `same-week lift / сигналов в неделю`. Same-week lift сравнивает сигнал со случайным доступным днём той же ISO-недели и поэтому строже raw lift.",
        "",
        "## Итоговая таблица",
        "",
        "| Индикатор | UZS | TJS | KGS | AMD | KZT | Вердикт |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    order = [*FAMILY_LABELS]
    for indicator in order:
        current = matrix.loc[matrix["indicator"].eq(indicator)]
        if current.empty:
            continue
        values = {row["corridor"]: _format_cell(row) for _, row in current.iterrows()}
        reason = str(current.iloc[0]["verdict_reason"])
        lines.append(
            "| " + FAMILY_LABELS[indicator] + " | " + " | ".join(values.get(c, "—") for c in CORRIDORS) + f" | {reason} |"
        )
    lines.extend(
        [
            "",
            "Формат ячейки: **same-week lift / сигналов в неделю**. Число выше 1 означает, что среди срабатываний было больше удачных дат, чем ожидалось при случайном выборе дня в тех же неделях.",
            "",
            "## Полные метрики семейных строк",
            "",
            "| Индикатор | Коридор | Сигналы | Hit rate | Raw lift | Same-week lift | Сигналов/нед. | Средний regret, б.п. |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in matrix.iterrows():
        mean_regret = "—" if pd.isna(row["mean_regret_bps"]) else f"{float(row['mean_regret_bps']):.1f}"
        lines.append(
            f"| {row['indicator_label']} | {row['corridor']} | {int(row['signals'])} | {float(row['hit_rate']):.1%} | {float(row['raw_lift']):.2f} | {float(row['same_week_lift']):.2f} | {float(row['signals_per_week']):.2f} | {mean_regret} |"
        )
    lines.extend(
        [
            "",
            "## Методика",
            "",
            f"- Период: `{metadata['date_from']}` — `{metadata['date_to']}`.",
            f"- Цель: горизонт `{metadata['horizon']}` торговых дней; hit, если ожидание не улучшило курс более чем на `{metadata['tolerance_bps']}` б.п.",
            "- Цена — рублей за единицу валюты получателя; меньше означает лучше для клиента.",
            "- Каждый индикатор в дату `T` получает только срез панели с `known_at <= T`.",
            "- Последние `h` дат всей истории без полного будущего окна исключены из оценки; хвосты промежуточных кварталов не удаляются.",
            "- Параметры не выбирались по OOT. Семейная строка — заранее определённый union всех конфигураций, а детальные результаты каждой конфигурации лежат в соседнем CSV.",
            "- В исходном grid есть сезонность, но нет отдельного индикатора праздников. Поэтому строка не подтверждает и не опровергает праздничную гипотезу.",
            "- Строка модели перенесена из замороженного межкоридорного отчёта и помечена отдельным `row_type`; она не пересчитана rule-grid раннером.",
            "",
            "## Ограничения",
            "",
            "Семейный union отвечает на вопрос «сработало хотя бы одно заранее заданное правило», поэтому частота у него выше, чем у отдельных настроек. Это не выбор лучшего правила и не доказательство статистической значимости. Вердикты опираются на направление raw и same-week lift, частоту и повторяемость результата между пятью коридорами, а не на p-value. В истории уже сравнивалось много гипотез; окончательное подтверждение требует нового будущего или закрытого банковского holdout и реального исполнимого курса.",
            "",
            "## Воспроизведение",
            "",
            "```bash",
            "uv run python -m fxpulse.indicator_matrix",
            "```",
            "",
            "Машиночитаемые файлы:",
            "",
            "- `docs/indicator-corridor-matrix.csv` — строки семейства и замороженной модели;",
            "- `docs/indicator-corridor-matrix-configs.csv` — все конфигурации исходной сетки;",
            "- `docs/indicator-corridor-matrix-meta.json` — параметры, SHA входов и размеры результата.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    raw_dir: Path | str = Path("data/raw"),
    grid_path: Path | str = Path("configs/grid.json"),
    model_reference: Path | str = Path("configs/corridor_model_reference.json"),
    output_dir: Path | str = Path("docs"),
    date_from: str = "2022-01-01",
    date_to: str = "2026-09-02",
    horizon: int = 5,
    tolerance_bps: float = 25.0,
) -> dict[str, Any]:
    families, details = evaluate_rule_grid(
        raw_dir=raw_dir,
        grid_path=grid_path,
        date_from=date_from,
        date_to=date_to,
        horizon=horizon,
        tolerance_bps=tolerance_bps,
    )
    matrix = pd.concat([families, _model_rows(Path(model_reference))], ignore_index=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "indicator-corridor-matrix.csv"
    details_path = output / "indicator-corridor-matrix-configs.csv"
    md_path = output / "indicator-corridor-matrix.md"
    meta_path = output / "indicator-corridor-matrix-meta.json"
    matrix.to_csv(matrix_path, index=False)
    details.to_csv(details_path, index=False)
    metadata = {
        "schema_version": 1,
        "created_on": pd.Timestamp.now(tz="Europe/Moscow").date().isoformat(),
        "date_from": date_from,
        "date_to": date_to,
        "horizon": horizon,
        "tolerance_bps": tolerance_bps,
        "grid_path": str(grid_path),
        "grid_sha256": grid_sha256(grid_path),
        "model_reference_path": str(model_reference),
        "model_reference_sha256": _sha256(Path(model_reference)),
        "raw_cbr_sha256": _sha256(Path(raw_dir) / "cbr_daily.csv"),
        "family_rows": len(families),
        "configuration_rows": len(details),
        "method": "point-in-time family union; no result-driven configuration selection",
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(matrix, details, metadata), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--grid", type=Path, default=Path("configs/grid.json"))
    parser.add_argument("--model-reference", type=Path, default=Path("configs/corridor_model_reference.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument("--from", dest="date_from", default="2022-01-01")
    parser.add_argument("--to", dest="date_to", default="2026-09-02")
    args = parser.parse_args()
    print(json.dumps(run(raw_dir=args.raw_dir, grid_path=args.grid, model_reference=args.model_reference, output_dir=args.output_dir, date_from=args.date_from, date_to=args.date_to), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
