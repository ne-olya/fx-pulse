# FX Pulse

Ищем удачные моменты для трансграничных переводов по открытым данным о валютных курсах.

Постановка задачи — в [TASK.md](TASK.md).

## Документы

| Файл | Что внутри |
|---|---|
| [docs/plan.md](docs/plan.md) | План решения: рамка кейса, принятые решения по сигнальной модели, этапы |
| [docs/prototype-brief.md](docs/prototype-brief.md) | Рабочее задание на первый прототип конвейера: контракты модулей, метрики, порядок работ |
| [docs/hypothesis-leadlag-findings.md](docs/hypothesis-leadlag-findings.md) | Проверка гипотезы о лид-лаге: механика расчёта курсов ЦБ, отрицательный результат по базовым индикаторам |
| [docs/benchmark.md](docs/benchmark.md) | Как задачу «сейчас удачный момент» решают в финтехе, travel, e-commerce и энергетике |
| [docs/qa-kejsodatel.md](docs/qa-kejsodatel.md) | Вопросы кейсодателю и ответы |

## Данные

Только открытые и воспроизводимые источники. Выгрузки лежат в `data/` и **не коммитятся** — в репозитории загрузчики, а не данные.

| Источник | Что берём |
|---|---|
| ЦБ РФ, `XML_dynamic.asp` | Дневные официальные курсы: USD, EUR, CNY, TJS, UZS, KGS, KZT, AMD |
| MOEX ISS | `CNYRUB_TOM`, `USD000UTSTOM`, `KZTRUB_TOM` — дневные и внутридневные свечи, объёмы, число сделок |
| Нацбанки стран-получателей | Официальные курсы USD/XXX: НБ РК (RSS), ЦБ РУз (JSON), НБ КР (XML), ЦБ РА (JSON), НБТ (HTML) |

Точные эндпоинты и схемы файлов — в [docs/prototype-brief.md](docs/prototype-brief.md), раздел 3.

## Запуск

Окружение: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Текущее состояние — разведочные скрипты в `scripts/`:

```bash
uv run python scripts/fetch_cbr.py     # выгрузка курсов ЦБ
uv run python scripts/fetch_hist.py    # выгрузка MOEX ISS
uv run python scripts/leadlag.py       # проверка лид-лага
uv run python scripts/signal_test.py   # базовые индикаторы
uv run python scripts/wf_test.py       # walk-forward прогон
```

Пакет `src/fxpulse/` с воспроизводимым конвейером и целями `make data / make test / make backtest` собирается по [docs/prototype-brief.md](docs/prototype-brief.md).

## Ограничения

* Никаких персональных данных и внутренних данных банка — только открытые источники.
* Никакого заглядывания вперёд: сигнал на дату `T` считается только по данным, доступным на `T`.
* Курс ЦБ не является курсом исполнения. Все метрики посчитаны на публичном ряду, клиент переводит по курсу приложения.
