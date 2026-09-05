# FX Pulse

Ищем удачные моменты для трансграничных переводов по открытым данным о валютных курсах.

Постановка задачи — в [TASK.md](TASK.md).

## Документы

### Продукт

| Файл | Что внутри |
|---|---|
| [docs/audit-2026-09-05.md](docs/audit-2026-09-05.md) | Аудит материалов против постановки, ответов кейсодателя и критериев организаторов; список задач со статусами |
| [docs/corridor-criteria.md](docs/corridor-criteria.md) | Критерии выбора коридора, зафиксированные до получения матрицы моделей |
| [docs/value-and-abtest.md](docs/value-and-abtest.md) | Ценность в числах: ощутимость выгоды, доход банка, дизайн A/B-теста и план пилота |
| [docs/prototype-rationale.md](docs/prototype-rationale.md) | Обоснование прототипа: текущий путь, шесть механик с критериями выбора, обзор практик со ссылками |
| [docs/prefill-concept.md](docs/prefill-concept.md) | Предзаполнение экрана перевода: развилки, данные, рекомендуемый вариант, что нужно для внедрения |
| [docs/scenarios-to-texts.md](docs/scenarios-to-texts.md) | Сценарий → формулировка пуша: что реально срабатывает и один шаблон текста |
| [docs/send-timing.md](docs/send-timing.md) | Когда отправляем пуш: готовность сигнала, три варианта, часовые пояса |
| [docs/push-texts.md](docs/push-texts.md) | Библиотека текстов, запрещённые формулировки, результаты прогона на персонах |
| [docs/persona-and-communication.md](docs/persona-and-communication.md) | Портрет отправителя и ролевая модель для симуляции |
| [docs/market-and-persona.md](docs/market-and-persona.md) | Объёмы по коридорам и портрет отправителя с подтверждением каждого вывода |
| [docs/reference-user-portrait.md](docs/reference-user-portrait.md) | Средний чек и число операций по коридорам из данных ЦБ РФ |
| [docs/bigtech-migrant-ux.md](docs/bigtech-migrant-ux.md) | Как большие компании упрощают приложения и пуши для мигрантов |
| [docs/prototype/](docs/prototype/) | Кликабельный прототип клиентского пути на интерфейсе Альфа-мобайла |
| [docs/presentation/](docs/presentation/) | Презентации проекта и генератор |
| [docs/presentation-plan.md](docs/presentation-plan.md) | План работ под нарратив защиты, по слайдам |
| [docs/product-backlog.md](docs/product-backlog.md) | Продуктовый бэклог: последовательность задач |
| [docs/benchmark.md](docs/benchmark.md) | Как задачу «сейчас удачный момент» решают в финтехе, travel, e-commerce и энергетике |
| [docs/qa-kejsodatel.md](docs/qa-kejsodatel.md) | Вопросы кейсодателю и ответы, три раунда |
| [docs/project-description.md](docs/project-description.md) | Полное описание проекта |
| [docs/product-materials.md](docs/product-materials.md) | Продуктовая постановка и выводы для пользовательского сценария |

### Исследование

| Файл | Что внутри |
|---|---|
| [docs/plan.md](docs/plan.md) | План решения: рамка кейса, принятые решения по сигнальной модели, этапы |
| [docs/prototype-brief.md](docs/prototype-brief.md) | Рабочее задание на первый прототип конвейера: контракты модулей, метрики, порядок работ |
| [docs/alternatives.md](docs/alternatives.md) | Дополнения и альтернативы: трендовые сценарии, волатильность, заявка по целевому курсу, rule-play, Rust — с вердиктами и быстрыми проверками |
| [docs/hypothesis-leadlag-findings.md](docs/hypothesis-leadlag-findings.md) | Проверка гипотезы о лид-лаге: механика расчёта курсов ЦБ, отрицательный результат по базовым индикаторам |
| [docs/O_news-experiment-plan.md](docs/O_news-experiment-plan.md) | Простой план новостного эксперимента: GDELT, защита от временного лика, часовые и дневные признаки |
| [docs/hypothesis-results.md](docs/hypothesis-results.md) | Реестр и результаты дополнительных проверок на пятилетнем OOT-периоде |
| [docs/moex-universe.md](docs/moex-universe.md) | Каркас межрыночного universe MOEX и гейты перед массовым backfill |
| [docs/rule-selection.md](docs/rule-selection.md) | Nested walk-forward поиск rule-based и интерпретируемых ML-сигналов по межрыночному universe |
| [docs/local-minimum-models.md](docs/local-minimum-models.md) | Проверка рекомендаций о будущем минимуме CNY/RUB на горизонтах 1/5/20 дней |
| [docs/benchmark.md](docs/benchmark.md) | Как задачу «сейчас удачный момент» решают в финтехе, travel, e-commerce и энергетике |
| [docs/product-materials.md](docs/product-materials.md) | Продуктовая постановка и выводы для пользовательского сценария |
| [docs/O_additional-materials-interim.md](docs/O_additional-materials-interim.md) | Самодостаточные дополнительные материалы для промежуточной сдачи |
| [docs/O_labeling.md](docs/O_labeling.md) | Схема разметки, целевые переменные и выбор дневной/часовой гранулярности |
| [docs/O_experiment-plan.md](docs/O_experiment-plan.md) | Короткий словарь и план эксперимента с сигнальной моделью |
| [docs/O_experiment-results.md](docs/O_experiment-results.md) | Данные и честный итог первого дневного/часового эксперимента |
| [docs/O_final-research-plan.md](docs/O_final-research-plan.md) | Единый итог всех проверок, готовые данные и приоритетные следующие гипотезы |
| [docs/O_hypothesis-log.md](docs/O_hypothesis-log.md) | Короткий журнал заранее зафиксированных новых проверок |
| [docs/O_robust-innovation-results.md](docs/O_robust-innovation-results.md) | Перепроверка прежних победителей, 21 зафиксированный вариант и честный итог |
| [docs/amd-feature-enrichment.md](docs/amd-feature-enrichment.md) | Абляция 305 рыночных признаков вокруг основного AMD h=5 кандидата |
| [docs/uzs-enriched-autoresearch.md](docs/uzs-enriched-autoresearch.md) | Воспроизводимый UZS h=5 эксперимент, 94 локальных признака и 10 итераций авторесерча |
| [docs/O_uzs-final-experiment-plan.md](docs/O_uzs-final-experiment-plan.md) | Заранее зафиксированный план финальной проверки UZS |
| [docs/O_uzs-final-results.md](docs/O_uzs-final-results.md) | Итог UZS: выбранная модель, новости, защита от переобучения и клиентский путь |
| [docs/O_best-model.md](docs/O_best-model.md) | Короткая карточка замороженной модели RUB→UZS |

## Данные

Только открытые и воспроизводимые источники. Выгрузки лежат в `data/` и **не коммитятся** — в репозитории загрузчики, а не данные.

| Источник | Что берём |
|---|---|
| ЦБ РФ, `XML_dynamic.asp` | Дневные официальные курсы: USD, EUR, CNY, TJS, UZS, KGS, KZT, AMD |
| MOEX ISS | `CNYRUB_TOM`, `USD000UTSTOM`, `KZTRUB_TOM` — дневные и внутридневные свечи, объёмы, число сделок |
| Нацбанки стран-получателей | Уже скачаны USD и RUB НБ Казахстана/ЦБ Узбекистана; KGS/AMD/TJS пока в плане |

Точные эндпоинты и схемы файлов — в [docs/prototype-brief.md](docs/prototype-brief.md), раздел 3.

## Запуск

Окружение: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/).

```bash
make setup
```

Первый воспроизводимый слой запускается так:

```bash
make data       # ЦБ, дневной MOEX и 10-минутные свечи MOEX
make data-quality
make test       # включая 50 срезов против заглядывания вперёд
make backtest   # signals.csv, metrics.csv и run_meta.json
```

`make data` загружает данные в игнорируемую Git папку `data/raw/`. По умолчанию
дневная история запрашивается с 2018-01-01, а 10-минутная — только с
2026-08-01, чтобы быстрый запуск не создавал большой файл. Интервал можно
изменить явно:

```bash
make data CANDLE_FROM=2026-01-01 DATA_TO=2026-09-02
```

Отдельный дневной и часовой эксперимент:

```bash
make experiment-data  # снимок CNY/RUB с 2018 года
make experiment       # разметка, walk-forward и таблицы результатов
```

Новые данные и зарегистрированные проверки:

```bash
make hourly-factor-data          # CNY/USD/KZT/золото/серебро, час
make recipient-bank-data         # НБ Казахстана/ЦБ Узбекистана с кэшем ответов
make research-panel
make next-hypotheses             # 397 дневных задач на пяти коридорах
make adaptive-threshold          # causal trailing threshold
make recipient-leg-experiment   # независимые USD/local ноги
make hourly-factor-experiment    # внешние факторы до CNY-свечи
make regret-formulation-experiment
make multi-horizon-experiment
make temporal-sequence-experiment
make meta-labeling-experiment
make value-downside-experiment
make calendar-experiment
make interest-rate-experiment
make regime-policy-experiment
make path-label-experiment
make garch-gate-experiment
make event-sampling-experiment
make holiday-experiment
make ranking-experiment
make training-history-experiment
make technical-rule-experiment
make momentum-streak-experiment
make optimal-stopping-simulation
make brent-data && make brent-experiment
make shared-head-experiment
make conformal-abstention-experiment
make target-rate-simulation
make uzs-final-experiment        # 175 финальных UZS-комбинаций
make uzs-final-selection         # воспроизвести frozen-выбор и robustness-аудит
```

Панель загружается через `fxpulse.panel.load_panel`. Она физически исключает
строки с `known_at > as_of`; поддерживаются `CBR:<CCY>`, дневной
`MOEX:<SECID>` и `MOEX10M:<SECID>`. Это основа теста против заглядывания
вперёд. `signals_as_of(T, config)` — единственная точка расчёта сигналов; его
же вызывает walk-forward-раннер.

`make data-quality` формирует [docs/data-quality.md](docs/data-quality.md).
`make backtest` разворачивает все 37 комбинаций из
[configs/grid.json](configs/grid.json), считает их на одном фиксинге ЦБ
(`CBR:TJS`) и одном закрытии MOEX (`MOEX:CNYRUB_TOM`) и пишет игнорируемые Git
артефакты в `artifacts/`:

- `signals.csv` — срабатывания в контрактной схеме;
- `metrics.csv` — разрезы «конфигурация × ряд × горизонт × квартал OOT»;
- `run_meta.json` — SHA-256 сетки, диапазоны данных и параметры прогона.

Это намеренно узкий базовый прогон: все пять коридоров ЦБ и остальные инструменты
MOEX уже поддержаны в панели. Их можно добавить в проверочный запуск без смены
логики, например:

```bash
./.venv/bin/uv run python -m fxpulse.backtest \
  --series 'RUB->TJS=CBR:TJS' \
  --series 'RUB->UZS=CBR:UZS' \
  --series 'RUB->CNY=MOEX:CNYRUB_TOM'
```

Для каждого квартального out-of-time блока метки, пересекающие его начало,
вычищаются на `h` наблюдений; такой же буфер после блока фиксируется как embargo.
Порогов, обучаемых на итоговой метрике, в прототипе нет — сетка зарегистрирована
до прогона.

## Межрыночный universe

Стартовый реестр валют, металлов, индексов, акций и planned фьючерсов лежит в
[configs/moex_universe.json](configs/moex_universe.json). Загрузчик создаёт
неизменяемый снапшот в `data/raw/moex_universe/`: им можно пользоваться только
после появления `manifest.json`, поэтому частично скачанные данные не будут
приняты за готовые. Для пятилетнего audit/backfill фиксированных инструментов
и кандидатов:

```bash
make universe-data UNIVERSE_FROM=2021-09-03 UNIVERSE_TO=2026-09-02
```

Для него же вместе с непрерывными Brent и Gold:

```bash
make universe-data-all UNIVERSE_FROM=2021-09-03 UNIVERSE_TO=2026-09-02
```

Второй вариант существенно медленнее: для каждого буднего дня он выбирает
контракт по ликвидности, доступной именно в тот день, и фиксирует roll без
заглядывания вперёд. Получить статус `ready` для использования в модели можно
только после проверки истории, единицы котировки, ликвидности и `known_at`.
Контракт и следующий порядок работы описаны в
[docs/moex-universe.md](docs/moex-universe.md).

## Поиск межрыночных сигналов

После `make universe-data` можно запускать два независимых, интерпретируемых
поиска на самом длинном manifest-gated снапшоте:

```bash
make rule-selection       # 240 прозрачных правил «фактор × return × хвост»
make interpretable-models # две ridge-logistic scorecard-модели с коэффициентами
make local-minimum-models # сравнение 8 моделей для будущих минимумов 1/5/20 дней
```

Оба процесса ежеквартально выбирают конфигурацию только на expanding
horizon-purged train-prefix, проверяют её на следующем квартале и применяют
хронологический cap не более двух срабатываний в неделю. Частота `[0.5, 2]`
в неделю — исследовательский гейт текущего поиска; `no_send` сохраняется, если
ни одно правило не проходит train-проверку. Результаты, коэффициенты scorecard
и диагностика кучности пишутся в `artifacts/rule_selection/` и
`artifacts/interpretable_models/`.

`make local-minimum-models` проверяет более прямую продуктовую цель для
покупателя валюты: окажется ли текущий закрывающий курс не выше любого курса в
следующих 1, 5 или 20 торговых наблюдениях. Основной сценарий рассматривает
все доступные дни; факт, что курс уже является минимумом прошлого окна,
поступает как объяснимый диагностический признак, а не как обязательный фильтр.
Будущий минимум используется только как ретроспективная метка интересной точки
для обучения и оценки. Модели и
порог выбираются на внутреннем хронологическом validation-отрезке, затем
переобучаются на полном прошлом и проверяются на следующем квартале.
Артефакты пишутся в `artifacts/local_minimum_models/`; методика — в
[docs/local-minimum-models.md](docs/local-minimum-models.md).

## Проверка гипотез

Зарегистрированные гипотезы запускаются одной командой после загрузки
пятилетнего intraday-источника CNY/RUB:

```bash
make hypotheses-data  # месячные чанки с retry; 2018-09-03 → сегодня
make hypotheses
```

Раннер пишет `artifacts/hypotheses/metrics.csv`, включая единый пятилетний и
годовые out-of-time разрезы, а также SHA-256
`configs/hypotheses.json` в `run_meta.json`. H1–H3/H6–H10 — фиксированные
дневные правила; H11/H12 — ежемесячно переобучаемые ridge-logit модели с
purging; H13–H15 — дневные правила по диапазону и активности торгов. H4
ежемесячно переобучает фильтр времени сессии на предыдущих 24 месяцах. H5
намеренно имеет статус `not_testable`, пока не передан разрешённый источник
исполнимых котировок приложения. Отсутствие такой котировки — результат
проверки, а не замена её рыночной ценой MOEX. Вердикты и критерии отбора — в
[docs/hypothesis-results.md](docs/hypothesis-results.md).

## Ограничения

* Никаких персональных данных и внутренних данных банка — только открытые источники.
* Никакого заглядывания вперёд: сигнал на дату `T` считается только по данным, доступным на `T`.
* Курс ЦБ не является курсом исполнения. Все метрики посчитаны на публичном ряду, клиент переводит по курсу приложения.
