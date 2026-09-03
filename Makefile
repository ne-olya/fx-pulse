# Reuse the project-managed uv when it exists; a fresh clone still falls back
# to a user-installed `uv` so that `make setup` can create the environment.
UV ?= $(if $(wildcard .venv/bin/uv),.venv/bin/uv,uv)
PYTHON ?= $(UV) run python

DATA_FROM ?= 2018-01-01
DATA_TO ?= $(shell date +%F)
# MOEX keeps a finite rolling intraday history. Override this when a longer
# interval is available, for example: make data CANDLE_FROM=2026-01-01.
CANDLE_FROM ?= 2026-08-01
HYPOTHESIS_CANDLE_FROM ?= 2018-09-03
HYPOTHESIS_CANDLE_TO ?= $(DATA_TO)
UNIVERSE_FROM ?= 2021-09-03
UNIVERSE_TO ?= $(DATA_TO)
SECIDS ?= CNYRUB_TOM USD000UTSTOM KZTRUB_TOM

.PHONY: setup data data-quality test backtest hypotheses hypotheses-data universe-data universe-data-all rule-selection interpretable-models local-minimum-models

setup:
	$(UV) sync --all-groups

data:
	$(PYTHON) -m fxpulse.data.cbr --from $(DATA_FROM) --to $(DATA_TO)
	$(PYTHON) -m fxpulse.data.moex daily --from $(DATA_FROM) --to $(DATA_TO) --secids $(SECIDS)
	$(PYTHON) -m fxpulse.data.moex candles --from $(CANDLE_FROM) --to $(DATA_TO) --secids $(SECIDS)

data-quality:
	$(PYTHON) -m fxpulse.data.quality

test:
	$(PYTHON) -m pytest

backtest: data-quality
	$(PYTHON) -m fxpulse.backtest

hypotheses:
	$(PYTHON) -m fxpulse.hypotheses

hypotheses-data:
	$(PYTHON) -m fxpulse.data.moex candles --from $(HYPOTHESIS_CANDLE_FROM) --to $(HYPOTHESIS_CANDLE_TO) --secids CNYRUB_TOM --output data/raw/moex_cny_candles_8y.csv --chunk-days 31

# Fixed SECIDs plus candidates: the five-year audit/backfill dataset.
universe-data:
	$(PYTHON) -m fxpulse.data.universe --from $(UNIVERSE_FROM) --to $(UNIVERSE_TO)

# Also resolves BR and GOLD at every historical close. This is deliberately
# slower: it makes two point-in-time ISS requests for every weekday and root.
universe-data-all:
	$(PYTHON) -m fxpulse.data.universe --from $(UNIVERSE_FROM) --to $(UNIVERSE_TO) --include-planned

rule-selection:
	$(PYTHON) -m fxpulse.rule_selection

interpretable-models:
	$(PYTHON) -m fxpulse.interpretable_models

local-minimum-models:
	$(PYTHON) -m fxpulse.local_minimum_models
