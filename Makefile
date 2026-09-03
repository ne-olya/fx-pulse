# Reuse the project-managed uv when it exists; a fresh clone still falls back
# to a user-installed `uv` so that `make setup` can create the environment.
UV ?= $(if $(wildcard .venv/bin/uv),.venv/bin/uv,uv)
PYTHON ?= $(UV) run python

DATA_FROM ?= 2018-01-01
DATA_TO ?= $(shell date +%F)
# MOEX keeps a finite rolling intraday history. Override this when a longer
# interval is available, for example: make data CANDLE_FROM=2026-01-01.
CANDLE_FROM ?= 2026-08-01
SECIDS ?= CNYRUB_TOM USD000UTSTOM KZTRUB_TOM

.PHONY: setup data test

setup:
	$(UV) sync --all-groups

data:
	$(PYTHON) -m fxpulse.data.cbr --from $(DATA_FROM) --to $(DATA_TO)
	$(PYTHON) -m fxpulse.data.moex daily --from $(DATA_FROM) --to $(DATA_TO) --secids $(SECIDS)
	$(PYTHON) -m fxpulse.data.moex candles --from $(CANDLE_FROM) --to $(DATA_TO) --secids $(SECIDS)

test:
	$(PYTHON) -m pytest
