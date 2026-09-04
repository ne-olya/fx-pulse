# Reuse the project-managed uv when it exists; a fresh clone still falls back
# to a user-installed `uv` so that `make setup` can create the environment.
UV ?= $(if $(wildcard .venv/bin/uv),.venv/bin/uv,uv)
PYTHON ?= $(UV) run python

DATA_FROM ?= 2018-01-01
DATA_TO ?= $(shell date +%F)
# Keep the default 10-minute download small for a quick first run. For
# CNYRUB_TOM, ISS was checked to contain intraday candles from 2013-04-15.
CANDLE_FROM ?= 2026-08-01
EXPERIMENT_FROM ?= 2018-01-01
EXPERIMENT_TO ?= 2026-09-02
HYPOTHESIS_CANDLE_FROM ?= 2018-09-03
HYPOTHESIS_CANDLE_TO ?= $(DATA_TO)
UNIVERSE_FROM ?= 2021-09-03
UNIVERSE_TO ?= $(DATA_TO)
SECIDS ?= CNYRUB_TOM USD000UTSTOM KZTRUB_TOM

.PHONY: setup data experiment-data hourly-factor-data recipient-bank-data interest-data brent-data experiment next-hypotheses adaptive-threshold recipient-leg-experiment hourly-factor-experiment regret-formulation-experiment multi-horizon-experiment temporal-sequence-experiment meta-labeling-experiment value-downside-experiment calendar-experiment holiday-experiment interest-rate-experiment regime-policy-experiment path-label-experiment garch-gate-experiment event-sampling-experiment ranking-experiment training-history-experiment technical-rule-experiment momentum-streak-experiment optimal-stopping-simulation brent-experiment shared-head-experiment conformal-abstention-experiment target-rate-simulation research-panel data-quality test backtest hypotheses hypotheses-data universe-data universe-data-all rule-selection interpretable-models local-minimum-models regret-benchmark boosting-calibration hybrid-targets dual-regret-policy

setup:
	$(UV) sync --all-groups

data:
	$(PYTHON) -m fxpulse.data.cbr --from $(DATA_FROM) --to $(DATA_TO)
	$(PYTHON) -m fxpulse.data.moex daily --from $(DATA_FROM) --to $(DATA_TO) --secids $(SECIDS)
	$(PYTHON) -m fxpulse.data.moex candles --from $(CANDLE_FROM) --to $(DATA_TO) --secids $(SECIDS)

# Separate files for the experiment described in docs/O_experiment-plan.md.
experiment-data:
	$(PYTHON) -m fxpulse.data.moex daily --from $(EXPERIMENT_FROM) --to $(EXPERIMENT_TO) --secids CNYRUB_TOM --output data/raw/moex_cny_daily.csv
	$(PYTHON) -m fxpulse.data.moex candles --from $(EXPERIMENT_FROM) --to $(EXPERIMENT_TO) --secids CNYRUB_TOM --interval 60 --chunk-days 31 --output data/raw/moex_cny_60m.csv

hourly-factor-data:
	$(PYTHON) -m fxpulse.data.moex candles --from $(EXPERIMENT_FROM) --to $(EXPERIMENT_TO) --secids CNYRUB_TOM USD000UTSTOM KZTRUB_TOM GLDRUB_TOM SLVRUB_TOM --interval 60 --chunk-days 366 --output data/raw/moex_cets_factors_60m.csv

recipient-bank-data:
	$(PYTHON) -m fxpulse.data.recipient_banks --cbr-dates data/raw/cbr_daily.csv --output data/raw/recipient_bank_daily.csv

interest-data:
	$(PYTHON) -m fxpulse.data.cbr_key_rate --from $(EXPERIMENT_FROM) --to $(EXPERIMENT_TO)

brent-data:
	$(PYTHON) -m fxpulse.data.fred --from 2010-01-01 --to $(EXPERIMENT_TO) --output data/raw/fred_brent_daily.csv

experiment:
	$(PYTHON) -m fxpulse.experiment

next-hypotheses:
	$(PYTHON) -m fxpulse.next_hypotheses --config configs/next_hypotheses.json --artifact-dir artifacts/next_hypotheses/wave_1

adaptive-threshold:
	$(PYTHON) -m fxpulse.adaptive_threshold --config configs/adaptive_threshold.json --artifact-dir artifacts/next_hypotheses/adaptive_threshold

recipient-leg-experiment:
	$(PYTHON) -m fxpulse.recipient_leg_experiment --config configs/recipient_leg_experiment.json --artifact-dir artifacts/next_hypotheses/recipient_legs

hourly-factor-experiment:
	$(PYTHON) -m fxpulse.hourly_factor_experiment --config configs/hourly_factor_experiment.json --artifact-dir artifacts/next_hypotheses/hourly_factors

regret-formulation-experiment:
	$(PYTHON) -m fxpulse.regret_formulation_experiment --config configs/regret_formulation_experiment.json --artifact-dir artifacts/next_hypotheses/regret_formulations

multi-horizon-experiment:
	$(PYTHON) -m fxpulse.multi_horizon_experiment --config configs/multi_horizon_experiment.json --artifact-dir artifacts/next_hypotheses/multi_horizon

temporal-sequence-experiment:
	$(PYTHON) -m fxpulse.temporal_sequence_experiment --config configs/temporal_sequence_experiment.json --artifact-dir artifacts/next_hypotheses/temporal_sequence

meta-labeling-experiment:
	$(PYTHON) -m fxpulse.meta_labeling_experiment --config configs/meta_labeling_experiment.json --artifact-dir artifacts/next_hypotheses/meta_labeling

value-downside-experiment:
	$(PYTHON) -m fxpulse.value_downside_experiment --config configs/value_downside_experiment.json --artifact-dir artifacts/next_hypotheses/value_downside

calendar-experiment:
	$(PYTHON) -m fxpulse.calendar_experiment --config configs/calendar_experiment.json --artifact-dir artifacts/next_hypotheses/calendar

holiday-experiment:
	$(PYTHON) -m fxpulse.holiday_experiment --config configs/holiday_experiment.json --artifact-dir artifacts/next_hypotheses/holidays

interest-rate-experiment:
	$(PYTHON) -m fxpulse.interest_rate_experiment --config configs/interest_rate_experiment.json --artifact-dir artifacts/next_hypotheses/interest_rate

regime-policy-experiment:
	$(PYTHON) -m fxpulse.regime_policy_experiment --config configs/regime_policy_experiment.json --artifact-dir artifacts/next_hypotheses/regime_policy

path-label-experiment:
	$(PYTHON) -m fxpulse.path_label_experiment --config configs/path_label_experiment.json --artifact-dir artifacts/next_hypotheses/path_labels

garch-gate-experiment:
	$(PYTHON) -m fxpulse.garch_gate_experiment --config configs/garch_gate_experiment.json --artifact-dir artifacts/next_hypotheses/garch_gate

event-sampling-experiment:
	$(PYTHON) -m fxpulse.event_sampling_experiment --config configs/event_sampling_experiment.json --artifact-dir artifacts/next_hypotheses/event_sampling

ranking-experiment:
	$(PYTHON) -m fxpulse.ranking_experiment --config configs/ranking_experiment.json --artifact-dir artifacts/next_hypotheses/ranking

training-history-experiment:
	$(PYTHON) -m fxpulse.training_history_experiment --config configs/training_history_experiment.json --artifact-dir artifacts/next_hypotheses/training_history

technical-rule-experiment:
	$(PYTHON) -m fxpulse.technical_rule_experiment --config configs/technical_rule_experiment.json --artifact-dir artifacts/next_hypotheses/technical_rules

momentum-streak-experiment:
	$(PYTHON) -m fxpulse.technical_rule_experiment --config configs/momentum_streak_experiment.json --artifact-dir artifacts/next_hypotheses/momentum_streak

optimal-stopping-simulation:
	$(PYTHON) -m fxpulse.optimal_stopping_simulation --config configs/optimal_stopping_simulation.json --artifact-dir artifacts/next_hypotheses/optimal_stopping

brent-experiment:
	$(PYTHON) -m fxpulse.brent_experiment --config configs/brent_experiment.json --artifact-dir artifacts/next_hypotheses/brent

shared-head-experiment:
	$(PYTHON) -m fxpulse.shared_head_experiment --config configs/shared_head_experiment.json --artifact-dir artifacts/next_hypotheses/shared_head

conformal-abstention-experiment:
	$(PYTHON) -m fxpulse.conformal_abstention_experiment --config configs/conformal_abstention_experiment.json --artifact-dir artifacts/next_hypotheses/conformal

target-rate-simulation:
	$(PYTHON) -m fxpulse.target_rate_simulation --config configs/target_rate_simulation.json --artifact-dir artifacts/next_hypotheses/target_rate

research-panel:
	$(PYTHON) -m fxpulse.data.assemble

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

regret-benchmark:
	$(PYTHON) -m fxpulse.regret_benchmark

boosting-calibration:
	$(PYTHON) -m fxpulse.boosting_calibration

hybrid-targets:
	$(PYTHON) -m fxpulse.hybrid_targets

dual-regret-policy:
	$(PYTHON) -m fxpulse.dual_regret_policy
