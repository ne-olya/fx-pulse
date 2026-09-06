"""Export a hash-verified historical replay of the frozen UZS candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fxpulse.uzs_final_experiment import _period_metrics, load_config


EXPECTED = {
    "signals": 140,
    "raw_lift": 1.56,
    "same_week_lift": 1.25,
    "signals_per_week": 0.61,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(frame: pd.DataFrame, frozen: dict[str, Any]) -> pd.Series:
    return (
        frame["feature_set"].eq(frozen["feature_set"])
        & frame["model_variant"].eq(frozen["model_variant"])
        & frame["policy"].eq(frozen["policy"])
    )


def _verify_metrics(metrics: dict[str, float | int], *, tolerance: float = 0.025) -> None:
    if int(metrics["signals"]) != EXPECTED["signals"]:
        raise RuntimeError(
            f"frozen replay has {metrics['signals']} strong signals; expected {EXPECTED['signals']}"
        )
    for name in ("raw_lift", "same_week_lift", "signals_per_week"):
        if abs(float(metrics[name]) - float(EXPECTED[name])) > tolerance:
            raise RuntimeError(
                f"frozen replay {name}={metrics[name]:.6f} differs from published {EXPECTED[name]}"
            )


def export_replay(
    *,
    artifact_dir: Path | str,
    experiment_config: Path | str,
    frozen_config: Path | str,
    replay_path: Path | str,
    manifest_path: Path | str,
    graph_path: Path | str,
    graph_from: str = "2025-09-02",
    graph_to: str = "2026-09-02",
) -> dict[str, Any]:
    artifact = Path(artifact_dir)
    policy_path = artifact / "policy_scores.csv.gz"
    signals_path = artifact / "signals.csv.gz"
    run_meta_path = artifact / "run_meta.json"
    config = load_config(experiment_config)
    frozen = json.loads(Path(frozen_config).read_text(encoding="utf-8"))
    scores = pd.read_csv(policy_path, parse_dates=["timestamp"])
    signals = pd.read_csv(signals_path, parse_dates=["timestamp"])
    selected_scores = scores.loc[_identity(scores, frozen)].copy()
    selected_signals = signals.loc[
        _identity(signals, frozen)
        & signals["signal_tier"].eq(frozen["primary_signal_tier"])
    ].copy()
    if selected_scores["timestamp"].duplicated().any() or selected_signals["timestamp"].duplicated().any():
        raise RuntimeError("frozen UZS inputs contain duplicate dates")
    metrics = _period_metrics(selected_scores, selected_signals, list(map(int, config["test_years"])))
    _verify_metrics(metrics)

    fired_dates = set(selected_signals["timestamp"])
    tier_by_date = selected_signals.set_index("timestamp")["signal_tier"].to_dict()
    replay = selected_scores.sort_values("timestamp", kind="mergesort").copy()
    replay["date"] = replay["timestamp"].dt.date.astype(str)
    replay["available_at"] = replay["date"] + "T20:00:00+03:00"
    replay["fired"] = replay["timestamp"].isin(fired_dates)
    replay["strength"] = replay["consensus_score"].clip(0, 1)
    replay["signal_tier"] = replay["timestamp"].map(tier_by_date).fillna("none")
    if "raw_model_score" not in replay:
        replay["raw_model_score"] = replay["score"]
    columns = [
        "date",
        "available_at",
        "fired",
        "strength",
        "signal_tier",
        "test_year",
        "raw_model_score",
        "own_rank",
        "other_rank_mean",
        "other_rank_std",
        "other_positive_share",
        "consensus_score",
    ]
    replay = replay[columns]
    destination = Path(replay_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    replay.to_csv(destination, index=False, compression={"method": "gzip", "mtime": 0})

    graph = replay.loc[
        replay["fired"]
        & pd.to_datetime(replay["date"]).between(pd.Timestamp(graph_from), pd.Timestamp(graph_to))
    ][["date"]].copy()
    graph["corridor"] = "RUB->UZS"
    graph_destination = Path(graph_path)
    graph_destination.parent.mkdir(parents=True, exist_ok=True)
    graph.to_csv(graph_destination, index=False)

    manifest = {
        "schema_version": 1,
        "indicator": "uzs_frozen_consensus",
        "artifact_kind": "historical_annual_fold_oot_replay",
        "research_status": "frozen_exploratory_candidate_not_production_ready",
        "candidate": {
            key: frozen[key]
            for key in (
                "target_corridor",
                "horizon_trading_days",
                "evaluation_tolerance_bps",
                "feature_set",
                "model_variant",
                "policy",
                "primary_signal_tier",
                "training_window_years",
                "training_label",
                "seeds",
            )
        },
        "replay_rows": len(replay),
        "replay_from": replay["date"].min(),
        "replay_to": replay["date"].max(),
        "replay_sha256": _sha256(destination),
        "strong_signal_rows": int(replay["fired"].sum()),
        "graph_signal_rows": len(graph),
        "graph_from": graph_from,
        "graph_to": graph_to,
        "verified_metrics": {key: metrics[key] for key in EXPECTED},
        "source_artifacts": {
            "policy_scores": {"path": str(policy_path), "sha256": _sha256(policy_path)},
            "signals": {"path": str(signals_path), "sha256": _sha256(signals_path)},
            "run_meta": {"path": str(run_meta_path), "sha256": _sha256(run_meta_path)},
            "experiment_config": {
                "path": str(experiment_config),
                "sha256": _sha256(Path(experiment_config)),
            },
            "frozen_config": {"path": str(frozen_config), "sha256": _sha256(Path(frozen_config))},
        },
        "availability_contract": "daily replay row becomes usable at 20:00 Europe/Moscow on its date",
        "limitation": "This file replays past OOT decisions. Live dates require the full point-in-time feature and retraining service.",
    }
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/uzs_final_experiment_20260905"))
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/uzs_final_experiment.json"))
    parser.add_argument("--frozen-config", type=Path, default=Path("configs/uzs_final_frozen_candidate.json"))
    parser.add_argument("--replay", type=Path, default=Path("models/uzs_frozen_oot_replay.csv.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("models/uzs_frozen_oot_replay.manifest.json"))
    parser.add_argument("--graph", type=Path, default=Path("docs/prototype/signals-uzs.csv"))
    args = parser.parse_args()
    manifest = export_replay(
        artifact_dir=args.artifact_dir,
        experiment_config=args.experiment_config,
        frozen_config=args.frozen_config,
        replay_path=args.replay,
        manifest_path=args.manifest,
        graph_path=args.graph,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
