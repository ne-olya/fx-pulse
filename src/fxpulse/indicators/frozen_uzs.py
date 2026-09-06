"""Historical point-in-time replay of the frozen UZS model policy.

The official candidate is an annual rolling OOT system, not one estimator
fitted once on the entire history.  A compact replay is therefore the only
honest artifact for historical ``signals_as_of`` calls: it preserves the
score produced by the correct past-only fold and the causal policy decision.
It is not a substitute for a live retraining and feature-serving pipeline.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pandas as pd

from fxpulse.indicators import IndicatorOut, indicator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _load_replay(
    replay_path: str,
    replay_mtime_ns: int,
    manifest_path: str,
    manifest_mtime_ns: int,
) -> pd.DataFrame:
    del replay_mtime_ns, manifest_mtime_ns
    replay_file = Path(replay_path)
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("indicator") != "uzs_frozen_consensus":
        raise ValueError("invalid frozen UZS replay manifest")
    expected_hash = manifest.get("replay_sha256")
    if not expected_hash or _sha256(replay_file) != expected_hash:
        raise ValueError("frozen UZS replay hash differs from its manifest")
    frame = pd.read_csv(replay_file)
    required = {"date", "available_at", "fired", "strength", "signal_tier"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"frozen UZS replay lacks {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="raise").dt.tz_convert(
        "Europe/Moscow"
    )
    if not pd.api.types.is_bool_dtype(frame["fired"]):
        normalized = frame["fired"].astype(str).str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError("frozen UZS replay fired must be boolean")
        frame["fired"] = normalized.eq("true")
    frame["strength"] = pd.to_numeric(frame["strength"], errors="raise")
    if frame["date"].duplicated().any():
        raise ValueError("frozen UZS replay must contain one row per date")
    if not frame["strength"].between(0, 1).all():
        raise ValueError("frozen UZS replay strength must be in [0, 1]")
    if len(frame) != int(manifest.get("replay_rows", -1)):
        raise ValueError("frozen UZS replay row count differs from its manifest")
    return frame.set_index("date").sort_index()


@indicator("uzs_frozen_consensus")
def uzs_frozen_consensus(
    panel: pd.DataFrame,
    *,
    model_path: str,
    manifest_path: str,
) -> IndicatorOut:
    """Return the frozen annual-fold UZS decision for the requested date.

    ``model_path`` retains the task's public config name, but points to a
    compact, hash-verified OOT model replay rather than a single leaky CBM.
    """

    series_id = panel.attrs.get("fxpulse_series_id")
    as_of = panel.attrs.get("fxpulse_as_of")
    if series_id != "CBR:UZS" or as_of is None or panel.empty:
        return IndicatorOut(False, 0.0, {"reason": "not_uzs_or_no_as_of_data"})
    replay_file = Path(model_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not replay_file.exists() or not manifest_file.exists():
        raise FileNotFoundError("frozen UZS replay and manifest must both exist")
    replay = _load_replay(
        str(replay_file),
        replay_file.stat().st_mtime_ns,
        str(manifest_file),
        manifest_file.stat().st_mtime_ns,
    )
    decision_date = pd.Timestamp(as_of).tz_convert("Europe/Moscow").tz_localize(None).normalize()
    if decision_date not in replay.index:
        return IndicatorOut(False, 0.0, {"reason": "date_outside_frozen_replay"})
    row = replay.loc[decision_date]
    if pd.Timestamp(row["available_at"]) > pd.Timestamp(as_of):
        return IndicatorOut(False, 0.0, {"reason": "score_not_available_yet"})
    fired = bool(row["fired"])
    details = {
        "direction": "favorable",
        "consensus": float(row["strength"]),
        "signal_tier": str(row["signal_tier"]),
        "artifact": replay_file.name,
    }
    return IndicatorOut(fired, float(row["strength"]) if fired else 0.0, details)


__all__ = ["uzs_frozen_consensus"]
