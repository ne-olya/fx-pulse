"""Download the open AI-GPR CSV files with atomic local checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

USER_AGENT = "fx-pulse-research/0.1"


def load_config(path: Path | str = Path("configs/ai_gpr.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("AI-GPR config must be preregistered schema_version 1")
    if set(config.get("urls", {})) != set(config.get("raw_files", {})):
        raise ValueError("AI-GPR URL and raw-file keys must match")
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(config: dict[str, Any], *, force: bool = False) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for name, url in config["urls"].items():
        path = Path(config["raw_files"][name])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or force:
            request = urllib.request.Request(str(url), headers={"User-Agent": USER_AGENT})
            temporary = path.with_suffix(path.suffix + ".tmp")
            with urllib.request.urlopen(request, timeout=180) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    metadata = {
        "source_page": config["source_page"],
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "files": files,
    }
    meta_path = Path("data/raw/ai_gpr/download_meta.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ai_gpr.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download(load_config(args.config), force=args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
