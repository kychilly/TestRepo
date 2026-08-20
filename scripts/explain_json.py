#!/usr/bin/env python3
"""Create or refresh a plain-English TXT companion for JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbm_study.plain_english import companion_path, explain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON must contain an object: {path}")
        output = companion_path(path)
        output.write_text(explain(payload, source=str(path)), encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
