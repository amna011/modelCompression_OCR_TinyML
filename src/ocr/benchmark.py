from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_size_kb(entry: Dict) -> None:
    if "size_kb" in entry:
        return
    tflite_path = entry.get("tflite_path")
    if tflite_path and os.path.exists(tflite_path):
        entry["size_kb"] = round(os.path.getsize(tflite_path) / 1024.0, 2)


def _format_table(rows: List[Dict]) -> str:
    headers = [
        "Model",
        "Compression",
        "Size (KB)",
        "RAM (KB)",
        "Latency (ms)",
        "FPS",
        "CER",
        "Exact Match",
        "Deployable",
    ]

    table = [headers]
    for r in rows:
        table.append(
            [
                str(r.get("model", "")),
                str(r.get("compression", "")),
                str(r.get("size_kb", "")),
                str(r.get("ram_kb", "")),
                str(r.get("latency_ms", "")),
                str(r.get("fps", "")),
                str(r.get("cer", "")),
                str(r.get("exact_match", "")),
                str(r.get("deployable", "")),
            ]
        )

    col_widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines = []
    for idx, row in enumerate(table):
        padded = [row[i].ljust(col_widths[i]) for i in range(len(headers))]
        lines.append(" | ".join(padded))
        if idx == 0:
            lines.append("-+-".join("-" * w for w in col_widths))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render unified benchmark table.")
    parser.add_argument("--results", required=True, help="JSON file containing benchmark entries.")
    args = parser.parse_args()

    rows = _load_results(args.results)
    for entry in rows:
        _maybe_size_kb(entry)
    table = _format_table(rows)
    print(table)


if __name__ == "__main__":
    main()
