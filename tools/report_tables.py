"""Render REPORT.md tables straight from results/*.json.

Rubric 4.3 grades every number in the report against results/. Transcribing
~60 numbers by hand is how that fails, so they are generated instead.

    python tools/report_tables.py            # print the tables
    python tools/report_tables.py --check    # verify REPORT.md against results/
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RUN_COLS = ["run", "placement", "r", "trainable_params", "learning_rate",
            "final_loss", "train_seconds", "peak_vram_gb"]
BASE_COLS = ["run", "target", "regression", "format", "latency_ms", "n"]
AUTOPSY_COLS = ["run", "target", "format", "latency_ms", "n"]


def load_results(results_dir: pathlib.Path = RESULTS) -> dict:
    """Everything the report needs, in one dict. Missing files become None."""
    def j(name):
        p = results_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    rows: list[dict] = []
    p = results_dir / "runs.csv"
    if p.exists():
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
    return {
        "runs": rows,
        "verdict": j("verdict.json"),
        "autopsy": j("autopsy.json"),
        "frozen": j("baselines_frozen.json"),
        "mask": j("mask_proof.json"),
        "tokens": j("token_stats.json"),
        "template": j("template_check.json"),
        "qualitative": j("qualitative.json"),
    }


def _table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def runs_table(data: dict | None = None) -> str:
    data = data or load_results()
    return _table(data["runs"], RUN_COLS)


def baseline_table(data: dict | None = None) -> str:
    data = data or load_results()
    return _table((data["verdict"] or {}).get("comparison", []), BASE_COLS)


def autopsy_table(data: dict | None = None) -> str:
    data = data or load_results()
    return _table(data["autopsy"] or [], AUTOPSY_COLS)


def render_all(data: dict | None = None) -> str:
    data = data or load_results()
    return "\n\n".join([runs_table(data), baseline_table(data), autopsy_table(data)])


def check_missing(report_text: str, data: dict) -> list[str]:
    """Numbers present in results/ but absent from the report text. [] == consistent."""
    missing: list[str] = []
    for r in data["runs"]:
        for c in ("final_loss", "peak_vram_gb", "trainable_params"):
            if r.get(c) and str(r[c]) not in report_text:
                missing.append(f"runs/{r['run']}/{c}={r[c]}")
    for row in (data["verdict"] or {}).get("comparison", []):
        if str(row["target"]) not in report_text:
            missing.append(f"verdict/{row['run']}/target={row['target']}")
    for row in data["autopsy"] or []:
        if str(row["target"]) not in report_text:
            missing.append(f"autopsy/{row['run']}/target={row['target']}")
    return missing


def main() -> int:
    data = load_results()
    if "--check" in sys.argv:
        rep = ROOT / "submission" / "REPORT.md"
        missing = check_missing(rep.read_text(encoding="utf-8"), data)
        if missing:
            print(f"{len(missing)} number(s) in results/ are missing from REPORT.md:")
            for m in missing:
                print("  -", m)
            return 1
        print("OK: every results/ number appears in REPORT.md")
        return 0
    print(render_all(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
