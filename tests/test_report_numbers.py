"""Rubric 4.3: every number in REPORT.md must match results/*.json.

Skipped when results/ is absent, so a fresh clone still passes `make smoke`.
"""
from __future__ import annotations

import csv
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

pytestmark = pytest.mark.skipif(
    not (RESULTS / "runs.csv").exists(),
    reason="no results/ — run the pipeline first (NB1-NB5)",
)


def test_runs_table_matches_runs_csv():
    from tools.report_tables import runs_table

    rows = list(csv.DictReader((RESULTS / "runs.csv").open(encoding="utf-8")))
    md = runs_table()
    for r in rows:
        assert r["run"] in md
        assert str(r["final_loss"]) in md, f"{r['run']} final_loss missing"
        assert str(r["peak_vram_gb"]) in md, f"{r['run']} peak_vram_gb missing"
        assert str(r["trainable_params"]) in md, f"{r['run']} trainable_params missing"


def test_baseline_table_matches_verdict_json():
    from tools.report_tables import baseline_table

    v = json.loads((RESULTS / "verdict.json").read_text(encoding="utf-8"))
    md = baseline_table()
    for row in v["comparison"]:
        assert row["run"] in md
        assert str(row["target"]) in md, f"{row['run']} target missing"


def test_autopsy_table_matches_autopsy_json():
    from tools.report_tables import autopsy_table

    a = json.loads((RESULTS / "autopsy.json").read_text(encoding="utf-8"))
    md = autopsy_table()
    for row in a:
        assert row["run"] in md
        assert str(row["target"]) in md, f"{row['run']} target missing"


def test_check_can_actually_fail():
    """The guard must be able to fail, or it proves nothing."""
    from tools import report_tables

    good = report_tables.load_results(RESULTS)
    assert report_tables.check_missing(report_tables.render_all(good), good) == []

    bad = json.loads(json.dumps(good))
    bad["autopsy"][0]["target"] = 0.4242
    assert report_tables.check_missing(report_tables.render_all(good), bad) != []
