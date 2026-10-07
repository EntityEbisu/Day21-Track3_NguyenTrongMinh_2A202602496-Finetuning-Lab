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
    """The table must carry every run's numbers -- as collapsed by load_results().

    Reading runs.csv raw would include the superseded `correct` row from a resumed
    session and demand a number the table correctly does not show.
    """
    from tools import report_tables

    data = report_tables.load_results(RESULTS)
    md = report_tables.runs_table(data)
    for r in data["runs"]:
        assert r["run"] in md
        assert str(r["final_loss"]) in md, f"{r['run']} final_loss missing"
        assert str(r["peak_vram_gb"]) in md, f"{r['run']} peak_vram_gb missing"
        assert str(r["trainable_params"]) in md, f"{r['run']} trainable_params missing"

    # And the raw file must still be readable (the collapse is a view, not a rewrite).
    raw = list(csv.DictReader((RESULTS / "runs.csv").open(encoding="utf-8")))
    assert len(raw) >= len(data["runs"]), "collapsing must not invent rows"


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


def test_runs_table_collapses_a_repeated_run_to_its_last_row(tmp_path):
    """NB4 appends a fresh row each time a run executes and reads the LAST per key.

    A resumed session retrains `correct` while NB4 skips the three contrasts
    ("skip attn_only: adapters/attn_only/ already trained"), so runs.csv ends up
    holding two `correct` rows. Showing both would put a stale adapter's loss in
    the report table, and the stale adapter no longer exists on disk.
    """
    from tools import report_tables

    (tmp_path / "runs.csv").write_text(
        "run,final_loss,peak_vram_gb,trainable_params\n"
        "correct,0.6274,8.78,32464896\n"
        "attn_only,0.538,8.79,32456704\n"
        "correct,0.6255,8.78,32464896\n",
        encoding="utf-8",
    )
    data = report_tables.load_results(tmp_path)
    md = report_tables.runs_table(data)

    assert md.count("| correct |") == 1, "a repeated run must appear once"
    assert "0.6255" in md, "the LAST row per run is the live one"
    assert "0.6274" not in md, "the superseded row must not be shown"
    assert "attn_only" in md, "collapsing must not drop the other runs"


def test_check_missing_uses_the_collapsed_runs(tmp_path):
    """Otherwise it demands the superseded number be present in the report."""
    from tools import report_tables

    (tmp_path / "runs.csv").write_text(
        "run,final_loss,peak_vram_gb,trainable_params\n"
        "correct,0.6274,8.78,32464896\n"
        "correct,0.6255,8.78,32464896\n",
        encoding="utf-8",
    )
    data = report_tables.load_results(tmp_path)
    missing = report_tables.check_missing("0.6255 32464896 8.78", data)
    assert missing == [], f"stale row leaked into the check: {missing}"

