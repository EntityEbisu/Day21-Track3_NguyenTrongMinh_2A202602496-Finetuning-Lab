#!/usr/bin/env python3
"""Normalize typographic punctuation in the graded deliverables to ASCII.

**Why.** `submission/REPORT.md` is graded by cross-checking its numbers against
`results/*.json` (rubric 4.3). A grader that greps for `-0.202` finds nothing when
the report writes U+2212 MINUS SIGN, and `>= 0.02` finds nothing when the report
writes U+2265. The number is right and the search still fails — the same class of
silent mismatch this lab is about.

**What is NOT touched.** Vietnamese diacritics (the report is written in
Vietnamese by requirement), and the emoji/status marks the lab's own template
uses. Only punctuation that carries a numeric or relational meaning is rewritten.

    python tools/normalize_ascii.py            # report what would change
    python tools/normalize_ascii.py --write    # apply
"""
from __future__ import annotations

import pathlib
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]

TARGETS = [
    ROOT / "submission" / "REPORT.md",
    ROOT / "submission" / "REFLECTION.md",
    ROOT / "docs" / "RUNS.md",
]

# Only characters that carry numeric/relational meaning. Deliberately NOT here:
# Vietnamese diacritics (U+00A0..U+1EFF letters), U+00A7 SECTION SIGN (the lab's
# own docs cite deck sections as §11.2), and the emoji the template ships with.
REPLACEMENTS: dict[str, str] = {
    "\u2212": "-",     # MINUS SIGN              -> hyphen-minus
    "\u2013": "-",     # EN DASH                 -> hyphen-minus
    "\u2014": "-",     # EM DASH                 -> hyphen-minus
    "\u2265": ">=",    # GREATER-THAN OR EQUAL   -> >=
    "\u2264": "<=",    # LESS-THAN OR EQUAL      -> <=
    "\u00d7": "x",     # MULTIPLICATION SIGN     -> x
    "\u0394": "delta", # GREEK CAPITAL DELTA     -> delta
    "\u2192": "->",    # RIGHTWARDS ARROW        -> ->
    "\u21d2": "=>",    # RIGHTWARDS DOUBLE ARROW -> =>
    "\u2026": "...",   # HORIZONTAL ELLIPSIS     -> ...
    "\u00b7": "*",     # MIDDLE DOT              -> *
    "\u201c": '"',     # LEFT DOUBLE QUOTATION
    "\u201d": '"',     # RIGHT DOUBLE QUOTATION
    "\u2018": "'",     # LEFT SINGLE QUOTATION
    "\u2019": "'",     # RIGHT SINGLE QUOTATION
}


def normalize_text(text: str) -> tuple[str, dict[str, int]]:
    """Return (normalized, counts-per-replacement)."""
    counts: dict[str, int] = {}
    for src, dst in REPLACEMENTS.items():
        n = text.count(src)
        if n:
            counts[src] = n
            text = text.replace(src, dst)
    return text, counts


def describe(ch: str) -> str:
    try:
        return unicodedata.name(ch)
    except ValueError:                                    # pragma: no cover
        return "?"


def main() -> int:
    write = "--write" in sys.argv
    changed = 0
    for path in TARGETS:
        if not path.exists():                             # pragma: no cover
            print(f"skip (absent): {path.relative_to(ROOT)}")
            continue
        original = path.read_text(encoding="utf-8")
        new, counts = normalize_text(original)
        label = path.relative_to(ROOT)
        if not counts:
            print(f"{label}: already ASCII-clean")
            continue
        changed += 1
        total = sum(counts.values())
        print(f"{label}: {total} replacement(s)")
        for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    U+{ord(src):04X} {describe(src):<26} x{n}  ->  {REPLACEMENTS[src]!r}")
        if write:
            path.write_text(new, encoding="utf-8")
    if write:
        print(f"\nwrote {changed} file(s)")
    else:
        print("\n(dry run — pass --write to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
