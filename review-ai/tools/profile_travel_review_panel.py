# -*- coding: utf-8 -*-
"""Profile all available 01_Travel_Review.{YYYY}.tsv files (Phase 0).

Thin wrapper around parse_travel_review_panel.py that writes the canonical
panel profile JSON used by the research plan.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools/parse_travel_review_panel.py"),
        "--report",
        "--report_path",
        str(PROJECT_ROOT / "results/travel_review_panel_profile.json"),
    ]
    raise SystemExit(subprocess.call(cmd, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
