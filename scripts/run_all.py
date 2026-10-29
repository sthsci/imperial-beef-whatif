"""Run every model and regenerate results/, figures/ and paper/generated/.

    python scripts/run_all.py            # full run (about a minute)
    python scripts/run_all.py --quick    # fewer Monte Carlo draws, for testing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beefwhatif import analysis, figures, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer Monte Carlo draws")
    args = ap.parse_args()
    t0 = time.time()
    kw = {"n_mc": 2000, "n_campus": 300} if args.quick else {}
    R = analysis.run(**kw)
    print(f"analysis done in {time.time() - t0:.1f} s")
    figures.make_all(R)
    print(f"figures done in {time.time() - t0:.1f} s")
    report.write_all(R)
    print(f"report files done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
