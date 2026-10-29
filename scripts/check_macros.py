"""List macros used in paper/*.tex that are not defined in paper/generated/numbers.tex."""
import re
import sys
from pathlib import Path

paper = Path(__file__).resolve().parents[1] / "paper"
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", (paper / "generated" / "numbers.tex").read_text()))
defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", (paper / "preamble.tex").read_text()))
missing = {}
for tex in list(paper.glob("*.tex")):
    for m in re.findall(r"\\([A-Z][A-Za-z]+)", tex.read_text()):
        if m not in defined and m[:1].isupper() and m not in {"MakeUppercase", "Large", "LARGE", "Huge", "TeX", "LaTeX"}:
            missing.setdefault(m, set()).add(tex.name)
for k, v in sorted(missing.items()):
    print(k, sorted(v))
sys.exit(1 if missing else 0)
