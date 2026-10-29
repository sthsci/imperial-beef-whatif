"""Derive Taste Imperial meat purchase volumes from Imperial College London FOI releases.

Usage:
    python scripts/foi_meat_purchases.py IMPFOI-24-576.xlsx IMPFOI-25-493.xlsx

The two workbooks (food purchase line items, one sheet per academic year) are
published on Imperial's FOI disclosure log (Estates and Facilities):
https://www.imperial.ac.uk/about/university-secretary/institutional-compliance-and-risk-management/request-report-and-access/disclosure-log/imperials-estates-and-facilities/

For every purchase line the pack weight is parsed from the "Additional
Information" column (e.g. "4 x 2.5kg") and multiplied by the order quantity.
Lines whose pack weight cannot be parsed are counted but excluded, so the
totals are lower bounds (chicken has the most unparsed lines). Output is the
CSV stored in data/imperial_foi_meat.csv.
"""
from __future__ import annotations

import re
import sys

import openpyxl

BEEF_WORDS = r"\bbeef|brisket|pastrami|bresaola|salt beef|corned|wagyu|chuck|sirloin|ribeye|rib eye|rump"
NOT_BEEF = (r"tomato|beefeater|quorn|vegan|vegetarian|meat free|oumph|gravy|stock|bouillon|jus|chips|"
            r"fries|crisps|monster munch|seasoning|flavour|marmite|bovril|oxo|noodle")

GROUPS = {
    "beef_kg": lambda n, c: bool(re.match(r"beef", c, re.I)) or (
        bool(re.search(BEEF_WORDS, n, re.I)) and not re.search(NOT_BEEF, f"{n} {c}", re.I)),
    "venison_kg": lambda n, c: bool(re.search(r"venison", f"{n} {c}", re.I)) and not re.search(r"boar", n, re.I),
    "lamb_kg": lambda n, c: bool(re.match(r"lamb", c, re.I)),
    "pork_kg": lambda n, c: bool(re.match(r"pork", c, re.I)),
    "chicken_kg": lambda n, c: bool(re.match(r"chicken \(|chicken$|prepared poultry|poultry", c, re.I)),
}


def pack_kg(info) -> float | None:
    """kg per ordered unit parsed from free text such as '6 x 1kg' or '500g'."""
    if info is None:
        return None
    s = str(info).strip().lower().replace("kgs", "kg")
    if s in ("kg", "price per kg", "per kg", "1 kg", "1kg", "1 x kg"):
        return 1.0
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s*x\s*)?(?:(\d+(?:\.\d+)?)\s*x\s*)?(\d+(?:\.\d+)?)\s*(kg|g|gm|gr)\b.*", s)
    if not m:
        return None
    a = float(m.group(1)) if m.group(1) else 1.0
    b = float(m.group(2)) if m.group(2) else 1.0
    w = float(m.group(3)) * (1.0 if m.group(4) == "kg" else 1e-3)
    return a * b * w


def sheets(path: str) -> dict[str, list[tuple]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        hdr = next((r for r in rows[:10] if r and "Name" in r and "Product Category" in r), None)
        if hdr is None:
            continue
        start = rows.index(hdr) + 1
        ni, ci = hdr.index("Name"), hdr.index("Product Category")
        qi, ai = hdr.index("Order Quantity"), hdr.index("Additional Information")
        out[ws.title] = [(str(r[ni]), str(r[ci]), r[qi], r[ai]) for r in rows[start:] if r and r[ni] is not None]
    return out


def main(paths: list[str]) -> None:
    print("sheet," + ",".join(GROUPS) + ",unparsed_lines")
    for path in paths:
        for title, recs in sheets(path).items():
            totals = dict.fromkeys(GROUPS, 0.0)
            unparsed = 0
            for name, cat, qty, info in recs:
                for key, match in GROUPS.items():
                    if match(name, cat):
                        kg = pack_kg(info)
                        if kg is None or qty is None:
                            unparsed += 1
                        else:
                            totals[key] += float(qty) * kg
                        break
            print(f"{title}," + ",".join(f"{totals[k]:.1f}" for k in GROUPS) + f",{unparsed}")


if __name__ == "__main__":
    main(sys.argv[1:])
