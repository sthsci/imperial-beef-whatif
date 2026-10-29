"""Loading and Monte Carlo sampling of model parameters.

Every uncertain scalar in ``data/parameters.toml`` is a TOML table with a
central ``value`` and, optionally, ``low``/``high`` bounds plus a ``dist`` key
that says how Monte Carlo draws are generated:

* ``fixed``      - always the central value;
* ``uniform``    - uniform on [low, high];
* ``tri``        - triangular with mode = value on [low, high];
* ``lognormal``  - median = value, with [low, high] read as a 90% interval.

Each entry also carries ``unit``, ``source`` (a BibTeX key in
``paper/references.bib``) and a free-text ``note`` so that the parameter table
in the Supplementary Information can be generated straight from this file.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
GENERATED = ROOT / "paper" / "generated"

Z90 = 1.6448536269514722  # standard-normal quantile bounding a central 90% interval


@dataclass(frozen=True)
class Param:
    name: str
    value: float
    low: float | None = None
    high: float | None = None
    dist: str = "fixed"
    unit: str = ""
    source: str = ""
    note: str = ""

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` Monte Carlo samples from the parameter's distribution."""
        if self.dist == "fixed" or self.low is None or self.high is None:
            return np.full(n, float(self.value))
        if self.dist == "uniform":
            return rng.uniform(self.low, self.high, n)
        if self.dist == "tri":
            return rng.triangular(self.low, self.value, self.high, n)
        if self.dist == "lognormal":
            sigma = (np.log(self.high) - np.log(self.low)) / (2 * Z90)
            return self.value * np.exp(sigma * rng.standard_normal(n))
        raise ValueError(f"unknown distribution {self.dist!r} for {self.name}")


class Params:
    """Dotted-path access to ``data/parameters.toml``."""

    def __init__(self, path: Path = DATA / "parameters.toml"):
        with open(path, "rb") as fh:
            self.raw = tomllib.load(fh)

    def node(self, dotted: str):
        node = self.raw
        for part in dotted.split("."):
            node = node[part]
        return node

    def __getitem__(self, dotted: str) -> Param:
        node = self.node(dotted)
        if isinstance(node, dict):
            return Param(name=dotted, **node)
        return Param(name=dotted, value=float(node))

    def v(self, dotted: str) -> float:
        """Central value of a parameter."""
        return float(self[dotted].value)

    def sample(self, dotted: str, rng: np.random.Generator, n: int) -> np.ndarray:
        return self[dotted].sample(rng, n)

    def table(self) -> pd.DataFrame:
        """Flatten every scalar parameter into one table (for the SI)."""
        rows: list[dict] = []

        def walk(node, prefix):
            if isinstance(node, dict) and "value" in node and not isinstance(node["value"], dict):
                rows.append({"name": prefix, **node})
                return
            if isinstance(node, dict):
                for key, child in node.items():
                    walk(child, f"{prefix}.{key}" if prefix else key)

        walk(self.raw, "")
        return pd.DataFrame(rows)


def load_csv(name: str, index: str | None = None) -> pd.DataFrame:
    """Read a CSV from ``data/``, skipping ``#`` comment lines."""
    df = pd.read_csv(DATA / name, comment="#", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()
    return df.set_index(index) if index else df
