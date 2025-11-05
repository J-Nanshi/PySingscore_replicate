#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
singscore_single_sample.py

Compute singscore for a SINGLE sample against an up- (and optional down-) gene set.
Implements the rank-based method from Foroutan et al., BMC Bioinformatics 2018.

Inputs:
  - Expression file (CSV/TSV) with columns [gene, value] for ONE sample (TPM recommended).
  - Text file of up-genes (one symbol per line).
  - Optional text file of down-genes.

Outputs (printed as JSON and/or saved to --out):
  - up_score_norm_centered, down_score_norm_centered, total_score
  - diagnostic counts (N_total, N_up_mapped, N_down_mapped)

References:
  Foroutan M. et al., 2018, BMC Bioinformatics 19:404 (singscore method).
  Vignette: https://davislaboratory.github.io/singscore/articles/singscore.html
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, Optional, Set

import pandas as pd
import numpy as np


def _read_expr(expr_path: Path) -> pd.DataFrame:
    """Read a single-sample expression table; return DataFrame with columns ['gene','value']."""
    # Let pandas sniff the delimiter; works for CSV/TSV
    df = pd.read_csv(expr_path, sep=None, engine="python")
    # Standardize column names
    cols = {c.lower().strip(): c for c in df.columns}
    gene_col_candidates = [k for k in cols.keys() if k in {"gene", "genes", "gene_symbol", "symbol", "hgnc_symbol"}]
    value_col_candidates = [k for k in cols.keys() if k in {"tpm", "fpkm", "rpkm", "rsem", "expr", "expression", "value"}]

    if not gene_col_candidates:
        # fallback = first column
        gene_col = df.columns[0]
    else:
        gene_col = cols[gene_col_candidates[0]]

    if not value_col_candidates:
        # fallback = second column
        if len(df.columns) < 2:
            raise ValueError("Expression file must have at least two columns: gene and value.")
        value_col = df.columns[1]
    else:
        value_col = cols[value_col_candidates[0]]

    out = df[[gene_col, value_col]].rename(columns={gene_col: "gene", value_col: "value"}).copy()
    # Clean gene symbols, drop missing values
    out["gene"] = out["gene"].astype(str).str.strip()
    out = out.dropna(subset=["gene", "value"])
    # Keep only finite numeric values
    out = out[np.isfinite(out["value"].values)]
    # Ensure unique genes: keep the max value per gene (or mean—choice does not affect ranks much)
    out = out.groupby("gene", as_index=False)["value"].max()
    return out


def _read_gene_list(path: Path) -> Set[str]:
    """Read a newline-separated gene list (HGNC symbols)."""
    if path is None:
        return set()
    genes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if g and not g.startswith("#"):
                genes.append(g)
    # Normalize case to match expression (we'll upper both sides)
    return set(genes)


def _rank_series(values: pd.Series) -> pd.Series:
    """
    Rank genes by increasing abundance (1 = lowest). Average ranks for ties (as in Wilcoxon).
    """
    # pandas rank: method="average", ascending=True -> 1..N
    return values.rank(method="average", ascending=True)


def _singscore_direction(
    ranks: pd.Series,
    gene_set: Set[str],
    direction: str,
) -> Tuple[Optional[float], int]:
    """
    Compute normalized, centered score for one direction (up or down).

    Steps (Foroutan et al. 2018):
    - rank genes by increasing abundance
    - for DOWN set, use reversed ranks: r' = N_total - r + 1
    - mean rank over the set -> S_dir
    - normalize to theoretical [S_min, S_max]
    - center at 0 by subtracting 0.5 (so each dir is in [-0.5, +0.5])
    """
    assert direction in {"up", "down"}
    # Intersect set with ranked index
    present = set(ranks.index) & set(gs_upper(gene_set))
    n_dir = len(present)
    n_total = ranks.shape[0]
    if n_dir == 0:
        return None, 0

    # Extract ranks for present genes
    sub = ranks.loc[sorted(present)]
    if direction == "down":
        sub = (n_total - sub + 1)  # reverse ranks for expected down-regulated genes

    S_dir = float(sub.mean())
    S_min = (n_dir + 1) / 2.0
    S_max = (2 * n_total - n_dir + 1) / 2.0
    # Normalize to [0,1]
    S_norm = (S_dir - S_min) / (S_max - S_min)
    # Center to [-0.5, +0.5]
    S_norm_centered = S_norm - 0.5
    return S_norm_centered, n_dir


def gs_upper(gs: Iterable[str]) -> Set[str]:
    return {g.upper() for g in gs}


def main():
    ap = argparse.ArgumentParser(description="Compute singscore for a single sample.")
    ap.add_argument("--expr", required=True, type=Path, help="Expression file (CSV/TSV) with columns [gene, value].")
    ap.add_argument("--up", required=True, type=Path, help="Text file with up-gene symbols (one per line).")
    ap.add_argument("--down", type=Path, default=None, help="Optional text file with down-gene symbols.")
    ap.add_argument("--min-genes", type=int, default=5, help="Minimum mapped genes required per direction to report.")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to write JSON output.")
    args = ap.parse_args()

    # Load data
    expr = _read_expr(args.expr)
    expr["gene_upper"] = expr["gene"].str.upper()
    expr = expr.drop_duplicates(subset=["gene_upper"]).set_index("gene_upper")

    up_set = _read_gene_list(args.up)
    down_set = _read_gene_list(args.down) if args.down else set()

    # Rank within sample
    ranks = _rank_series(expr["value"])

    # Directional scores
    up_score, n_up = _singscore_direction(ranks, up_set, "up")
    down_score, n_down = (None, 0)
    if down_set:
        down_score, n_down = _singscore_direction(ranks, down_set, "down")

    # Enforce min-genes threshold
    if up_score is None or n_up < args.min_genes:
        up_score_out = None
    else:
        up_score_out = float(up_score)

    down_score_out = None
    if down_set:
        if down_score is not None and n_down >= args.min_genes:
            down_score_out = float(down_score)

    # Total score: sum available centered components
    parts = [s for s in [up_score_out, down_score_out] if s is not None]
    total_score = float(np.sum(parts)) if parts else None

    result = {
        "n_total_genes": int(ranks.shape[0]),
        "n_up_mapped": int(n_up),
        "n_down_mapped": int(n_down) if down_set else 0,
        "up_score_norm_centered": up_score_out,     # in [-0.5, +0.5]
        "down_score_norm_centered": down_score_out, # in [-0.5, +0.5]
        "total_score": total_score,                  # in [-1.0, +1.0] if both directions available
        "notes": (
            "Scores are normalized to theoretical min/max per sample and centered at 0; "
            "total_score is the sum of centered up/down components."
        ),
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
