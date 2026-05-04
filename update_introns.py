#!/usr/bin/env python3
"""
update_introns.py
=================

Adjust intron start / end coordinates (column 3) in a BED-like file
using positions predicted by a second file.

Input files
-----------
1. intron_file
   id  chrom  coord  is_end  strand
   1   chr1   14829  0       -
   …

2. pred_file
   chrom  coord  score  dummy  strand
   chr1   14830  0.96   0      -
   …

Rules
-----
For every chromosome:
    * Walk through intron coords (sorted) and predicted coords.
    * If |intron_coord - pred_coord| < threshold → snap intron to pred.
    * If the intron row is marked as `is_end==0`, subtract 1.

CLI
---
$ ./update_introns.py introns.tsv preds.tsv -t 21 -o introns.updated.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# IO helpers
def read_introns(path: Path) -> pd.DataFrame:
    """
    Load intron file (5 columns, no header).

    Returns a DataFrame with named columns for clarity.
    """
    cols = ["id", "chrom", "coord", "is_end", "strand"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, dtype={"coord": int})
    df["is_end"] = df["is_end"].astype(bool)
    return df


def read_predictions(path: Path) -> pd.DataFrame:
    """
    Load prediction file (≥2 columns; we only need chrom & coord).

    Returns a DataFrame with 'chrom' and 'coord' columns.
    """
    cols = ["chrom", "coord"]
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=cols)
    df["coord"] = df["coord"].astype(int)
    return df


def write_introns(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame without header / index."""
    df.to_csv(path, sep="\t", header=False, index=False)


# Core algorithm
def _snap_chromosome(
    intron_chr: pd.DataFrame, pred_coords: np.ndarray, threshold: int
) -> pd.DataFrame:
    """Adjust coordinates for a single chromosome group."""
    intron_chr = intron_chr.sort_values("coord").copy()
    values = intron_chr["coord"].values
    is_end = intron_chr["is_end"].values

    i, j = 0, 0
    n_introns = len(values)
    n_preds = len(pred_coords)

    while i < n_introns and j < n_preds:
        dist = values[i] - pred_coords[j]

        # Two consecutive predictions close to the same intron -> start & end pair
        if (
            j + 1 < n_preds
            and abs(values[i] - pred_coords[j]) < threshold
            and abs(values[i] - pred_coords[j + 1]) < threshold
        ):
            values[i] = pred_coords[j] if is_end[i] else pred_coords[j + 1] - 1
            i += 1
        elif abs(dist) < threshold:
            values[i] = pred_coords[j] if is_end[i] else pred_coords[j] - 1
            i += 1
        elif dist < 0:
            # intron < prediction: move intron pointer forward
            i += 1
        else:
            # intron > prediction: move prediction pointer forward
            j += 1

    intron_chr["coord"] = values
    return intron_chr


def update_introns(
    introns: pd.DataFrame, preds: pd.DataFrame, threshold: int
) -> pd.DataFrame:
    """
    Return a new DataFrame where `coord` is adjusted using predictions.
    """
    # Build quick access dict: chrom -> sorted numpy array of prediction coords
    pred_dict: Dict[str, np.ndarray] = {
        chrom: grp["coord"].values for chrom, grp in preds.groupby("chrom")
    }

    updated_chunks: List[pd.DataFrame] = []

    for chrom, intron_grp in introns.groupby("chrom"):
        if chrom not in pred_dict:
            updated_chunks.append(intron_grp.drop(columns="is_end"))
            continue

        snapped = _snap_chromosome(intron_grp, pred_dict[chrom], threshold)
        updated_chunks.append(snapped.drop(columns="is_end"))

    return pd.concat(updated_chunks, ignore_index=True)


# CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Snap intron coords to predictions.")
    p.add_argument("introns", type=Path, help="Input intron file (BED-like, 5 cols)")
    p.add_argument("preds", type=Path, help="Prediction file (≥2 cols: chrom, coord)")
    p.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=21,
        help="Max distance (bp) to consider a match [default: 21]",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: <introns>.updated_<preds>.threshold<N>.tsv)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase verbosity (INFO level logging)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    logging.info("Reading input files …")
    intron_df = read_introns(args.introns)
    pred_df = read_predictions(args.preds)

    logging.info("Updating coordinates …")
    updated = update_introns(intron_df, pred_df, args.threshold)

    # Determine output path if not provided
    if args.output is None:
        pred_base = args.preds.name
        args.output = args.introns.with_name(
            f"{args.introns.name}.updated_{pred_base}.threshold{args.threshold}"
        )

    logging.info("Writing output → %s", args.output)
    write_introns(updated, args.output)


if __name__ == "__main__":
    main()
