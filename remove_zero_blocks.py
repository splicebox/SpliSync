#!/usr/bin/env python3
"""
remove_zero_blocks.py
=====================

Purpose
-------
Filter a BED12‐like file so that:

1.  Any row that contains a negative block size (i.e. "-" in column 10)
    is completely removed and its *row index* is written to a companion
    `<bed>.filtered_indices` file.

2.  For the remaining rows, exon blocks whose size is **exactly 0** are
   removed.  Columns are then updated:
      • column 9  (blockCount)   – new number of blocks
      • column 10 (blockSizes)   – comma-separated list without zeros
      • column 11 (chromStarts)  – corresponding starts

The cleaned BED is written to  `<bed>.cleaned.bed`.

Usage
-----
$ ./remove_zero_blocks.py input.bed

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


# --------------------------------------------------------------------------- #
#                               Core functions                                #
# --------------------------------------------------------------------------- #
def _strip_commas(text: str) -> List[str]:
    """Split a BED comma-separated field, discarding any trailing empty item."""
    return [x for x in text.rstrip(",").split(",") if x]


def cleanse_row(row: pd.Series) -> pd.Series:
    """
    Remove zero-length blocks from a single BED12 row.

    The caller must guarantee that the row does *not* contain negative sizes.
    """
    block_sizes = _strip_commas(row[10])
    chrom_starts = _strip_commas(row[11])

    if "0" not in block_sizes:
        # Nothing to do
        return row

    # Keep only non-zero blocks
    new_sizes, new_starts = [], []
    for size, start in zip(block_sizes, chrom_starts):
        if size != "0":
            new_sizes.append(size)
            new_starts.append(start)

    row[10] = ",".join(new_sizes)
    row[11] = ",".join(new_starts)
    row[9] = len(new_sizes)  # update blockCount
    return row


def process_bed(bed: pd.DataFrame) -> tuple[pd.DataFrame, List[int]]:
    """
    Return a cleaned BED DataFrame and list of indices that were dropped.
    """
    # 1. Drop rows with negative sizes in column 10
    neg_mask = bed[10].str.contains("-", regex=False)
    removed_idx = bed[neg_mask].index.tolist()
    bed = bed.loc[~neg_mask].copy()

    # 2. Remove zero-length blocks within each remaining row
    bed = bed.apply(cleanse_row, axis=1)

    return bed, removed_idx


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Remove zero-length BED blocks.")
    p.add_argument("bed", type=Path, help="Input BED12 file")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Keep col 10 & 11 as strings – they are comma-separated lists
    dtype = {10: "string", 11: "string"}
    bed_df = pd.read_csv(args.bed, sep="\t", header=None, dtype=dtype)

    cleaned_df, filtered_idx = process_bed(bed_df)

    # Write cleaned BED
    out_bed = args.bed.with_name(f"{args.bed.stem}.cleaned.bed")
    cleaned_df.to_csv(out_bed, sep="\t", header=False, index=False)
    print(f"Cleaned BED written → {out_bed}")

    # Write indices of filtered-out rows (one per line)
    idx_path = args.bed.with_name(f"{args.bed.name}.filtered_indices")
    with idx_path.open("w") as fh:
        fh.writelines(f"{i}\n" for i in filtered_idx)
    print(f"Filtered indices written → {idx_path}")


if __name__ == "__main__":
    main()
