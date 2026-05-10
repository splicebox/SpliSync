#!/usr/bin/env python3
"""
correct_bed_blocks.py
=================

Update the exon block sizes (`column 10`) and relative starts (`column 11`)
of a BED12 file using a list of *corrected intron* coordinates.

Input
-----

1. BED12 file (`bed_fn`)
   Example line (tab-separated):
       chr1 11675 14413 uuid 1 + 11675 14413 255,0,0 1 2738 0
   Important columns:
       1  chromStart
       2  chromEnd
       10 blockSizes  (comma-separated list)
       11 chromStarts (comma-separated list)

2. Updated intron file (`updated_intron_fn`)
   Columns (no header):
       id  chrom  coord  strand
   • `id` is the 0-based row index of the transcript in the BED file.
   • For every transcript with N exons there must be 2 × (N-1) rows,
     interleaved as: intron-start, intron-end, intron-start, intron-end …

Algorithm
---------

For each transcript (row *i* in the BED file):

    intron_coords = all rows with id == i, sorted by coord

    For every exon *k* (0-based, except the last exon):
        new_block_size[k]   = intron_start[k] - chromStart - old_block_start[k]
        new_block_start[k+1] = intron_end[k]  - chromStart

    The last exon’s size is recomputed so that:
        new_block_size[last] = chromEnd - chromStart - new_block_start[last]

Output
------

A corrected BED file

Usage
-----

$ ./correct_bed_blocks.py path/to/file.bed path/to/updated_introns.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


#I/O helper functions                              
def read_bed(path: Path) -> pd.DataFrame:
    """
    Read a BED(12) file.  Keep col 10 and 11 as *strings* (comma lists).
    """
    dtype = {10: "string", 11: "string"}
    return pd.read_csv(path, sep="\t", header=None, dtype=dtype)


def read_introns(path: Path) -> pd.DataFrame:
    """
    Read the intron update file.  The first column is used as index (transcript id).
    """
    cols = ["id", "chrom", "coord", "strand"]
    intron_df = pd.read_csv(path, sep="\t", header=None, names=cols, index_col="id")
    # Sort by transcript id then by coord; keeps the required pairing order.
    return intron_df.sort_values(["id", "coord"])


#Core transformation                               
def _parse_comma_list(text: str) -> List[int]:
    """
    Convert a BED comma-separated list into a list of ints.
    Trailing commas are tolerated.
    """
    return [int(x) for x in text.rstrip(",").split(",") if x]


def _join_comma_list(values: List[int]) -> str:
    """Return a comma-separated string (no trailing comma)."""
    return ",".join(map(str, values))


def update_transcript_blocks(
    row: pd.Series, intron_coords: List[int]
) -> tuple[str, str]:
    """
    Given a BED row and a list of new intron coordinates,
    return *new* (blockSizes, chromStarts) strings.
    """
    chrom_start = int(row[1])
    chrom_end = int(row[2])

    block_sizes = _parse_comma_list(row[10])
    block_starts = _parse_comma_list(row[11])

    n_exons = len(block_sizes)
    expected_introns = 2 * (n_exons - 1)
    if expected_introns != len(intron_coords):
        raise ValueError(
            f"Row {row.name}: expected {expected_introns} intron coords, "
            f"got {len(intron_coords)}."
        )

    # Iterate over exons except the last
    for exon_idx in range(n_exons - 1):
        intron_start = intron_coords[2 * exon_idx]
        intron_end = intron_coords[2 * exon_idx + 1]

        block_sizes[exon_idx] = intron_start - chrom_start - block_starts[exon_idx]
        block_starts[exon_idx + 1] = intron_end - chrom_start

    # Recompute size of the last exon
    block_sizes[-1] = chrom_end - chrom_start - block_starts[-1]

    return _join_comma_list(block_sizes), _join_comma_list(block_starts)


def fix_bed_blocks(bed: pd.DataFrame, introns: pd.DataFrame) -> pd.DataFrame:
    """
    Return a *new* BED DataFrame with corrected columns 10 & 11.
    """
    bed = bed.copy()
    for idx, group in introns.groupby(level=0):
        block_sizes_str, block_starts_str = update_transcript_blocks(
            bed.loc[idx], group["coord"].tolist()
        )
        bed.at[idx, 10] = block_sizes_str
        bed.at[idx, 11] = block_starts_str
    return bed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Correct BED block columns.")
    p.add_argument("bed", type=Path, help="Input BED12 file")
    p.add_argument("introns", type=Path, help="Updated intron coordinate file")
    p.add_argument(
        "-o", 
        "--output", 
        type=Path, 
        help="Output file path (default: appends '.corrected.bed' to input file name)"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    bed_df = read_bed(args.bed)
    intron_df = read_introns(args.introns)

    fixed_bed = fix_bed_blocks(bed_df, intron_df)

    # Use specified output path, or default to the original behavior
    if args.output:
        out_path = args.output
    else:
        out_path = args.bed.with_name(f"{args.bed.name}.corrected.bed")

    fixed_bed.to_csv(out_path, sep="\t", header=False, index=False)
    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
