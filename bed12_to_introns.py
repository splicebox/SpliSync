#!/usr/bin/env python3
"""
bed12_to_introns.py
===================

Convert a BED12 file into a 5-column table that marks the beginning (flag=0)
and end (flag=1) of every intron for every transcript.

Output columns
--------------

1. record_id   – index (row number) in the original BED file
2. chrom       – chromosome / contig
3. coordinate  – 0-based coordinate
4. flag        – 0 = intron start, 1 = intron end
5. strand      – "+" or "-"

Usage
-----

$ python bed12_to_introns.py example.bed > example.intron_marks.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def load_bed(path: Path, *, columns: int = 12) -> pd.DataFrame:
    """
    Load a BED(+) file into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        Path to the BED file.
    columns : int, optional
        Expected number of columns (default: 12).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the BED file with no header.
    """
    dtype = {i: "string" for i in range(columns)}
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        usecols=range(columns),
        dtype=dtype,
    )


def iter_intron_markers(df: pd.DataFrame) -> Iterable[Tuple[int, str, int, int, str]]:
    """
    Iterate over the DataFrame and yield intron start/end coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        BED12 data.

    Yields
    ------
    Tuple[int, str, int, int, str]
        (row_index, chrom, coordinate, flag, strand)
    """
    for idx, row in df.iterrows():
        block_count = int(row[9])
        if block_count <= 1:
            # Single-exon features have no introns.
            continue

        chrom = row[0]
        chrom_start = int(row[1])
        strand = row[5]

        # BED12 stores sizes/starts as comma-separated strings ending with a comma.
        block_sizes = [int(x) for x in row[10].rstrip(",").split(",")]
        block_starts = [int(x) for x in row[11].rstrip(",").split(",")]

        for exon_idx in range(block_count - 1):
            intron_start = chrom_start + block_starts[exon_idx] + block_sizes[exon_idx]
            intron_end = chrom_start + block_starts[exon_idx + 1]

            yield (idx, chrom, intron_start, 0, strand)
            yield (idx, chrom, intron_end, 1, strand)


def write_output(
    markers: Iterable[Tuple[int, str, int, int, str]], handle=sys.stdout
) -> None:
    """
    Write intron markers to an open file handle (default: STDOUT).
    """
    for record in markers:
        handle.write("\t".join(map(str, record)) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="bed12_to_introns",
        description="Generate intron start (flag=0) and end (flag=1) coordinates "
        "from a BED12 file.",
    )
    parser.add_argument("bedfile", type=Path, help="Input BED12 file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write result to this file instead of STDOUT",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = load_bed(args.bedfile)
    markers = iter_intron_markers(df)

    if args.output:
        with args.output.open("w") as fh:
            write_output(markers, fh)
    else:
        write_output(markers)


if __name__ == "__main__":
    main()
