#!/usr/bin/env python3
"""
extract_introns_from_bed12.py
===================

Read a BED12 file and convert every exon–exon junction to an “intron‐block”
record of the form

    chr   intron_start   intron_end   min_left_blk_len   min_right_blk_len
          label1,label2,…

where each label is

    <left_exon_len>-<right_exon_len>:<transcript_id>

If multiple transcripts share the same intron, their labels are concatenated
with commas and the *minimum* exon lengths of all transcripts are reported.

The script reproduces the exact output of the original prototype while being
easier to understand and maintain.

Usage
-----
$ ./extract_introns_from_bed12.py input.bed12
# → writes: input.bed12.intron.block.bed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# I/O helpers                                                                 
BED12_COLUMNS = [
    "chr",
    "bed_start",
    "bed_end",
    "name",
    "score",
    "strand",
    "thickStart",
    "thickEnd",
    "reserved",
    "blockCount",
    "blockStarts",
    "blockEnds",
]


def read_bed12(path: Path) -> pd.DataFrame:
    """
    Load a BED12 file into a DataFrame with the correct dtypes.
    """
    dtypes = {
        "chr": str,
        "bed_start": int,
        "bed_end": int,
        "name": str,
        "score": int,
        "strand": str,
        "thickStart": int,
        "thickEnd": int,
        "reserved": str,
        "blockCount": int,
        "blockStarts": str,
        "blockEnds": str,
    }

    return pd.read_csv(path, sep="\t", names=BED12_COLUMNS, dtype=dtypes)


def build_intron_dict(
    df: pd.DataFrame,
) -> Tuple[
    Dict[Tuple[str, str, str], List[str]],
    Dict[Tuple[str, str, str], List[int]],
]:
    """
    Build two dictionaries keyed by (chr, intron_start, intron_end):

    1. labels_dict[key] -> list[str]
       All “left-right:transcript_id” labels encountered for that intron.

    2. min_blocks_dict[key] -> [min_left_block_len, min_right_block_len]
       Running minima of the flanking exon lengths.
    """
    labels_dict: Dict[Tuple[str, str, str], List[str]] = {}
    min_blocks_dict: Dict[Tuple[str, str, str], List[int]] = {}

    # Keep only multi-exon transcripts
    multi_exon_df = df[df["blockCount"] > 1]

    for _, row in multi_exon_df.iterrows():
        chrom = row["chr"]
        name = row["name"]

        # Split block start / end strings, remove trailing empty entry caused
        # by the terminal comma that bedtools often keeps.
        block_starts = [int(x) for x in row["blockStarts"].rstrip(",").split(",")]
        block_ends = [int(x) for x in row["blockEnds"].rstrip(",").split(",")]

        # Iterate over all introns (there are blockCount-1 of them)
        for i in range(row["blockCount"] - 1):
            intron_start = block_ends[i]
            intron_end = block_starts[i + 1]

            left_block_len = block_ends[i] - block_starts[i]
            right_block_len = block_ends[i + 1] - block_starts[i + 1]

            key = (chrom, str(intron_start), str(intron_end))
            label = f"{left_block_len}-{right_block_len}:{name}"

            # Update label list
            labels_dict.setdefault(key, []).append(label)

            # Update minimum block lengths
            mins = min_blocks_dict.setdefault(key, [left_block_len, right_block_len])
            mins[0] = min(mins[0], left_block_len)
            mins[1] = min(mins[1], right_block_len)

    return labels_dict, min_blocks_dict


def write_output(
    out_path: Path,
    labels_dict: Dict[Tuple[str, str, str], List[str]],
    min_blocks_dict: Dict[Tuple[str, str, str], List[int]],
) -> None:
    """
    Write the intron information to *out_path* in the same format
    as the original script.
    """
    with out_path.open("w") as fh:
        for key in labels_dict:
            chrom, intron_start, intron_end = key
            min_left, min_right = min_blocks_dict[key]
            labels = ",".join(labels_dict[key])
            line = (
                f"{chrom}\t{intron_start}\t{intron_end}"
                f"\t{min_left}\t{min_right}\t{labels}\n"
            )
            fh.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract intron information from a BED12 file."
    )
    parser.add_argument(
        "bed12",
        type=Path,
        help="Input BED12 file (e.g. from gtfToGenePred + genePredToBed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bed12_path: Path = args.bed12
    output_path: Path = bed12_path.with_suffix(bed12_path.suffix + ".intron.block.bed")

    df = read_bed12(bed12_path)
    labels_dict, min_blocks_dict = build_intron_dict(df)
    write_output(output_path, labels_dict, min_blocks_dict)


if __name__ == "__main__":
    main()
