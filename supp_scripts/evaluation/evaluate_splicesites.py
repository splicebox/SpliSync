#!/usr/bin/env python3
"""
compare_introns.py
==================

Compare a set of *predicted* introns with a set of *reference / actual*
introns and report the number of:

    • TP – present in both prediction and reference  
    • FP – present only in prediction  
    • FN – present only in reference  

Input
-----
Two tab–separated files with (at least) the first two columns describing an
intron (e.g. chromosome and start coordinate).  The script keeps only those
columns, coerces them to string and removes duplicates before comparing.

Example
-------
$ python evaluate_splicesites.py \
      --pred   predicted_introns.tsv \
      --actual reference_introns.tsv

Author: Your Name
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def read_introns(path: Path, key_cols: list[int]) -> pd.DataFrame:
    """
    Read a tab-separated intron file, cast the key columns to str, and drop
    duplicate introns.

    Parameters
    ----------
    path : Path
        Location of the input file.
    key_cols : list[int]
        Column indices that uniquely identify an intron.

    Returns
    -------
    pd.DataFrame
        A frame with only unique introns (w.r.t. `key_cols`).
    """
    df = pd.read_csv(path, sep="\t", header=None)
    df[key_cols] = df[key_cols].astype(str)
    return df.drop_duplicates(subset=key_cols)


def get_counts(
    df_pred: pd.DataFrame, df_actual: pd.DataFrame, key_cols: list[int]
) -> tuple[int, int, int]:
    """
    Compute TP / FP / FN counts.

    Returns
    -------
    tuple[int, int, int]
        (true_positive, false_positive, false_negative)
    """
    merged = pd.merge(
        df_actual[key_cols],      # left
        df_pred[key_cols],        # right
        on=key_cols,
        how="outer",
        indicator=True,
    )
    tp = (merged["_merge"] == "both").sum()
    fn = (merged["_merge"] == "left_only").sum()
    fp = (merged["_merge"] == "right_only").sum()
    return tp, fp, fn


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare predicted introns with a reference set."
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="Tab-separated file with *predicted* introns.",
    )
    parser.add_argument(
        "--actual",
        required=True,
        type=Path,
        help="Tab-separated file with *reference/actual* introns.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    KEY_COLS = [0, 1]  # columns that uniquely define an intron

    df_pred = read_introns(args.pred, KEY_COLS)
    df_actual = read_introns(args.actual, KEY_COLS)

    tp, fp, fn = get_counts(df_pred, df_actual, KEY_COLS)

    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    # convenient one-liner for downstream parsing
    #print(f"{tp} {fp} {fn}")


if __name__ == "__main__":
    main()
