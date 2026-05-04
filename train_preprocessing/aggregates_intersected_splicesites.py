#!/usr/bin/env python3
"""
Aggregates intersected BED/GTF splice site data.

This script processes the output of an intersection operation (e.g., bedtools),
grouping intron features by their parent genomic coordinates and aggregating 
splice site details into comma-separated lists.
"""

import argparse
from pathlib import Path
import pandas as pd


def load_intersect_data(filepath: Path) -> pd.DataFrame:
    """
    Loads the intersected BED/GTF file into a Pandas DataFrame.

    Args:
        filepath (Path): Path to the input intersected file.

    Returns:
        pd.DataFrame: DataFrame containing the parsed genomic features.
    """
    col_names = [
        'chr', 'bed_start', 'bed_end',
        'intron_chr', 'intron_start', 'intron_end', 
        'min_left_block', 'min_right_block', 'labels'
    ]

    df = pd.read_csv(
        filepath, 
        sep='\t', 
        header=None, 
        names=col_names,
        dtype=str
    )
    
    return df


def aggregate_splice_sites(input_file: Path, output_file: Path) -> None:
    """
    Groups the data by genomic coordinates and aggregates intron features.

    Args:
        input_file (Path): Path to the input intersected file.
        output_file (Path): Path to the destination output file.
    """
    df = load_intersect_data(input_file)
    
    grouped = df.groupby(['chr', 'bed_start', 'bed_end'])
    
    # Aggregate intron and block information into comma-separated strings.
    intron_formatted = grouped.agg({
        'intron_start': ','.join,
        'intron_end': ','.join,
        'min_left_block': ','.join,
        'min_right_block': ','.join,
    }).reset_index()
    
    intron_formatted.to_csv(output_file, header=None, index=False, sep='\t')


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate splice site intersections for RNA transcriptomics pipelines."
    )
    parser.add_argument(
        "input_bed",
        type=Path,
        help="Path to the input BED file containing intersected GTF data."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Optional: Path to the output file. Defaults to <input_bed>.intersect.splicesites.bed",
        default=None
    )

    args = parser.parse_args()

    input_file = args.input_bed
    
    if args.output:
        output_file = args.output
    else:
        output_file = input_file.with_name(f"{input_file.name}.intersect.splicesites.bed")

    aggregate_splice_sites(input_file, output_file)


if __name__ == '__main__':
    main()
