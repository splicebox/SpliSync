#!/usr/bin/env python3
"""
Filter a BED12 file to retain only records matching a specific list of target IDs.
"""

import argparse
import pandas as pd
from pathlib import Path

def filter_bed12(bed_path: str, target_ids_path: str, output_path: str) -> None:
    """
    Loads a BED12 file and filters it based on a list of target transcript IDs.
    """
    print(f"Loading BED12: {bed_path}")
    bed_df = pd.read_csv(bed_path, header=None, sep='\t')
    
    print(f"Loading target IDs: {target_ids_path}")
    target_ids_df = pd.read_csv(target_ids_path, header=None, sep='\t')
    target_ids = target_ids_df.tolist()
    
    print(f"Filtering records...")
    filtered_bed = bed_df[bed_df.isin(target_ids)]
    
    filtered_bed.to_csv(output_path, index=False, header=None, sep='\t')
    print(f"Success! Saved {len(filtered_bed)} records to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Filter a BED12 file by a list of target IDs.")
    parser.add_argument("-i", "--input_bed", required=True, help="Path to the input BED12 file.")
    parser.add_argument("-t", "--target_ids", required=True, help="Path to the list of target transcript IDs.")
    parser.add_argument("-o", "--output", help="Path for the output filtered BED12 file. (Optional)")
    
    args = parser.parse_args()
    
    if args.output:
        out_file = args.output
    else:
        bed_path = Path(args.input_bed)
        out_file = f"{bed_path.stem}_isin_txid.bed12"
        
    filter_bed12(args.input_bed, args.target_ids, out_file)

if __name__ == "__main__":
    main()
