#!/bin/bash
# ==============================================================================
# Script: gtf_to_bed12.sh
# Description: Converts a GTF file to BED12 format.
# Dependencies: gtfToGenePred (UCSC Bioinformatics Utilities)
# ==============================================================================

# Exit immediately if a command fails, a variable is unset, or a pipe fails
set -euo pipefail

# 1. Input Validation
if [ "$#" -ne 1 ]; then
    echo "Error: Missing input file."
    echo "Usage: $0 <input.gtf>"
    echo "Example: $0 annotations.gtf"
    exit 1
fi

INPUT_GTF="$1"
TMP_FILE="${INPUT_GTF}.tmp"
# Strip the .gtf extension (if present) and append .bed12
OUTPUT_BED="${INPUT_GTF%.*}.bed12" 

# 2. Dependency Management
# Allows users to define their own path to the tool, defaulting to the system PATH
GTF_TO_GENEPRED="${GTF_TO_GENEPRED:-gtfToGenePred}"

if ! command -v "$GTF_TO_GENEPRED" &> /dev/null; then
    echo "Error: '$GTF_TO_GENEPRED' command not found."
    echo "Please ensure it is installed and in your PATH, or set the GTF_TO_GENEPRED environment variable."
    echo "Download: http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/gtfToGenePred"
    exit 1
fi

# 3. Cleanup Routine
# Automatically delete the temp file when the script exits (even if it crashes)
trap 'rm -f "$TMP_FILE"' EXIT

# 4. Execution
echo "Step 1: Converting GTF to GenePred..."
"$GTF_TO_GENEPRED" -genePredExt -geneNameAsName2 "$INPUT_GTF" "$TMP_FILE"

echo "Step 2: Formatting GenePred to BED12..."
# Reordering columns to match BED12 specifications:
# chrom, txStart, txEnd, name, score(0), strand, cdsStart, cdsEnd, itemRgb(0), exonCount, exonSizes, exonStarts
awk '{print $2"\t"$4"\t"$5"\t"$1"\t0\t"$3"\t"$6"\t"$7"\t0\t"$8"\t"$9"\t"$10}' "$TMP_FILE" > "$OUTPUT_BED"

echo "Success! Output saved to: $OUTPUT_BED"
