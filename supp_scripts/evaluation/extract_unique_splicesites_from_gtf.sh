#!/usr/bin/env bash

# ==============================================================================
# Description: Uses AGAT to explicitly annotate introns in a GTF, then extracts 
#              a unique list of all intron start and end coordinates.
#
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.gtf>"
    exit 1
fi

INPUT_GTF="$1"

# Extract base name to keep output filenames clean (e.g., "sample.gtf" -> "sample")
PREFIX=$(basename "${INPUT_GTF}" .gtf)

INTRON_GFF="${PREFIX}.with_introns.gff"
INTRON_GTF="${PREFIX}.with_introns.gtf"
FINAL_OUT="${PREFIX}.unique_splicesites.txt"

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "[1/3] Adding intron features using AGAT..."
agat_sp_add_introns.pl --gff "${INPUT_GTF}" -o "${INTRON_GFF}"

echo "[2/3] Converting back to GTF format..."
agat_convert_sp_gff2gtf.pl --gff "${INTRON_GFF}" --relax -o "${INTRON_GTF}"

echo "[3/3] Extracting and sorting unique start/end coordinates..."
awk -F'\t' 'BEGIN {OFS="\t"} $3 == "intron" {print $1, $4; print $1, $5}' "${INTRON_GTF}" | sort -u > "${FINAL_OUT}"

echo "Pipeline completed successfully. Output saved to: ${FINAL_OUT}"
