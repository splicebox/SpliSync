#!/usr/bin/env bash

# ==============================================================================
# Description: Converts a BED file to GFF/GTF, explicitly annotates introns 
#              using AGAT, and extracts unique intron start/end coordinates.
#
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.bed>"
    exit 1
fi

INPUT_BED="$1"

# Extract clean base name for output files (e.g., "sample.bed" -> "sample")
PREFIX=$(basename "${INPUT_BED}" .bed)

BASE_GFF="${PREFIX}.gff"
BASE_GTF="${PREFIX}.gtf"
INTRON_GFF="${PREFIX}.with_introns.gff"
INTRON_GTF="${PREFIX}.with_introns.gtf"
FINAL_COORDS="${PREFIX}.unique_splicesites.txt"

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "[1/4] Converting input BED to GFF..."
agat_convert_bed2gff.pl --bed "${INPUT_BED}" > "${BASE_GFF}"

# Moving this step up to keep the base conversions logically grouped
echo "[2/4] Generating base GTF format..."
agat_convert_sp_gff2gtf.pl --gff "${BASE_GFF}" --relax -o "${BASE_GTF}"

echo "[3/4] Adding intron features via AGAT..."
agat_sp_add_introns.pl --gff "${BASE_GFF}" -o "${INTRON_GFF}"
agat_convert_sp_gff2gtf.pl --gff "${INTRON_GFF}" --relax -o "${INTRON_GTF}"

echo "[4/4] Extracting unique intron coordinates..."
awk -F'\t' 'BEGIN {OFS="\t"} $3 == "intron" {print $1, $4; print $1, $5}' "${INTRON_GTF}" | sort -u > "${FINAL_COORDS}"

echo "Pipeline completed successfully."
echo "Results:"
echo "  - Base GTF:       ${BASE_GTF}"
echo "  - Intron GTF:     ${INTRON_GTF}"
echo "  - Intron Coords:  ${FINAL_COORDS}"
