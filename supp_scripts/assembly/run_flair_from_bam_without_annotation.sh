#!/usr/bin/env bash

# ==============================================================================
# Description: Converts a BAM alignment to BED12, formats it for FLAIR, and 
#              collapses transcript isoforms.
#
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, and catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.bam> <reads.fq> <genome.fa>"
    exit 1
fi

INPUT_BAM="$1"
READS_FQ="$2"
GENOME_FA="$3"

THREADS=128

BED12_OUT="${INPUT_BAM}.bed12"
FLAIRBED_OUT="${INPUT_BAM}.flairbed"

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "[1/3] Converting BAM to BED12..."
bedtools bamtobed -i "${INPUT_BAM}" -bed12 > "${BED12_OUT}"


echo "[2/3] Formatting BED12 for FLAIR..."
python3 bed_to_flairbed.py < "${BED12_OUT}" > "${FLAIRBED_OUT}"


echo "[3/3] Collapsing isoforms with FLAIR..."
flair collapse \
    -g "${GENOME_FA}" \
    -q "${FLAIRBED_OUT}" \
    -r "${READS_FQ}" \
    --threads "${THREADS}"


echo "Pipeline completed successfully."
