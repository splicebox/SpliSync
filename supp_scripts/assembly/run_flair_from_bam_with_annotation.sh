#!/usr/bin/env bash

# ==============================================================================
# Description: Converts BAM to BED12, formats for FLAIR, corrects splice sites
#              using a reference annotation, and collapses isoforms.
#
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, and catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

# Ensure exactly 4 arguments are passed
if [ "$#" -ne 4 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.bam> <genome.fa> <annotation.gtf> <reads.fq>"
    exit 1
fi

# Assign arguments to descriptive variables
INPUT_BAM="$1"
GENOME_FA="$2"
ANNOTATION_GTF="$3"
READS_FQ="$4"

# Configuration (Users can modify these if needed)
THREADS=128
OUT_DIR="correct_gtf"

# Assuming the python script is provided alongside this bash script in the repo
BED_TO_FLAIR_SCRIPT="bed_to_flairbed.py" 

# Define intermediate output filenames
BED12_OUT="${INPUT_BAM}.bed12"
FLAIRBED_OUT="${INPUT_BAM}.flairbed"

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "[1/4] Converting BAM to BED12..."
bedtools bamtobed -i "${INPUT_BAM}" -bed12 > "${BED12_OUT}"


echo "[2/4] Formatting BED12 for FLAIR..."
# Redirecting the file directly into Python is more efficient than piping from 'cat'
python3 "${BED_TO_FLAIR_SCRIPT}" < "${BED12_OUT}" > "${FLAIRBED_OUT}"


echo "[3/4] Correcting splice sites with FLAIR..."
mkdir -p "${OUT_DIR}"

flair correct \
    -q "${FLAIRBED_OUT}" \
    --genome "${GENOME_FA}" \
    --gtf "${ANNOTATION_GTF}" \
    --threads "${THREADS}" \
    --output "${OUT_DIR}/flair"


echo "[4/4] Collapsing isoforms with FLAIR..."
flair collapse \
    -g "${GENOME_FA}" \
    -q "${OUT_DIR}/flair_all_corrected.bed" \
    -r "${READS_FQ}" \
    --threads "${THREADS}" \
    --output "${OUT_DIR}/flair.collapse"


echo "Pipeline completed successfully. Output saved to: ${OUT_DIR}/"
