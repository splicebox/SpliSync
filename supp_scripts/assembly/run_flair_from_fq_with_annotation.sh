#!/usr/bin/env bash

# ==============================================================================
# Description: Runs the core FLAIR (Full-Length Alternative Isoform analysis of RNA) 
#              pipeline: align, correct, and collapse.
#
# ==============================================================================

# Exit immediately if a command fails, treat unset variables as errors, 
# and catch failures in piped commands.
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <genome.fa> <annotation.gtf> <reads.fq>"
    exit 1
fi

GENOME_FA="$1"
ANNOTATION_GTF="$2"
READS_FILE="$3"

THREADS=128
OUT_DIR="correct_gtf"

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "[1/3] Running FLAIR align..."
flair align \
    -g "${GENOME_FA}" \
    -r "${READS_FILE}" \
    --threads "${THREADS}"


echo "[2/3] Running FLAIR correct..."
mkdir -p "${OUT_DIR}"

flair correct \
    -q flair.aligned.bed \
    --genome "${GENOME_FA}" \
    --gtf "${ANNOTATION_GTF}" \
    --threads "${THREADS}" \
    --output "${OUT_DIR}/flair"


echo "[3/3] Running FLAIR collapse..."
flair collapse \
    -g "${GENOME_FA}" \
    -q "${OUT_DIR}/flair_all_corrected.bed" \
    -r "${READS_FILE}" \
    --threads "${THREADS}" \
    --output "${OUT_DIR}/flair.collapse"


echo "Pipeline completed successfully. Output saved to: ${OUT_DIR}/"
