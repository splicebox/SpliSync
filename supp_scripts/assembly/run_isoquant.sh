#!/usr/bin/env bash

# ==============================================================================
# Description: Runs IsoQuant (v3.10+) in two parallel modes on a single BAM file.
#              Job 1: Reference-guided quantification (outputs GTF).
#              Job 2: Standard/discovery quantification.
#
# ==============================================================================

# Exit immediately on error and treat unset variables as errors
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.bam> <reference.fa> <annotation.gtf>"
    exit 1
fi

INPUT_BAM="$1"
REFERENCE_FA="$2"
ANNOTATION_GTF="$3"

THREADS_ANNOTATED=48
THREADS_DISCOVERY=48

# ==============================================================================
# 2. Parallel IsoQuant Execution
# ==============================================================================

echo "Starting parallel IsoQuant analysis (v3.10)..."

echo "[Job 1] Starting reference-guided quantification (Threads: ${THREADS_ANNOTATED})..."
isoquant.py \
    --reference "${REFERENCE_FA}" \
    --bam "${INPUT_BAM}" \
    --genedb "${ANNOTATION_GTF}" \
    --data_type nanopore \
    --count_exons \
    --complete_genedb \
    --threads "${THREADS_ANNOTATED}" \
    -o "${INPUT_BAM}.IQ.gencode.gtf" &

echo "[Job 2] Starting standard quantification (Threads: ${THREADS_DISCOVERY})..."
isoquant.py \
    --reference "${REFERENCE_FA}" \
    --bam "${INPUT_BAM}" \
    --data_type nanopore \
    --threads "${THREADS_DISCOVERY}" \
    -o "${INPUT_BAM}.IQ" &

echo "Waiting for IsoQuant jobs to complete..."
wait

echo "Pipeline completed successfully."
