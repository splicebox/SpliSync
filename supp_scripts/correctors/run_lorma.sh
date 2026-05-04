#!/usr/bin/env bash

# ==============================================================================
# Script: run_lorma.sh
# Description: Executes LoRMA for long-read error correction on a FASTQ file.
#
# Usage: ./run_lorma.sh <input.fq.gz>
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, and catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.fq.gz>"
    exit 1
fi

# Assign argument to a descriptive variable
INPUT_FASTQ="$1"

LORMA_BIN="lorma.sh"
THREADS=32

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "Starting LoRMA error correction on: ${INPUT_FASTQ}"
echo "Allocating ${THREADS} threads..."

${LORMA_BIN} \
    -threads "${THREADS}" \
    "${INPUT_FASTQ}"

echo "LoRMA execution completed successfully."
