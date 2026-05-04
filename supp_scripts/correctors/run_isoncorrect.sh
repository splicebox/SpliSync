#!/usr/bin/env bash

# ==============================================================================
# Script: run_isoncorrect.sh
# Description: Runs isONcorrect for error correction of long-read RNA-seq data.
#
# Usage: ./run_isoncorrect.sh <input.fastq>
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, and catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.fastq>"
    exit 1
fi

INPUT_FASTQ="$1"

OUT_FOLDER="${INPUT_FASTQ}.isoncor"

ISONCORRECT_BIN="isONcorrect"

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "Starting error correction with isONcorrect for: ${INPUT_FASTQ}"

# Run isONcorrect with line continuations for readability
${ISONCORRECT_BIN} \
    --fastq "${INPUT_FASTQ}" \
    --outfolder "${OUT_FOLDER}"

echo "Error correction completed successfully. Results saved to: ${OUT_FOLDER}/"
