#!/usr/bin/env bash

# ==============================================================================
# Script: run_transcriptclean.sh
# Description: Streams a BAM file into TranscriptClean for error correction.
#
# Usage: ./run_transcriptclean.sh <input.bam> <reference.fa>
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.bam> <reference.fa>"
    exit 1
fi

INPUT_BAM="$1"
REFERENCE_FA="$2"

PREFIX=$(basename "${INPUT_BAM}")

SAMTOOLS_BIN="samtools"
TRANSCRIPTCLEAN_BIN="transcriptclean"
THREADS=48

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "Starting TranscriptClean processing for ${INPUT_BAM}..."

${TRANSCRIPTCLEAN_BIN} \
    --sam <(${SAMTOOLS_BIN} view -h "${INPUT_BAM}") \
    --genome "${REFERENCE_FA}" \
    -t "${THREADS}" \
    --outprefix "${PREFIX}.transcriptclean"

echo "TranscriptClean processing completed successfully."
