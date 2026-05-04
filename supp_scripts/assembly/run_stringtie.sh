#!/usr/bin/env bash

# ==============================================================================
# Description: Runs StringTie (v3.0.1+) transcript assembly in two parallel modes 
#              on a single BAM file:
#              Job 1: Reference-guided assembly (using an annotation GTF).
#              Job 2: De novo discovery assembly (no annotation).
#
# ==============================================================================

# Exit immediately on error and treat unset variables as errors
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.bam> <annotation.gtf>"
    exit 1
fi

INPUT_BAM="$1"
ANNOTATION_GTF="$2"

STRINGTIE_BIN="stringtie" 

THREADS=128

# ==============================================================================
# 2. Parallel StringTie Execution
# ==============================================================================

echo "Starting parallel StringTie assemblies..."

echo "[Job 1] Starting reference-guided assembly (Threads: ${THREADS})..."
${STRINGTIE_BIN} \
    -L \
    -p "${THREADS}" \
    -G "${ANNOTATION_GTF}" \
    -o "${INPUT_BAM}.stringtie.anno.gencode.gtf" \
    "${INPUT_BAM}" &


echo "[Job 2] Starting de novo assembly (Threads: ${THREADS})..."
${STRINGTIE_BIN} \
    -L \
    -p "${THREADS}" \
    -o "${INPUT_BAM}.stringtie.no_anno.gtf" \
    "${INPUT_BAM}" &


echo "Waiting for StringTie jobs to complete..."
wait

echo "StringTie assemblies completed successfully."
