#!/usr/bin/env bash

# ==============================================================================
# Script: run_desalt.sh
# Description: Builds a reference genome index and aligns long reads using deSALT.
#
# Usage: ./run_desalt.sh <reference.fa> <reads.fq> <index_directory>
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, and catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <reference.fa> <reads.fq> <index_directory>"
    exit 1
fi

REFERENCE_FA="$1"
READS_FQ="$2"
INDEX_DIR="$3"

DESALT_BIN="deSALT"

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "[1/2] Building deSALT index for ${REFERENCE_FA}..."
${DESALT_BIN} index "${REFERENCE_FA}" "${INDEX_DIR}"


echo "[2/2] Aligning reads from ${READS_FQ}..."
${DESALT_BIN} aln "${INDEX_DIR}" "${READS_FQ}"


echo "deSALT pipeline completed successfully."
