#!/usr/bin/env bash

# ==============================================================================
# Description: Extracts junctions from a BAM file using PsiCLASS and formats
#              them into splice sites.
#
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, and catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.bam>"
    exit 1
fi

INPUT_BAM="$1"

JUNC_BIN="junc"
FORMAT_SCRIPT="junc_to_splicesites.py"

JUNC_OUT="${INPUT_BAM}.junc_aj0"
SPLICESITES_OUT="${JUNC_OUT}.splicesites"

# ==============================================================================
# 2. Pipeline Execution
# ==============================================================================

echo "[1/2] Extracting junctions using junc..."
${JUNC_BIN} "${INPUT_BAM}" -a -j 0 > "${JUNC_OUT}"


echo "[2/2] Formatting junctions to splice sites..."
python3 "${FORMAT_SCRIPT}" "${JUNC_OUT}" > "${SPLICESITES_OUT}"


echo "Pipeline completed successfully. Output saved to: ${SPLICESITES_OUT}"
