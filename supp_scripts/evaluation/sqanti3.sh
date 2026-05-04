#!/usr/bin/env bash

# ==============================================================================
# Description: Runs SQANTI3 QC on an input GTF file against a reference genome
#              and annotation. Pre-processes the GTF to fix missing strands.
#
# ==============================================================================

# Exit immediately on error, treat unset variables as errors, catch pipe failures
set -euo pipefail

# ==============================================================================
# 1. Input Validation & Configuration
# ==============================================================================

# Ensure exactly 3 arguments are passed
if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <input.gtf> <genome.fa> <annotation.gtf>"
    exit 1
fi

INPUT_GTF="$1"
GENOME_FA="$2"
ANNOTATION_GTF="$3"

# Extract a clean base name from the input GTF to use for naming outputs safely
# e.g., "data/sample_A.gtf" becomes "sample_A"
PREFIX=$(basename "${INPUT_GTF}" .gtf)

SQANTI_BIN="sqanti3_qc.py" 
THREADS=32
WORKDIR="./sqanti_out_${PREFIX}"

# ==============================================================================
# 2. Pre-processing
# ==============================================================================

mkdir -p "${WORKDIR}"
TMP_GTF="${WORKDIR}/tmp_${PREFIX}.gtf"

echo "Using Annotation: ${ANNOTATION_GTF}"
echo "[1/2] Preparing GTF and standardizing strand information..."

# Sets undefined strands (.) to positive (+)
awk 'BEGIN{FS=OFS="\t"} $7=="." {$7="+"} 1' "${INPUT_GTF}" > "${TMP_GTF}"


# ==============================================================================
# 3. SQANTI3 Execution
# ==============================================================================

echo "[2/2] Running SQANTI3 QC..."

${SQANTI_BIN} \
    -t "${THREADS}" \
    --force_id_ignore \
    --min_ref_len 0 \
    --skipORF \
    --saturation \
    --report pdf \
    -d "${WORKDIR}" \
    -o "${PREFIX}" \
    "${TMP_GTF}" \
    "${ANNOTATION_GTF}" \
    "${GENOME_FA}" \
    &> "${WORKDIR}/sqanti3_${PREFIX}.log"


# ==============================================================================
# 4. Cleanup
# ==============================================================================

echo "Cleaning up temporary files..."
rm -f "${TMP_GTF}"

echo "Pipeline completed successfully. Results are in ${WORKDIR}/"
