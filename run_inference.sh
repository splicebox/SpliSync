#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# ==============================================================================
# 1. Configuration & Input Variables
# ==============================================================================
# Input parameters
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_bam> <ref_fasta>"
    exit 1
fi

INPUT_BAM="$1"
REF_FASTA="$2"

# Set SpliSync directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLISYNC_DIR="$SCRIPT_DIR"
MODEL_CKPT="${SPLISYNC_DIR}/model_weights/cdna_r9.ckpt"

# Prefix for intermediate and output files
PREFIX="${INPUT_BAM%.bam}"

# Intermediate file definitions
MERGED_BED="${PREFIX}.bam.merge"
DATA_DIR="inferencedata_${PREFIX}.bam.merge"
PREDICTIONS="${DATA_DIR}.predictions.tsv"
BED12_OUT="${PREFIX}.bed12"
INTRON_MARKS="${PREFIX}.intron_marks"
UPDATED_INTRONS="${PREFIX}.updated_${PREDICTIONS}_threshold21"
CORRECTED_BED="${PREFIX}.corrected.bed"
CLEAN_BED="${PREFIX}.corrected.cleaned.bed"
CLEAN_BAM="${PREFIX}.corrected.cleaned.bam"

# ==============================================================================
# 2. Preprocessing & Data Curation
# ==============================================================================
echo "[1/5] Preparing reference and BAM files..."
samtools faidx "$REF_FASTA"
samtools index "$INPUT_BAM"
bedtools merge -i "$INPUT_BAM" > "$MERGED_BED"

echo "[2/5] Curating dataset for inference..."
python "${SPLISYNC_DIR}/curate_dataset.inference.py" "$INPUT_BAM" "$REF_FASTA"

# ==============================================================================
# 3. SpliSync Model Inference
# ==============================================================================
echo "[3/5] Running SpliSync inference..."
python "${SPLISYNC_DIR}/run_splisync.py" \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_CKPT"

# ==============================================================================
# 4. BED Block Correction
# ==============================================================================
echo "[4/5] Applying predictions and correcting BED blocks..."

# Convert BAM to BED12
bedtools bamtobed -i "$INPUT_BAM" -bed12 > "$BED12_OUT"

# Extract introns and apply model predictions
# Output format:
# index  chrom  coord  is_intron_end  strand
python "${SPLISYNC_DIR}/bed12_to_introns.py" "$BED12_OUT" > "$INTRON_MARKS"

# Update intron start / end coordinates using positions predicted by SpliSync
python "${SPLISYNC_DIR}/update_introns.py" "$INTRON_MARKS" "$PREDICTIONS" -o "$UPDATED_INTRONS"

# Correct the BED12 blocks based on updated splice sites
python "${SPLISYNC_DIR}/correct_bed_blocks.py" "$BED12_OUT" "$UPDATED_INTRONS" -o "$CORRECTED_BED"

# Clean up zero-length blocks
# Output: cleaned.bed, filtered_indices
python "${SPLISYNC_DIR}/remove_zero_blocks.py" "$CORRECTED_BED" 

# ==============================================================================
# 5. BAM Reconstruction & Tag Transfer
# ==============================================================================
echo "[5/5] Reconstructing corrected BAM file..."

# Extract FASTA sequences in the background 
bedtools getfasta -fi "$REF_FASTA" -bed "$CLEAN_BED" -split -bedOut > "${CLEAN_BED}.getfasta" &

# Convert corrected BED12 back to BAM
bedToBam -i "$CLEAN_BED" -g "${REF_FASTA}.fai" -bed12 > "$CLEAN_BAM"
samtools index "$CLEAN_BAM"

# Add XS tags for splicing analysis
samtools view -h "$CLEAN_BAM" | "${SPLISYNC_DIR}/addXS" "$REF_FASTA" > "${CLEAN_BAM}.addxs"

# Wait for background processes (e.g., getfasta) to finish
wait

# Transfer original BAM tags to the newly corrected BAM
python "${SPLISYNC_DIR}/transfer_bam_tags.py" "$INPUT_BAM" "$CLEAN_BAM" "${CORRECTED_BED}.filtered_indices" "${CLEAN_BED}.getfasta" "${CLEAN_BAM}.addxs"

echo "Workflow complete! Corrected BAM: ${CLEAN_BAM%.*}.withtags.bam"
