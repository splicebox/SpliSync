bash gtf_to_bed12.sh gencode.v38.annotation.gtf

# Remove the header and extract non-zero expression transcript IDs for training from the nanosim expression profile
awk '$3 > 0 {print}' quant.sf.lncRNA_protein_coding.nanosim_exp.tsv | \
    cut -f1 -d'|' > quant.sf.lncRNA_protein_coding.nanosim_exp.tsv.non0_txid.list

python get_gtf_isin_txid.py -i gencode.v38.annotation.gtf.bed -t quant.sf.lncRNA_protein_coding.nanosim_exp.tsv.non0_txid.list

# Generate intron blocks (intron.block.bed) from BED12
python extract_introns_from_bed12.py gencode.v38.annotation.gtf.isin_txid.bed12

SIM_BAM="ENCFF155CFF.fastq.1m.nanosim_model_aligned_reads.fastq.chr1-M.F3844.bam"
MERGED_BED="${SIM_BAM}.merge"
INTRON_BED="gencode.v38.annotation.gtf.bed12.isin_tsid.bed12.intron.block.bed"
INTERSECT_OUT="${MERGED_BAM}.intersect.intron.block.bed"

# Merge BAM intervals
bedtools merge -i "${SIM_BAM}" > "${MERGED_BED}"

# Intersect merged alignments with intron blocks
bedtools intersect \
    -a "${MERGED_BED}" \
    -b "${INTRON_BED}" \
    -wa -wb -F 1 > "${INTERSECT_OUT}"

python aggregates_intersected_splicesites.py "${INTERSECT_OUT}"

for dataset in train val test
    do
        python ../curate_dataset.train.py $dataset "${SIM_BAM}" $FASTA "${INTERSECT_OUT}"
    done

python ../run_splisync.py --data_dir traindata_${INTERSECT_OUT} --stage train --batch_size $batch_size
