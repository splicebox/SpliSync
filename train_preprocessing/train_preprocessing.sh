gtf_to_bed12.sh gencode.v38.annotation.gtf

get_gtf_isin_txid.py -i gencode.v38.annotation.gtf.bed -t txid.list

# Generate intron blocks (intron.block.bed) from BED12
python3 extract_introns_from_bed12.py gencode.v38.annotation.gtf.isin_txid.bed12

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

python3 aggregates_intersected_splicesites.py "${INTERSECT_OUT}"

for dataset in train val test
do
python3 curate_dataset.train.py $dataset "${SIM_BAM}"  $FASTA "${INTERSECT_OUT}"
done

python ../run_splisync.py --data_dir $traindata --stage train --batch_size $batch_size
