samtools faidx /home/wlui3/data/GRCh38.p13.genome.chr1-M.fa

bedtools merge -i reads_aligned_reads.50k.fastq.chr1-M.F3844.bam > reads_aligned_reads.50k.fastq.chr1-M.F3844.bam.merge
# don't forget to provide a bam index
python /data/lflorea1/wlui3/tools/SpliSync/curate_dataset.inference.py reads_aligned_reads.50k.fastq.chr1-M.F3844.bam /home/wlui3/data/GRCh38.p13.genome.chr1-M.fa
python run_splisync.py --data_dir inferencedata_reads_aligned_reads.50k.fastq.chr1-M.F3844.bam.merge --model_path model_weights/cdna_r9.ckpt
bedtools bamtobed -i reads_aligned_reads.50k.fastq.chr1-M.F3844.bam -bed12 > reads_aligned_reads.50k.fastq.chr1-M.F3844.bam.bed12
python bed12_to_introns.py reads_aligned_reads.chr1-M.F3844.bam.bed12 > reads_aligned_reads.chr1-M.F3844.bam.bed12.intron_marks
python update_introns.py reads_aligned_reads.chr1-M.F3844.bam.bed12.intron_marks inferencedata_reads_aligned_reads.50k.fastq.chr1-M.F3844.bam.merge.predictions.tsv
python correct_bed_blocks.py reads_aligned_reads.chr1-M.F3844.bam.bed12 reads_aligned_reads.chr1-M.F3844.bam.bed12.intron_marks.updated_inferencedata_reads_aligned_reads.50k.fastq.chr1-M.F3844.bam.merge.predictions.tsv.threshold21
python python /data/lflorea1/wlui3/tools/SpliSync/remove_zero_blocks.py reads_aligned_reads.chr1-M.F3844.bam.bed12.corrected
bedtools getfasta -fi /home/wlui3/data/GRCh38.p13.genome.chr1-M.fa -bed reads_aligned_reads.chr1-M.F3844.bam.bed12.corrected.rm0block -split -bedOut > reads_aligned_reads.chr1-M.F3844.bam.bed12.corrected.rm0block.getfasta.bed &
bedToBam -i  reads_aligned_reads.chr1-M.F3844.bam.bed12.corrected.rm0block -g /home/wlui3/data/GRCh38.p13.genome.chr1-M.fa.fai -bed12 > reads_aligned_reads.chr1-M.F3844.bam.bed12.corrected.rm0block.bama
samtools index reads_aligned_reads.chr1-M.F3844.bam.bed12.corrected.rm0block.bam
samtools view -h reads_aligned_reads.chr1-M.F3844.bam.bed12.corrected.rm0block.bam| /data/lflorea1/wlui3/tools/SpliSync/addXS /home/wlui3/data/GRCh38.p13.genome.chr1-M.fa > reads_aligned_reads.chr1-M.F3844.bam.bed12.corrected.rm0block.bam.addxs
wait
python transfer_bam_tags.py reads_aligned_reads.chr1-M.F3844.bam reads_aligned_reads.chr1-M.F3844.bam.bed12.corrected.rm0block.bam *filtered_indices *getfasta.bed *addxs
