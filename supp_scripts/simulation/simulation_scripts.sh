# ==============================================================================
# Supplementary Commands: NanoSim Simulation
# Description: Individual commands and utility steps used for RNA read simulation.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Read Analysis
# ------------------------------------------------------------------------------
# Analyzes the input FASTQ reads against the reference genome/transcriptome.

read_analysis.py transcriptome \
    -i ENCFF155CFF.1m.fastq \
    -rg /path/to/GRCh38.p13.genome.fa \
    -rt /path/to/gencode.v38.transcripts.fa \
    -annot /path/to/gencode.v38.annotation.gtf \
    -t 128 \
    -o nanosim_model_out/ENCFF155CFF.1m \
    --no_intron_retention \
    --fastq


# ------------------------------------------------------------------------------
# 2. Transcript Quantification (Salmon)
# ------------------------------------------------------------------------------
# Quantifies transcript abundance using ONT-aligned reads.

salmon quant \
    -t /path/to/gencode.v38.transcripts.fa.gz \
    -l A \
    -a ENCFF155CFF.fastq.gencode.v38.transcripts.fa.ax_map-ont.bam \
    -o salmon_q_out \
    --ont


# ------------------------------------------------------------------------------
# 3. Data Curation & Training Preprocessing
# ------------------------------------------------------------------------------

# Filter for protein-coding and lncRNA transcripts
awk -v OFS="\t" '{print $1, $5, $4}' salmon_q_out/quant.sf | \
    grep -E 'protein_coding|lncRNA' > quant.sf.nanosim_exp.lncRNA_protein_coding

# ------------------------------------------------------------------------------
# 4. Read Simulation (NanoSim dRNA)
# ------------------------------------------------------------------------------

simulator.py transcriptome \
    -rg /path/to/GRCh38.p13.genome.fa \
    -rt /path/to/gencode.v38.transcripts.fa \
    -e /path/to/quant.sf.nanosim_exp.lncRNA_protein_coding \
    -o nanosim_model_out \
    -c nanosim_model_out/ENCFF155CFF.1m \ # custom or pretrained error models
    -b guppy \
    --fastq \
    -n 4000000 \
    -t 128 \
    --uracil \
    --no_model_ir

