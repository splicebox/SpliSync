# Train Preprocessing

Preprocessing pipeline for building train/val/test datasets from a GENCODE GTF, a NanoSim expression profile, and simulated alignments. The workflow converts annotations to BED12, filters transcript IDs by non-zero expression, generates intron blocks, intersects simulated alignments with introns, aggregates splice sites, and curates datasets before launching training.

## Inputs

- `${SIM_BAM}`
- A reference FASTA (set via `FASTA`)
- `gencode.v38.annotation.gtf`
- `quant.sf.lncRNA_protein_coding.nanosim_exp.tsv`

## Outputs

- Curated sequence and pile-up files produced by `curate_dataset.train.py`
- Model checkpoints

## Usage

1) Prepare required input files in the working directory.
2) Set required environment variables:

```bash
set SIM_BAM=/path/to/sim.bam
set FASTA=/path/to/reference.fasta
set batch_size=32
```

3) Run the script:

```bash
bash train_preprocessing.sh
```
