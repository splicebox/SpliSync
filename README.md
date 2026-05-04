# SpliSync
Genomic-language-model-driven splice site correction of long RNA sequencing reads.

SpliSync improves splice-site accuracy in long-read RNA-seq alignments using a genomic language model, producing corrected BED/BAM outputs for downstream isoform assembly and splicing analysis.

## ✨ Key Features
- 

## 🚀 Quick Start / Installation
1. Clone the repository and enter the project directory.
2. Create the conda environment from [splisync_env.yml](splisync_env.yml):

```bash
conda env create -f splisync_env.yml
conda activate splisync
```

3. Build the `addXS` helper binary from [AddXS.cpp](AddXS.cpp) (required if you want XS tags in corrected BAMs):

```bash
g++ -c -o addXS.o -I./samtools-0.1.19 -L./samtools-0.1.19 -Wall -O3 AddXS.cpp -lbam -lz -lm -lpthread
g++ -o addXS -I./samtools-0.1.19 -L./samtools-0.1.19 -Wall -O3 addXS.o -lbam -lz -lm -lpthread
```

4. Prepare inputs:
	 - A sorted, indexed BAM of long-read RNA-seq alignments.
	 - A reference genome FASTA (indexed with `samtools faidx`).

## 🧪 Usage
### Inference pipeline 
Edit the configuration variables at the top of [run_inference.sh](run_inference.sh), then run:

```bash
bash run_inference.sh
```

### Notebook demo
Open [demo.ipynb](demo.ipynb) for a step-by-step, reproducible walkthrough of the inference workflow.

## 📊 Inputs & Outputs

### Inputs
| File | Format | Description | Example |
|------|--------|-------------|---------|
| **Aligned Reads** | BAM (sorted, indexed) | Long-read RNA-seq alignments (Nanopore, PacBio cDNA/IsoSeq). Must be coordinate-sorted with `.bai` index. | `reads_aligned.bam` |
| **Reference Genome** | FASTA (faidx-indexed) | Reference genome sequence. Index generated automatically by `samtools faidx`. | `GRCh38.p13.genome.fa` |
| **Model Checkpoint** | PyTorch `.ckpt` | Pretrained weights for splice-site correction. Provided in [model_weights/](model_weights/). | `model_weights/cdna_r9.ckpt` |

### Outputs
| File | Format | Description |
|------|--------|-------------|
| **Corrected Alignments** | BED12 | Corrected transcript intervals with updated splice-site block boundaries. Each row is a corrected isoform. |
| **Corrected BAM** | BAM | Reconstructed BAM file with corrected coordinates and (optionally) XS tags for strand-specific junction analysis. Indexed `.bai` file included. |
| **Predictions** | TSV | Model confidence scores and splice-site calls for each junction. Columns: `junction_id`, `score`, `predicted_site`. |
| **Intermediate Data** | NumPy | Curated feature embeddings from the reference and reads; used for model inference. Stored in `inferencedata_*` directory. |

### Example Workflow I/O
```
Input:
├── reads_aligned.bam          # 500k aligned Nanopore cDNA reads
├── reads_aligned.bam.bai      # BAM index
└── GRCh38.p13.genome.fa       # Human reference (faidx-indexed)

Configuration (run_inference.sh):
├── MODEL_CKPT = "model_weights/cdna_r9.ckpt"
├── INPUT_BAM = "reads_aligned.bam"
└── REF_FASTA = "GRCh38.p13.genome.fa"

Output (after pipeline completion):
├── reads_aligned.bed12                                    # Corrected isoforms (BED12)
├── reads_aligned.corrected.bam                            # Corrected alignments (with XS tags)
├── reads_aligned.corrected.bam.bai                        # BAM index
├── inferencedata_reads_aligned_predictions.tsv            # Splice-site confidence scores
└── reads_aligned.intron_marks                             # Intermediate intron annotations
```

## 🤝 Contributing
Contributions are welcome!

- Open an issue for bugs or feature requests.
- Fork the repo and create a feature branch.
- Keep scripts reproducible and document any new inputs/outputs.
- Submit a pull request with a clear description of the change.

## 📄 License
GNU General Public License v3.0.
