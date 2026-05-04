#!/usr/bin/env Rscript

# ==============================================================================
# Description: Runs Bambu transcript discovery in two parallel modes:
#              1. Reference-guided discovery
#              2. De novo discovery (no annotation)
# ==============================================================================

# ==============================================================================
# 1. Configuration & Input Paths
# ==============================================================================

bam_file       <- "path/to/input/ENCFF683TBO.F3844.bam"
genome_fa      <- "path/to/reference/GRCm39.primary_assembly.genome.chr1-M.fa"
annotation_gtf <- "path/to/annotation/lrgasp_gencode_v38_sirvs.gtf.chr1-M.gtf"

out_guided <- paste0(bam_file, ".bambu_gencode.gtf")
out_denovo <- paste0(bam_file, ".bambu_noanno.gtf")


# ==============================================================================
# 2. Reference-Guided Discovery
# ==============================================================================

message("[1/2] Preparing reference annotations...")
bambu_annotations <- prepareAnnotations(annotation_gtf)

message("Running reference-guided Bambu discovery...")
se_guided <- bambu(
    reads       = bam_file, 
    annotations = bambu_annotations, 
    genome      = genome_fa, 
    quant       = FALSE
)

message("Writing guided results to GTF: ", out_guided)
writeToGTF(se_guided, out_guided)


# ==============================================================================
# 3. De Novo Discovery (No Annotation)
# ==============================================================================

message("[2/2] Running de novo Bambu discovery (NDR = 1)...")
se_denovo <- bambu(
    reads       = bam_file,
    annotations = NULL,
    genome      = genome_fa,
    NDR         = 1,
    quant       = FALSE
)

message("Writing de novo results to GTF: ", out_denovo)
writeToGTF(se_denovo, out_denovo)

message("Pipeline completed successfully.")
