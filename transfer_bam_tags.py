"""Transfer tags and sequences between BAM records using BED and addXS inputs."""

import sys

import pandas as pd
import pysam


def load_filtered_indices(path):
    filtered_indices = []
    with open(path, "r") as filtered_indices_file:
        for line in filtered_indices_file:
            filtered_indices.append(int(line.strip()))
    return filtered_indices


def advance_to_first_data_line(handle):
    line = next(handle)
    while line[0] == "@":
        line = next(handle)
    return line


def infer_strand(addxs_columns, alignment):
    suffix = addxs_columns[0][-4:]
    if suffix == "GTAG":
        strand = "+"
    elif suffix == "CTAC":
        strand = "-"
    else:
        strand = "?"
        if addxs_columns[-1][:2] == "YS":
            ys_tag_i = int(addxs_columns[-1].split(":")[-1])
            alignment.set_tag("YS", ys_tag_i, value_type="i")
    return strand


source_bam = sys.argv[1]
target_bam = sys.argv[2]
target_bam_filtered_indices_fn = sys.argv[3]
bed_fn = sys.argv[4]
addxs_fn = sys.argv[5]

bam_file_in1 = pysam.AlignmentFile(source_bam, "rb")
bam_file_in2 = pysam.AlignmentFile(target_bam, "rb")
bed_df = pd.read_csv(bed_fn, sep="\t", header=None)

addxs_in = open(addxs_fn, "r")
addxs_line = advance_to_first_data_line(addxs_in)

seqs_array = bed_df[12].values
target_bam_filtered_indices = load_filtered_indices(target_bam_filtered_indices_fn)
bam_file_out = pysam.AlignmentFile(
    f"{target_bam.rsplit('.', 1)[0]}.withtags.bam", "wb", template=bam_file_in2
)

seqs_itr = iter(seqs_array)

for i, alignment_in1 in enumerate(bam_file_in1):
    if i in target_bam_filtered_indices:
        continue

    alignment_in2 = next(bam_file_in2)
    seq = next(seqs_itr)

    alignment_in2.tags = alignment_in1.tags
    alignment_in2.mapping_quality = alignment_in1.mapping_quality
    alignment_in2.seq = seq.strip()

    if "N" in alignment_in2.cigarstring:
        addxs_cols = addxs_line.strip().split("\t")
        strand = infer_strand(addxs_cols, alignment_in2)
        alignment_in2.set_tag("XS", strand, value_type="A")

    bam_file_out.write(alignment_in2)
    addxs_line = next(addxs_in, None)

bam_file_in1.close()
bam_file_in2.close()
bam_file_out.close()
addxs_in.close()

