"""
Curate dataset for splice junction prediction.

This script processes genomic sequences and BAM files to generate inference datasets for machine learning models that predict splice junctions.
"""

import os
import sys

from Bio import SeqIO
import torch
import pysam
from dask import delayed, compute
from dask.distributed import Client
import pickle

#  TODO: make curate_dataset_shared

# Configuration
DEFAULT_CENTER_SEQ_SIZE = 10000
DEFAULT_FLANKING_CONTEXT_SIZE = 0
MIN_BASE_QUALITY = 0
MAX_PILEUP_DEPTH = 200000
PILEUP_FLAG_FILTER = 3844

def read_sequences_from_bag_file(fasta_path, bag_path):
    """
    Read genomic sequences from a FASTA file and extract regions from a BED-like BAG file.

    Args:
        fasta_path (str): Path to the reference genome FASTA file.
        bag_path (str): Path to the BAG file containing regions (chrom, start, end).

    Returns:
        dict: Dictionary mapping region tuples (chrom, start, end) to sequence lists.
    """
    # Load the reference genome into a dictionary
    genome = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

    # Initialize a dictionary to store mappings of sequences
    sequence_map = {}
    with open(bag_path, 'r') as bag_file:
        for line in bag_file:
            parts = line.strip().split()
            chrom = parts[0]
            bed_start = int(parts[1]) + 1  # Convert to 1-based indexing
            bed_end = int(parts[2])

            bed_id = (chrom, bed_start, bed_end)

            # Extract the sequence for the region
            if chrom in genome:
                bed_seq = str(genome[chrom].seq[bed_start - 1:bed_end])
            else:
                print(f"Chromosome {chrom} not found in the reference genome.")
                continue

            # Map the sequence to the region coordinates
            if bed_id not in sequence_map:
                sequence_map[bed_id] = [bed_seq]

    return sequence_map

def if_pos_zero_depth(chrom, intron_start, intron_end, bam_fn):
    """
    Check if splice junction positions have zero read coverage.

    Examines the donor (-1) and acceptor (+1) positions of a splice junction
    to see if either has zero coverage in the BAM file.

    Args:
        chrom (str): Chromosome name.
        intron_start (int): Start position of the intron.
        intron_end (int): End position of the intron.
        bam_fn (str): Path to the BAM file.

    Returns:
        bool: True if any position has zero coverage, False otherwise.
    """
    # Positions: (donor - 1, acceptor + 1) with 0-based indexing
    intron_pair = (intron_start - 1 - 1, intron_end - 1 - 1 + 1)
    num_of_ends_covered = 0

    with pysam.AlignmentFile(bam_fn, "rb") as samfile:
        for i in range(2):
            pos = intron_pair[i]

            for pileupcolumn in samfile.pileup(
                chrom, pos, pos + 1,
                min_base_quality=MIN_BASE_QUALITY,
                max_depth=MAX_PILEUP_DEPTH,
                flag_filter=PILEUP_FLAG_FILTER,
                truncate=True
            ):
                for pileupread in pileupcolumn.pileups:
                    if not pileupread.is_del and not pileupread.is_refskip:
                        num_of_ends_covered += 1
                        break

    return num_of_ends_covered == 0

def mask_seq(chrom, bed_start, bed_end, values, bam_fn):
    """
    Format sequence data with metadata.

    Args:
        chrom (str): Chromosome name.
        bed_start (int): Start position.
        bed_end (int): End position.
        values (list): List containing the sequence string.
        bam_fn (str): BAM file path (for compatibility).

    Returns:
        str: Formatted line with sequence, placeholder, and region identifier.
    """
    seq = values[0].upper()
    return f'{seq}\t.\t{chrom}_{bed_start}_{bed_end}\n'

def output_dataset_gene_seq_label(
    sequence_map, output_dir, bam_fn
):
    """
    Output sequence mapping data into inference datasets.

    Args:
        sequence_map (dict): Mapping of region tuples (chrom, start, end) to sequence lists.
        output_dir (str): Directory where output files will be saved.
        bam_fn (str): Path to the BAM file.
    """
    # Ensure the output directory exists
    working_dir = os.path.join(output_dir, 'bed_seq_label')
    os.makedirs(working_dir, exist_ok=True)

    delayed_list = []
    for key, values in sequence_map.items():
        chrom, bed_start, bed_end = key

        delayed_list.append(
            delayed(mask_seq)(chrom, bed_start, bed_end, values, bam_fn)
        )

    result_list = list(compute(*delayed_list, traverse=False, scheduler="processes"))
    with open(os.path.join(working_dir, 'test.txt'), 'w') as output_file:
        for line in result_list:
            if line:
                output_file.write(line)



def safe_division(numerator, denominator):
    """
    Perform safe division with zero handling.

    Args:
        numerator (float): The numerator.
        denominator (float): The denominator.

    Returns:
        float: Result of division, or 0 if denominator is zero.
    """
    return numerator / denominator if denominator else 0


def get_embeddings_from_pileup(
    bam_fn, chrom, start, end, out_dir,
    center_seq_size, window_cnt, ref_seq
):
    """
    Extract pileup embeddings from a BAM file for a given region.

    Creates embeddings from alignment pileup data, counting aligned reads
    and spliced reads at each position.

    Args:
        bam_fn (str): Path to the BAM file.
        chrom (str): Chromosome name.
        start (int): Start position.
        end (int): End position.
        out_dir (str): Output directory for pickle files.
        center_seq_size (int): Expected sequence size.
        window_cnt (int): Window counter for file naming.
        ref_seq (str): Reference sequence string.

    Returns:
        None (saves embeddings to pickle file)
    """
    embeddings = []
    with pysam.AlignmentFile(bam_fn, "rb") as samfile:
        pos = 0
        for pileupcolumn in samfile.pileup(
            chrom, start, end,
            min_base_quality=MIN_BASE_QUALITY,
            max_depth=MAX_PILEUP_DEPTH,
            flag_filter=PILEUP_FLAG_FILTER,
            truncate=True
        ):
            ref_base = ref_seq[pos]
            alm_count = 0
            refskip_count = 0  # Count of spliced reads

            for pileupread in pileupcolumn.pileups:
                if pileupread.is_refskip:
                    refskip_count += 1
                elif not pileupread.is_del:
                    alm_count += 1

            embeddings.append([alm_count, refskip_count])
            pos += 1

    if len(embeddings) == 0:
        print(f'Empty embeddings: {chrom}:{start}-{end}')
        embeddings = torch.zeros(center_seq_size, 2)
    elif len(embeddings) != center_seq_size:
        # Pad embeddings to expected size
        embeddings = torch.cat((
            torch.tensor(embeddings),
            torch.zeros(center_seq_size - len(embeddings), len(embeddings[0]))
        ), dim=0)

    with open(f'{out_dir}/{window_cnt}.{chrom}_{start}.pkl', 'wb') as fo:
        pickle.dump(embeddings, fo)


def chunkstring(string, length):
    """
    Chunk a string into fixed-length substrings.

    Args:
        string (str): String to chunk.
        length (int): Chunk size.

    Yields:
        str: Next chunk of specified length.
    """
    return (string[0 + i:length + i] for i in range(0, len(string), length))


def chunk_labeled_seq(label, seq_str, length):
    """
    Chunk a labeled sequence into fixed-length substrings with labels.

    Only yields chunks that contain at least one '1' in the label.

    Args:
        label (str): Label string (e.g., binary 0s and 1s).
        seq_str (str): Sequence string.
        length (int): Chunk size.

    Yields:
        tuple: (index, label_chunk, sequence_chunk)
    """
    return (
        (index, label[0 + i:length + i], seq_str[0 + i:length + i])
        for index, i in enumerate(range(0, len(label), length))
        if '1' in label[0 + i:length + i]
    )


def chunk_seq(seq_str, length):
    """
    Chunk a sequence into fixed-length substrings with indices.

    Args:
        seq_str (str): Sequence string.
        length (int): Chunk size.

    Yields:
        tuple: (index, sequence_chunk)
    """
    return (
        (index, seq_str[0 + i:length + i])
        for index, i in enumerate(range(0, len(seq_str), length))
    )
def output_dataset_inference(
    output_dir, bam_fn,
    flanking_context_size=None, center_seq_size=None
):
    """
    Generate datasets with sequence chunks and pileup embeddings.

    Args:
        output_dir (str): Base output directory.
        bam_fn (str): Path to the BAM file.
        flanking_context_size (int): Flanking context size. Defaults to DEFAULT_FLANKING_CONTEXT_SIZE.
        center_seq_size (int): Center sequence size. Defaults to DEFAULT_CENTER_SEQ_SIZE.
    """
    if flanking_context_size is None:
        flanking_context_size = DEFAULT_FLANKING_CONTEXT_SIZE
    if center_seq_size is None:
        center_seq_size = DEFAULT_CENTER_SEQ_SIZE

    input_prefix = 'test'

    bed_seq_label_dir = os.path.join(output_dir, 'bed_seq_label')
    working_dir = os.path.join(output_dir, 'training')
    pileup_embed_dir = os.path.join(working_dir, f'{input_prefix}.pileup_embed')
    os.makedirs(pileup_embed_dir, exist_ok=True)

    bed_seq_label_path = os.path.join(bed_seq_label_dir, f'{input_prefix}.txt')
    if not os.path.exists(bed_seq_label_path):
        raise FileNotFoundError(
            f"Expected labeled sequence file was not found: {bed_seq_label_path}"
        )

    bed_seq_label_file = open(bed_seq_label_path, 'r')
    text_dataset_file = open(os.path.join(working_dir, input_prefix), 'w')

    labels = []
    delayed_pileup_embed_list = []
    window_cnt = 0

    for line in bed_seq_label_file:
        parts = line.split('\t')
        seq_str = parts[0]
        seq_len = len(seq_str)

        # Pad sequence to multiple of center_seq_size
        pad_len = (center_seq_size - seq_len % center_seq_size) % center_seq_size
        seq_str = seq_str + 'N' * pad_len

        # Extract coordinates from region identifier
        chrom, bed_start, bed_end = parts[2].strip().rsplit('_', 2)
        bed_end = int(bed_end)

        # Chunk the sequence
        line_label = list(chunk_seq(seq_str, center_seq_size))

        for enum_label_seq in line_label:
            window_start = int(bed_start) + (enum_label_seq[0] * center_seq_size)
            window_end = int(bed_start) + ((enum_label_seq[0] + 1) * center_seq_size)

            if window_end > bed_end:
                window_end = bed_end

            delayed_pileup_embed_list.append(
                delayed(get_embeddings_from_pileup)(
                    bam_fn, chrom, window_start, window_end,
                    pileup_embed_dir, center_seq_size,
                    window_cnt, enum_label_seq[1]
                )
            )
            window_cnt += 1

        labels.extend(line_label)

    # Compute all pileup embeddings in parallel
    compute(*delayed_pileup_embed_list, traverse=False, scheduler="processes")

    # Write sequences to output file
    for enum_label_seq in labels:
        text_dataset_file.write(enum_label_seq[1] + '\n')

    bed_seq_label_file.close()
    text_dataset_file.close() 





if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: curate_dataset.inference.py <bam_file> <fasta_file> [num_workers] [threads_per_worker]")
        sys.exit(1)

    bam_fn = sys.argv[1]
    fasta_path = sys.argv[2]

    # Set default Dask configuration
    NUM_WORKERS = 64
    THREADS_PER_WORKER = 2

    # Override defaults if the user provided them via command line
    if len(sys.argv) >= 4:
        NUM_WORKERS = int(sys.argv[3])
    if len(sys.argv) >= 5:
        THREADS_PER_WORKER = int(sys.argv[4])

    # Input file for BAG regions
    bag_path = f'{bam_fn}.merge'

    # Generate output directory with parameters encoded in name
    base_name = os.path.basename(bag_path)
    output_dir = os.path.join(
        os.path.dirname(bag_path),
        (f'inferencedata_{base_name}')
    )
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Dask distributed client
    client = Client(n_workers=NUM_WORKERS, threads_per_worker=THREADS_PER_WORKER)

    try:
        # STEP 1: Extract sequences from FASTA and regions from BAG file
        print(f"[STEP 1] Reading sequences")
        sequence_mappings = read_sequences_from_bag_file(fasta_path, bag_path)
        print(f"  Found {len(sequence_mappings)} regions")

        # STEP 2: Output sequences with labels
        print(f"[STEP 2] Generating sequence label file")
        output_dataset_gene_seq_label(sequence_mappings, output_dir, bam_fn)

        # STEP 3: Generate data with pileup embeddings
        print(f"[STEP 3] Generating data with embeddings")
        output_dataset_inference(output_dir, bam_fn)

        print(f"[COMPLETE] Dataset generation finished")

    finally:
        client.close()
