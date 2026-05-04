import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq
import torch
import h5py
import time
import pysam
from dask import delayed, compute
from dask.distributed import Client
import pickle
import multiprocessing as mp

DEBUG = False

#  TODO: make curate_dataset_shared

def read_sequences_from_bag_file(fasta_path, bag_path):
    # Load the reference genome into a dictionary
    genome = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

    # Initialize a dictionary to store mappings of sequences
    sequence_map = {}
    with open(bag_path, 'r') as bag_file:
        for line in bag_file:
            parts = line.strip().split()
            chrom = parts[0]
            bed_start = int(parts[1]) + 1 #  change from 0- to 1-based for consistency
            bed_end = int(parts[2])
            
            bed_id = (chrom, bed_start, bed_end)

            exon_start_list= [int(x) for x in parts[3].split(',')]
            exon_end_list= [int(x) for x in parts[4].split(',')]
            min_left_blocks= [int(x) for x in parts[5].split(',')]
            min_right_blocks= [int(x) for x in parts[6].split(',')]

            # Extract the sequence for the second region
            if chrom in genome:
                bed_seq = str(genome[chrom].seq[bed_start-1:bed_end])
            else:
                print(f"Chromosome {chrom} not found in the reference genome.")
                continue

            # Map the sequence from the exon to the sequence from gene
            if bed_id not in sequence_map:
                sequence_map[bed_id] = [bed_seq, set()]

            seq_len = len(bed_seq)
            # exon_start and exon_end are now 1-based []
            for i in range(len(exon_start_list)):
                offset_start = exon_start_list[i] + 1 - bed_start
                offset_end = exon_end_list[i] - bed_start + 1 #  1-based added 1
                if offset_start < 0:
                    offset_start = 0
                if offset_end > seq_len:
                    offset_end = seq_len
                exon_id = (chrom, offset_start, offset_end, min_left_blocks[i], min_right_blocks[i])
                sequence_map[bed_id][1].add(exon_id)
    return sequence_map

def if_pos_zero_depth(chrom,intron_start,intron_end, bam_fn):
    #  (donor -1, acceptor +1 then one-based -1)
    intron_pair=(intron_start-1 -1, intron_end-1 -1 +1)
    num_of_ends_covered=0
    with pysam.AlignmentFile(bam_fn, "rb" ) as samfile:
        for i in range(2):
            pos=intron_pair[i]
            for pileupcolumn in samfile.pileup(chrom, pos, pos+1, min_base_quality=0,max_depth=200000,flag_filter=3844,truncate=True):
                for pileupread in pileupcolumn.pileups:
                    if not pileupread.is_del and not pileupread.is_refskip:
                        num_of_ends_covered+=1
                        break

    if num_of_ends_covered==0:
        #  all is_del or is_refskip
        return True
    else:
        return False

def mask_seq(chrom, bed_start, bed_end, values, bam_fn):



    seq = values[0].upper()
    sorted_lst = sorted(list(values[1]), key=lambda x: x[1])
    exon_str_lst = [] 

    for exon in sorted_lst:

        sub_seq = seq[exon[1] :exon[2]]
        if_canonical_and_nonzero=False
        if (sub_seq[:2]=='GT' and sub_seq[-2:]=='AG')==True or(sub_seq[:2]=='CT' and sub_seq[-2:]=='AC') ==True:
            if if_pos_zero_depth(chrom,bed_start+exon[1],bed_start+exon[2], bam_fn)==False:
                if_canonical_and_nonzero=True
        if if_canonical_and_nonzero==False:
            mask_start=exon[1]-exon[3]//2
            mask_end=exon[2]+exon[4]//2
            seq=''.join([seq[:mask_start], 'N' * (mask_end-mask_start), seq[mask_end:]])
            continue
            

        exon_str_lst.append(f'{exon[1]}_{exon[2]}')


    if not exon_str_lst:
        # no good exon candidates, skip this gene
        return ''
    return f'{seq}\t{ ",".join(exon_str_lst) }\t{chrom}_{bed_start}_{bed_end}\n'

def output_dataset_gene_seq_label(dataset, sequence_map, output_dir, bam_fn, test=['chr20', 'chr21'], val=['chr22']):
    """
    Outputs the sequence mapping data into training, testing, and validation datasets.

    Args:
    sequence_map (dict): A dictionary with keys as tuples representing sequence metadata (chromosome, start, end, strand)
                         and values as lists of sequences and their corresponding metadata.
    output_dir (str): Directory where the output files will be saved.
    test (list): List of chromosomes to include in the test dataset.
    val (list): List of chromosomes to include in the validation dataset.
    """
    # Ensure the output directory exists
    working_dir = os.path.join(output_dir, 'bed_seq_label')
    os.makedirs(working_dir, exist_ok=True)

    delayed_list=[]
    for key, values in sequence_map.items():
        chrom, bed_start, bed_end = key

        if chrom in test and dataset!='test':
            continue
        elif chrom in val and dataset!='val':
            continue
        elif chrom not in (test+val) and dataset!='train':
            continue
        delayed_list.append(delayed(mask_seq)(chrom, bed_start, bed_end, values, bam_fn))
    result_list=list(compute(*delayed_list,traverse=False,scheduler="processes"))
    with open(os.path.join(working_dir, f'{dataset}.txt'), 'w') as output_file:
        for line in result_list:
            if line:
                output_file.write(line)


def custom_division(n, d):
    return n / d if d else 0

def get_embeddings_from_pileup(bam_fn,chrom,start,end,out_dir, center_seq_size,window_cnt,ref_seq):
    embeddings=[]
    with pysam.AlignmentFile(bam_fn, "rb" ) as samfile:
        pos=0
        for pileupcolumn in samfile.pileup(chrom, start, end, min_base_quality=0,max_depth=200000,flag_filter=3844,truncate=True):
            ref_base=ref_seq[pos]
            alm_count=0
            del_count=0
            # TODO: feature experiments. current version only uses alm_count,refskip_count
            ins_and_mismatch_count=0
            refskip_count=0  #  count of spliced read
            for pileupread in pileupcolumn.pileups:
                if pileupread.is_refskip:
                    refskip_count+=1
                elif pileupread.is_del:
                    del_count+=1
                elif not pileupread.is_del:
                    alm_count+=1
                    if pileupread.alignment.query_sequence[pileupread.query_position]!=ref_base or pileupread.indel>0:
                        ins_and_mismatch_count+=1
            if alm_count!=0:
                embeddings.append([alm_count,refskip_count,(ins_and_mismatch_count+del_count)/(alm_count+del_count)])
            else:
                embeddings.append([alm_count,refskip_count,0])
            pos+=1
    #  possibly add pad_len to the rightmost window
    if len(embeddings)!=center_seq_size:
        embeddings=torch.cat((torch.tensor(embeddings),torch.zeros(center_seq_size-len(embeddings), len(embeddings[0]))),dim=0)
    with open(f'{out_dir}/{window_cnt}.pkl','wb') as fo:
        pickle.dump(embeddings,fo)
    
def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def chunkstring_if_labeled(string, length):
    return ((index,string[0+i:length+i]) for index,i in enumerate(range(0, len(string), length)) if '1' in string[0+i:length+i])

def chunk_labeled_seq(label, seq_str, length):
    return ((index,label[0+i:length+i],seq_str[0+i:length+i]) for index,i in enumerate(range(0, len(label), length)) if '1' in label[0+i:length+i])
def output_dataset_training(dataset, output_dir, bam_fn, flanking_context_size=0, center_seq_size=10000):
    bed_seq_label_dir = os.path.join(output_dir, 'bed_seq_label')
    working_dir = os.path.join(output_dir, 'training')
    os.makedirs(f'{working_dir}/{dataset}.pileup_embed/',exist_ok=True)

    bed_seq_label_file = open(os.path.join(bed_seq_label_dir, f'{dataset}.txt'), 'r')
    text_dataset_file = open(os.path.join(working_dir, f'{dataset}'), 'w')
   
    labels, delayed_pileup_embed_list = [], []
    window_cnt=0
    
    for line in bed_seq_label_file:
        parts = line.split('\t')
        seq_str = parts[0]
        N = len(seq_str)
        pad_len = (center_seq_size - N % center_seq_size) % center_seq_size

        seq_str = seq_str + 'N' * pad_len
        exons_str = parts[1]

        chrom,bed_start,bed_end=parts[2].strip().split('_')
        bed_end=int(bed_end)



        label = ['0'] * (N + pad_len)


        for x in exons_str.split(','):
            x0, x1 = x.split('_')
            x0, x1 = int(x0), int(x1)

            #  bed end 1-based

            label[x0], label[x1-1] = '1', '1'


        
        line_label=list(chunk_labeled_seq(label,seq_str,center_seq_size))
        for enum_label_seq in line_label:
            window_start=int(bed_start)+(enum_label_seq[0]*center_seq_size)
            window_end=int(bed_start)+((enum_label_seq[0]+1)*center_seq_size)
            if window_end>bed_end:
                window_end=bed_end

            delayed_pileup_embed_list.append(delayed(get_embeddings_from_pileup)(bam_fn, chrom, window_start, window_end,f'{working_dir}/{dataset}.pileup_embed/',center_seq_size,window_cnt,enum_label_seq[2]))
            window_cnt+=1
        labels.extend(line_label)

    compute(*delayed_pileup_embed_list,traverse=False,scheduler="processes")

    for enum_label_seq in labels:
        #  write seq
        text_dataset_file.write(enum_label_seq[2] + '\n')
        #  write label
        sub_label_str = ''.join(enum_label_seq[1])
        text_dataset_file.write(sub_label_str + '\n')




    bed_seq_label_file.close()
    text_dataset_file.close() 




def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) < 5:
        print("Usage: curate_dataset.py <dataset> <bam_file> <fasta_file> <bag_file>")
        return 1

    dataset = argv[1]
    bam_fn = argv[2]
    fasta_path = argv[3]
    bag_path = argv[4]

    if dataset not in {'train', 'val', 'test'}:
        print("Dataset must be one of: train, val, test")
        return 1

    base_name = os.path.basename(bag_path)
    output_dir = os.path.join(
        os.path.dirname(bag_path),
        (f'traindata_{base_name}')
    )
    os.makedirs(output_dir, exist_ok=True)

    client = Client(n_workers=24, threads_per_worker=24)

    try:
        print("[STEP 1] Reading sequences")
        sequence_mappings = read_sequences_from_bag_file(fasta_path, bag_path)
        print(f"  Found {len(sequence_mappings)} regions")

        print("[STEP 2] Generating sequence label file")
        output_dataset_gene_seq_label(dataset, sequence_mappings, output_dir, bam_fn)

        print("[STEP 3] Generating training data with embeddings")
        output_dataset_training(dataset, output_dir, bam_fn)

        print("[COMPLETE] Dataset generation finished")
    finally:
        client.close()

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
