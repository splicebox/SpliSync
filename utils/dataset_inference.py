"""Dataset helpers for sequence text and pileup embeddings."""

import glob
import os

import dill as pickle
import numpy as np
from torch.utils.data import Dataset

#  TODO: make dataset_shared 

class SeqTextDataset(Dataset):
    """Load sequence strings and paired pileup embeddings from disk."""

    def __init__(self, file_path, tokenizer, flanking, center):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.flanking = flanking
        self.center = center
        self.sequences, self.labels = [], []
        with open(self.file_path, "r") as file:
            while True:
                line1 = file.readline()
                if not line1:
                    break
                self.sequences.append(line1.strip())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index][self.flanking : self.flanking + self.center]
        sequence = self.tokenizer(seq,
                add_special_tokens=False,
                padding='do_not_pad',
                truncation=False)
        sequence = np.asarray(sequence["input_ids"], dtype=np.int64)

        embed_paths = glob.glob(f"{self.file_path}.pileup_embed/{index}.*.pkl")
        assert len(embed_paths) == 1

        with open(embed_paths[0], "rb") as fi:
            pileup_embed = np.asarray(pickle.load(fi),dtype=np.float32)

        location_token = os.path.basename(embed_paths[0]).rsplit('.', 1)[0].split('.', 1)[1]
        chromosome, start = location_token.rsplit('_', 1)
        return sequence, (chromosome, int(start)), pileup_embed
