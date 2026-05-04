import torch
from torch.utils.data import Dataset
import dill as pickle
import numpy as np

#  TODO: make dataset_shared

class SeqTextDataset(Dataset):
    def __init__(self, file_path,tokenizer,flanking,center):
        # Load the data from the .pt file
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.flanking= flanking
        self.center= center
        self.sequences, self.labels = [], []
        with open(self.file_path, 'r') as file:
            while True:
                line1 = file.readline()
                line2 = file.readline()
                if not line1 or not line2:
                    break
                self.sequences.append(line1.strip())
                self.labels.append(line2.strip())

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.sequences)

    def __getitem__(self, index):
        sequence=self.sequences[index][self.flanking:self.flanking+self.center]
        sequence = self.tokenizer(sequence,
                add_special_tokens=False,
                padding='do_not_pad',
                truncation=False)
        sequence = np.asarray(sequence["input_ids"], dtype=np.int64)
        label =  np.asarray([float(char) for char in self.labels[index]], dtype=np.float32)

        with open(f'{self.file_path}.pileup_embed/{index}.pkl','rb') as fi:
            pileup_embed = np.asarray(pickle.load(fi),dtype=np.float32)
        return sequence, label, pileup_embed
