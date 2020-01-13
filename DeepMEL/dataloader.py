from __future__ import absolute_import, division, print_function
import numpy as np
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges
from kipoiseq.dataloaders.sequence import BedDataset
from kipoiseq.extractors import FastaStringExtractor
from kipoiseq.transforms import OneHot
from kipoiseq.transforms.functional import resize_interval


class MyDataset(Dataset):
    """Example re-implementation of kipoiseq.dataloaders.SeqIntervalDl

    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
    """

    def __init__(self, intervals_file, fasta_file, ignore_targets=True):
        self.bt = BedDataset(intervals_file,
                             bed_columns=3,
                             ignore_targets=ignore_targets)
        self.fasta_file = fasta_file
        self.fasta_extractor = None
        self.transform = OneHot()  # one-hot encode DNA sequence

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        self.fasta_extractor = FastaStringExtractor(self.fasta_file)

        # get the intervals
        interval, targets = self.bt[idx]

        # resize to 500bp
        interval = resize_interval(interval, 500, anchor='center')

        # extract the sequence
        seq = self.fasta_extractor.extract(interval)

        # one-hot encode the sequence
        seq_onehot = self.transform(seq)
        seq_onehot_rc = seq_onehot[::-1, ::-1]
        
        ranges = GenomicRanges.from_interval(interval)
        ranges_rc = GenomicRanges.from_interval(interval)

        return {
            "inputs": [seq_onehot, seq_onehot_rc],
            "metadata":  [ranges, ranges_rc]
        }
