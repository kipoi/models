"""DeepCpG dataloader
"""
# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from pybedtools import BedTool
from genomelake.extractors import FastaExtractor
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges

# --------------------------------------------


class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
    """

    SEQ_WIDTH = 1001

    def __init__(self, intervals_file, fasta_file):

        # intervals
        self.bt = BedTool(intervals_file)
        self.fasta_file = fasta_file
        self.fasta_extractor = None

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaExtractor(self.fasta_file)
        interval = self.bt[idx]

        if interval.stop - interval.start != self.SEQ_WIDTH:
            raise ValueError("Expected the interval to be {0} wide. Recieved stop - start = {1}".
                             format(self.SEQ_WIDTH, interval.stop - interval.start))

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]), axis=0)
        return {
            "inputs": {"dna": seq},
            "metadata": {
                "ranges": GenomicRanges.from_interval(interval)
            }
        }
