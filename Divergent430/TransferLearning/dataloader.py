from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool
from genomelake.extractors import FastaExtractor
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges
import linecache


class BedToolLinecache(BedTool):

    def __getitem__(self, idx):
        line = linecache.getline(self.fn, idx + 1)
        return pybedtools.create_interval_from_list(line.strip().split("\t"))


class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3+1 file containing intervals+labels
        fasta_file: file path; Genome sequence
    """

    def __init__(self, intervals_file, fasta_file):
        self.bt = BedToolLinecache(intervals_file)
        self.fasta_file = fasta_file
        self.fasta_extractor = None

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaExtractor(self.fasta_file)
        interval = self.bt[idx]

        # Intervals need to be 1000bp wide
        assert interval.stop - interval.start == 1000

        if interval.name is not None:
            y = np.array([float(interval.name)])
        else:
            y = {}

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]))

        # Reformat so that it matches the Basset shape
        # seq = np.swapaxes(seq, 1, 0)[:,:,None]
        return {
            "inputs": {"data/genome_data_dir": seq},
            "targets": y,
            "metadata": {
                "ranges": GenomicRanges.from_interval(interval)
            }
        }
