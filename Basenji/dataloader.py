"""Basenji dataloader
"""
# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool
from genomelake.extractors import FastaExtractor
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges
import linecache
# --------------------------------------------


class BedToolLinecache(BedTool):
    """Faster BedTool accessor by Ziga Avsec

    Normal BedTools loops through the whole file to get the
    line of interest. Hence the access it o(n)

    Note: this might load the whole bedfile into memory
    """

    def __getitem__(self, idx):
        line = linecache.getline(self.fn, idx + 1)
        return pybedtools.create_interval_from_list(line.strip().split("\t"))


class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
    """

    def __init__(self, intervals_file, fasta_file,
                 use_linecache=False):

        # intervals
        if use_linecache:
            self.bt = BedToolLinecache(intervals_file)
        else:
            self.bt = BedTool(intervals_file)
        self.fasta_extractor = FastaExtractor(fasta_file)

        if len(self.bt) % 2 == 1:
            raise ValueError("Basenji strictly requires batch_size=2," +
                             " hence the bed file should have an od length")

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        interval = self.bt[idx]

        # Intervals need to be 131072bp wide
        assert interval.stop - interval.start == 131072

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]), axis=0)
        return {
            "inputs": seq,
            "targets": {},  # No Targets
            "metadata": {
                "ranges": GenomicRanges.from_interval(interval)
            }
        }
