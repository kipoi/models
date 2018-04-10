"""lsgkm-SVM dataloader
"""
# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool
from pysam import Fastafile
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges
import linecache
# --------------------------------------------

def rc_str(dna):
    """Reverse complement a string
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    ks = list(complement.keys())
    for k in ks:
        complement[k.lower()] = complement[k].lower()
    return ''.join([complement[base] for base in dna[::-1]])


class BedToolLinecache(BedTool):
    """Fast BedTool accessor by Ziga Avsec

    Normal BedTools loops through the whole file to get the
    line of interest. Hence the access it o(n)
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

    def __init__(self, intervals_file, fasta_file, use_linecache=False):

        # intervals
        if use_linecache:
            self.bt = BedToolLinecache(intervals_file)
        else:
            self.bt = BedTool(intervals_file)
        self.fasta = Fastafile(fasta_file)

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        interval = self.bt[idx]

        # Intervals can't be bigger than 1000bp
        if (interval.stop - interval.start) > 1000:
            raise Exception("Input sequences should be at maximum 1000bp.")

        # Fetch the fasta line
        seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)

        # Reverse complement input string is requested
        if interval.strand == "-":
            seq = rc_str(seq)

        """
        # generate an id
        id = str(interval.chrom) + ":" + str(interval.start) + "-" + str(interval.stop)
        if interval.name not in ["", ".", "*"]:
            id = interval.name
        """

        return {
            "inputs": seq,
            "metadata": {
                "ranges": GenomicRanges.from_interval(interval)
            }
        }
