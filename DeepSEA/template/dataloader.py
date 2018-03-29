"""DeepSEA dataloader
"""
# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool
from genomelake.extractors import BaseExtractor
from genomelake.util import one_hot_encode_sequence
from pysam import FastaFile
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges
import linecache
# --------------------------------------------

NUM_SEQ_CHARS = 4


class BedToolLinecache(BedTool):
    """Fast BedTool accessor by Ziga Avsec

    Normal BedTools loops through the whole file to get the
    line of interest. Hence the access it o(n)
    """

    def __getitem__(self, idx):
        line = linecache.getline(self.fn, idx + 1)
        return pybedtools.create_interval_from_list(line.strip().split("\t"))

class FasterFastaExtractor(BaseExtractor):
    '''Same as the FastaExtractor from genomelake.extractors with
       the only difference that the file object (instance of pysam.FastaFile)
       is being kept open, to reduce opening the same file repetitively.

    '''

    def __init__(self, datafile, use_strand=False, **kwargs):
        super(FasterFastaExtractor, self).__init__(datafile, **kwargs)
        self.use_strand = use_strand
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):
        fasta = self.fasta
        for index, interval in enumerate(intervals):
            seq = fasta.fetch(str(interval.chrom), interval.start,
                              interval.stop)
            one_hot_encode_sequence(seq, out[index, :, :])

            # reverse-complement seq the negative strand
            if self.use_strand and interval.strand == "-":
                out[index, :, :] = out[index, ::-1, ::-1]

        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        return (num_intervals, width, NUM_SEQ_CHARS)

class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
    """

    SEQ_WIDTH = 1000

    def __init__(self, intervals_file, fasta_file,
                 target_file=None, use_linecache=False):

        # intervals
        if use_linecache:
            self.bt = BedToolLinecache(intervals_file)
        else:
            self.bt = BedTool(intervals_file)
        self.fasta_extractor = FasterFastaExtractor(fasta_file)

        # Targets
        if target_file is not None:
            self.targets = pd.read_csv(target_file)
        else:
            self.targets = None

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        interval = self.bt[idx]

        if interval.stop - interval.start != self.SEQ_WIDTH:
            raise ValueError("Expected the interval to be {0} wide. Recieved stop - start = {1}".
                             format(self.SEQ_WIDTH, interval.stop - interval.start))

        if self.targets is not None:
            y = self.targets.iloc[idx].values
        else:
            y = {}

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]), axis=0)
        # Reformat so that it matches the DeepSEA shape
        seq = np.swapaxes(seq, 1, 0)[:, None, :]
        return {
            "inputs": seq,
            "targets": y,
            "metadata": {
                "ranges": GenomicRanges.from_interval(interval)
            }
        }
