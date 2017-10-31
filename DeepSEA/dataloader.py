"""DeepSEA dataloader
"""
# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from pybedtools import BedTool
from genomelake.extractors import FastaExtractor
from kipoi.data import Dataset

# --------------------------------------------


class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
    """

    def __init__(self, intervals_file, fasta_file, target_file=None):

        # intervals
        self.bt = BedTool(intervals_file)
        self.fasta_extractor = FastaExtractor(fasta_file)

        # Targets
        if target_file is not None:
            self.targets = pd.read_csv(target_file)
        else:
            self.targets = None

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        interval = self.bt[idx]

        # Intervals need to be 1000bp wide
        assert interval.stop - interval.start == 1000

        if self.targets is not None:
            y = self.targets.iloc[idx].values
        else:
            y = {}

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]), axis=0)
        return {
            # TODO - how to refect this in the yaml file?
            #        schema.feed_type: list # dict, single_array?
            "inputs": [seq],
            "targets": {
                "epigen_mod": y
            },
            "metadata": {
                "ranges": {
                    "chr": interval.chrom,
                    "start": interval.start,
                    "end": interval.stop,
                    "id": interval.name,
                    "strand": interval.strand,
                }
            }
        }
