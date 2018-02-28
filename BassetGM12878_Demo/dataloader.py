from __future__ import absolute_import, division, print_function
import six
import os
import inspect
from builtins import str, open, range, dict

import pickle
import numpy as np
import pybedtools

from genomelake.extractors import FastaExtractor
from kipoi.metadata import GenomicRanges
from kipoi.data import Dataset

filename = inspect.getframeinfo(inspect.currentframe()).filename
DATALOADER_DIR = os.path.dirname(os.path.abspath(filename))

# File paths
intervals_file = "example_files/intervals_files_ENCSR000EMT_chr21_10000.tsv"
fasta_file = "example_files/chr21.fa"

class FastaDataset(Dataset):
    """
    Args:
        intervals_file: file path; tsv file
            Assumes bed-like `chrom start end label` format.
        fasta_file: file path; Genome sequence
    """

    def __init__(self, intervals_file, fasta_file):
        # intervals
        self.bt = pybedtools.BedTool(intervals_file)

        # extractors
        self.input_data_extractors = {"data/genome_data_dir": FastaExtractor(fasta_file)}

        self.target_dataset = None

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        interval = self.bt[idx]

        out = {}

        out['inputs'] = {key: np.squeeze(extractor([interval]), axis=0)
                         for key, extractor in self.input_data_extractors.items()}

        
        #if self.target_dataset is not None:
        out["targets"] = np.array([float(interval.name)])

        # get metadata
        out['metadata'] = {}
        out['metadata']['ranges'] = {}
        out['metadata']['ranges']['chr'] = interval.chrom
        out['metadata']['ranges']['start'] = interval.start
        out['metadata']['ranges']['end'] = interval.stop
        out['metadata']['ranges']['id'] = interval.name
        #out['metadata']['ranges']['strand'] = interval.strand

        return out

# test batching
# from torch.utils.data import DataLoader

# dl = DataLoader(a, batch_size=3, collate_fn=numpy_collate)

# it = iter(dl)
# sample = next(it)

# sample["inputs"]["dist_polya_st"].shape
# sample["inputs"]["seq"].shape

