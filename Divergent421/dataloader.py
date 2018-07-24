from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool, Interval
from genomelake.extractors import FastaExtractor
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges


class TsvReader:
    """Reads a tsv file in the following format:

    chr  start  stop  task1  task2 ...

    Args:
      tsv_file: tsv file type
      label_dtype: data type of the labels
    """

    def __init__(self, tsv_file, num_chr=False, label_dtype=None):
        self.tsv_file = tsv_file
        self.num_chr = num_chr
        self.label_dtype = label_dtype
        df_peek = pd.read_table(self.tsv_file,
                                header=None,
                                nrows=1,
                                sep='\t')
        self.n_tasks = df_peek.shape[1] - 3
        assert self.n_tasks >= 0
        self.df = pd.read_table(self.tsv_file,
                                header=None,
                                dtype={i: d
                                       for i, d in enumerate([str, int, int] +
                                                             [self.label_dtype] * self.n_tasks)},
                                sep='\t')
        if self.num_chr and self.df.iloc[0][0].startswith("chr"):
            self.df[0] = self.df[0].str.replace("^chr", "")
        if not self.num_chr and not self.df.iloc[0][0].startswith("chr"):
            self.df[0] = "chr" + self.df[0]

    def __getitem__(self, idx):
        """Returns (pybedtools.Interval, labels)
        """
        row = self.df.iloc[idx]
        interval = Interval(row[0], row[1], row[2])

        if self.n_tasks == 0:
            labels = {}
        else:
            labels = row.iloc[3:].values.astype(self.label_dtype)
        return interval, labels

    def __len__(self):
        return len(self.df)


class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3+<columns> file containing intervals+labels
        fasta_file: file path; Genome sequence
        label_dtype: label data type
        num_chr_fasta: if True, the tsv-loader will make sure that the chromosomes
          don't start with chr
    """

    def __init__(self, intervals_file, fasta_file, num_chr_fasta=False):
        self.num_chr_fasta = num_chr_fasta
        self.intervals_file = intervals_file
        self.fasta_file = fasta_file

        self.tsv = TsvReader(self.intervals_file,
                             num_chr=self.num_chr_fasta,
                             label_dtype=bool)
        self.fasta_extractor = None

    def __len__(self):
        return len(self.tsv)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaExtractor(self.fasta_file)

        interval, labels = self.tsv[idx]

        # Intervals need to be 1000bp wide
        assert interval.stop - interval.start == 1000

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]))

        return {
            "inputs": {"data/genome_data_dir": seq},
            "targets": labels,
            "metadata": {
                "ranges": GenomicRanges(interval.chrom, interval.start, interval.stop, str(idx))
            }
        }
