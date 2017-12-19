from kipoi.data import Dataset

import os
import numpy as np
import pandas as pd
import inspect
import pdb

filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))
# TODO - attach this_path to the pythonpathGXS
import sys
sys.path.append(this_path)

from gtf_utils import loadgene
from fasta_utils import FastaFile

class SplicingMaxEntDataset(Dataset):
    """
    Args:
        AS_gtf: alternative splicing gtf file. Can be dowloaded from MISO.
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in MISO summary format.
        iscounts: given targets are miso reads count.
        sep_intron_exon: will return exon, intron by separately in a list. Otherwise a whole piece of sequence with intron + overhang in both sides.
        overhang: length of overhang.
    """

    def __init__(self,
                 AS_gtf,
                 fasta_file,
                 side='5prime',  # 5prime/3prime
                 target_file=None,
                 select_fn=None,
                 label_col='event_name'
                 ):
        self.AS = loadgene(AS_gtf)
        self.fasta = FastaFile(fasta_file)
        if side in ["5prime", "3prime"]:
            self.side = side
        else:
            raise Exception("side should be 5prime or 3prime")
        if self.side == "5prime":
            self.overhang_l = 3
            self.overhang_r = 6
        else:
            self.overhang_l = 3
            self.overhang_r = 20
        if target_file is not None:
            self.Y = Target(target_file, label_col)
        else:
            self.Y = None
        self.select_fn = select_fn
        self._genes = None
        self._name = None
        self._species = None

    def __len__(self):
        return len(self.AS)

    def __getitem__(self, idx):
        gene = self.AS[idx]
        out = {}
        out['inputs'] = self.get_seq(gene)
        if self.Y is not None:
            out['targets'] = self.Y.get_target(gene.geneName)
        else:
            out['targets'] = np.nan
        out['metadata'] = {}
        out['metadata']['geneName'] = gene.geneName
        out['metadata']['chrom'] = gene.chrom
        out['metadata']['strand'] = gene.strand
        out['metadata']['start'] = gene.start
        out['metadata']['stop'] = gene.stop
        if self.select_fn is not None:
            # do it here for training. Because tensors are still numpy
            out = self.select_fn(out)
        return out

    def get_seq(self, gene, genomic_reorder=True):
        """ Get exon and intron sequences """
        exons = gene.get_all_exons()
        N_exon = exons.shape[0]
        introns = gene.get_all_introns()

        # Take intron coordinate
        # Try both normal gtf and AS_gtf

        # seq_range = introns + np.array([-self.overhang, self.overhang])
        if self.side == "5prime":
            if gene.strand == "+":
                seq_ranges = exons[0:N_exon - 1, 1].reshape(-1, 1) + np.array([-self.overhang_l + 1, self.overhang_r])
            else:
                seq_ranges = exons[1:, 0].reshape(-1, 1) + np.array([-self.overhang_r, self.overhang_l - 1])
        else:
            if gene.strand == "+":
                seq_ranges = exons[1:, 0].reshape(-1, 1) + np.array([-self.overhang_r, self.overhang_l - 1])
            else:
                seq_ranges = exons[0:N_exon - 1, 1].reshape(-1, 1) + np.array([-self.overhang_l + 1, self.overhang_r])
        seq = [self.fasta.get_seq(gene.chrom,
                                  seq_range,
                                  gene.strand)
               for seq_range in seq_ranges]
        if genomic_reorder:
            if gene.strand == "-":
                seq = seq[::-1]
        return {"seq": seq}

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, value):
        self._genes = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        self._species = value


class Target(object):
    """ Read (miso) target file, counts or PSI
    """

    def __init__(self,
                 target_file,
                 label_col='event_name',
                 iscounts=True):
        self.label_col = label_col
        self._read_target(target_file)

    def _read_target(self, target_file):
        dt = pd.read_csv(target_file, index_col=0)
        event_names = dt[self.label_col].tolist()
        self._index = event_names
        dt = dt.drop(self.label_col, axis=1)
        tissues = dt.columns
        dt = dt.as_matrix()
        dt = np.stack((dt, 1 - dt), axis=2)  # might bug if only one tissue
        self.target = dt
        self.tissues = tissues

    def get_target(self, name):
        try:
            inx = self._index.index(name)
            return self.target[inx]
        except:
            dim = self.target.shape
            return nans((dim[1:]))

def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a
