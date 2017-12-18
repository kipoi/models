from kipoi.data import Dataset
import numpy as np
import pandas as pd

import inspect
import os
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))
# TODO - attach this_path to the pythonpathGXS
import sys
sys.path.append(this_path)
from gtf_utils import loadgene
from fasta_utils import FastaFile

class SplicingKmerDataset(Dataset):
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
                 overhang=80):  # intron + ~ bp exon from both side
        self.AS = loadgene(AS_gtf)
        self.fasta = FastaFile(fasta_file)
        self.overhang = overhang
        self._genes = None
        self._name = None
        self._species = None

    def __len__(self):
        return len(self.AS)

    def __getitem__(self, idx):
        gene = self.AS[idx]
        out = {}
        out['inputs'] = self.get_seq(gene)
        out['metadata'] = {}
        out['metadata']['geneName'] = gene.geneName
        out['metadata']['chrom'] = gene.chrom
        out['metadata']['strand'] = gene.strand
        out['metadata']['start'] = gene.start
        out['metadata']['stop'] = gene.stop
        return out

    def get_seq(self, gene, genomic_reorder=True):
        """ Get exon and intron sequences """
        exons = gene.get_all_exons()
        if gene.strand == "+":
            seq_range = exons[0:2, 1] + np.array([-self.overhang + 1, self.overhang])
        else:
            seq_range = exons[1:, 0] + np.array([-self.overhang, self.overhang - 1])
        seq = self.fasta.get_seq(gene.chrom,
                                 seq_range,
                                 gene.strand)
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
