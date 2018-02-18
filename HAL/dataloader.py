from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges

import numpy as np
import pandas as pd
import itertools

import inspect
import os
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))
# TODO - attach this_path to the pythonpathGXS
import sys
sys.path.append(this_path)
from gtf_utils import loadgene
from fasta_utils import FastaFile

class SpliceSite(object):
    ''' A splice site with flanking intron and exon sequence
    Args:
        order: order of splice site (donor or acceptor) in a transcript counted from 5' to 3'. 
    '''

    def __init__(self,
                 chrom,
                 start,
                 stop,
                 strand,
                 transcript_id,
                 gene_id,
                 biotype,
                 order=None):
        self.chrom = chrom
        self.grange = (start, stop)
        self.strand = strand
        self.transcriptID = transcript_id
        self.geneID = gene_id
        self.biotype = biotype
        self.order = order
        self._seq = None

    @property
    def seq(self):
        return self._seq

    @seq.setter
    def seq(self, value):
        self._seq = value

    def get_seq(self, fasta):
        seq = fasta.get_seq(self.chrom,
                            self.grange,
                            self.strand)
        return seq

class SplicingKmerDataset(Dataset):
    """
    Args:
        gtf_file: gtf file. Can be dowloaded from MISO or ensembl.
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in MISO summary format.
        overhang: length of overhang.
        MISO_AS: whether the used annotation file is from MISO alternative splicing annotation.
    """

    def __init__(self,
                 gtf_file,
                 fasta_file,
                 overhang=80,
                 MISO_AS=False):  # intron + ~ bp exon from both side
        self.genes = loadgene(gtf_file)
        self.fasta = FastaFile(fasta_file)
        self.overhang = overhang
        self.MISO_AS = MISO_AS
        if not MISO_AS:
            self.spliceSites = self.get_spliceSites()
        self._name = None
        self._species = None

    def __len__(self):
        if self.MISO_AS:
            return len(self.genes)
        else:
            return len(self.spliceSites)

    def __getitem__(self, idx):
        out = {}

        if self.MISO_AS:
            gene = self.genes[idx]
            out['inputs'] = self.get_seq(gene)
            out['metadata'] = {}
            out['metadata']['geneName'] = gene.geneName
            out['metadata']['chrom'] = gene.chrom
            out['metadata']['strand'] = gene.strand
            out['metadata']['start'] = gene.start
            out['metadata']['stop'] = gene.stop

        else:
            spliceSite = self.spliceSites[idx]
            out['inputs'] = spliceSite.get_seq(self.fasta)
            out['metadata'] = {}
            out['metadata']['geneID'] = spliceSite.geneID
            out['metadata']['transcriptID'] = spliceSite.transcriptID
            out['metadata']['biotype'] = spliceSite.biotype
            out['metadata']['order'] = spliceSite.order
            out['metadata']['ranges'] = GenomicRanges(spliceSite.chrom,
                                                      spliceSite.grange[0] - 1,  # use 0-base indexing
                                                      spliceSite.grange[1],
                                                      spliceSite.geneID,
                                                      spliceSite.strand)
        return out

    def get_seq(self, gene):
        """ Get splice site sequence with flanking exon and intron sequences. 
        This function is applied for MISO annotation with 3 exon alternative splicing model.
        """
        exons = gene.get_all_exons()
        # N_exon = exons.shape[0]
        if gene.strand == "+":
            seq_ranges = exons[:-1, 1].reshape(-1, 1) + np.array([-self.overhang + 1, self.overhang])
        else:
            seq_ranges = exons[:-1, 0].reshape(-1, 1) + np.array([-self.overhang, self.overhang - 1])
        seq = [self.fasta.get_seq(gene.chrom,
                                  seq_range,
                                  gene.strand)
               for seq_range in seq_ranges]
        return np.array(seq)

    def _get_spliceSites(self, gene):
        ''' Get splice site sequence for all transcripts of a single gene.
        Applied for normal gtf annotation.
        '''
        spliceSites = []
        for transcript in gene.trans:
            exons = transcript.exons
            ind = np.lexsort((exons[:, 1], exons[:, 0]))
            if len(exons) > 1:
                if gene.strand == "+":
                    seq_ranges = exons[:-1, 1].reshape(-1, 1) + np.array([-self.overhang + 1, self.overhang])
                else:
                    ind = ind[::-1]
                    exons = exons[ind]
                    seq_ranges = exons[:-1, 0].reshape(-1, 1) + np.array([-self.overhang, self.overhang - 1])
                for i in range(seq_ranges.shape[0]):
                    spliceSite = SpliceSite(gene.chrom,
                                            seq_ranges[i, 0],
                                            seq_ranges[i, 1],
                                            gene.strand,
                                            transcript.tranID,
                                            gene.geneID,
                                            gene.biotype,
                                            i)
                    # can call get_seq later in iterator to save memory
                    # spliceSite.seq = spliceSite.get_seq(self.fasta)
                    spliceSites.append(spliceSite)
        return spliceSites

    def get_spliceSites(self):
        ''' Get splice sites for all donors
        '''
        spliceSites = list(map(self._get_spliceSites, self.genes))
        spliceSites = list(itertools.chain.from_iterable(spliceSites))
        return spliceSites

    @property
    def name(self):
        return self._namem

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        self._species = value
