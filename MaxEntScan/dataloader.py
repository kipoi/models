from kipoi.data import Dataset, kipoi_dataloader
from kipoi.metadata import GenomicRanges

import sys
import os
import numpy as np
import pandas as pd
import inspect
import itertools
import inspect
import sys

filename = inspect.getframeinfo(inspect.currentframe()).filename
this_dir = os.path.dirname(os.path.abspath(filename))
sys.path.append(this_dir)


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


@kipoi_dataloader()
class SplicingMaxEntDataset(Dataset):
    """
    args:
      MISO_AS:
        doc: Whether the given annotation file is MISO alternative splicing annotation. default False.
      fasta_file:
        doc: Reference Genome sequence in fasta format
        example:
          md5: 936544855b253835442a0f253dd4b083
          url: https://zenodo.org/record/1466099/files/3prime-example_files-hg19.chr22.fa?download=1
        type: str
      gtf_file:
        doc: file path; Genome annotation GTF file
        example:
          md5: 174fd11303ae2c2369094bfcbe303c07
          url: https://zenodo.org/record/1466099/files/3prime-example_files-hg19.chr22.gtf?download=1
      label_col:
        doc: response label column name
      target_file:
        doc: path to the targets (txt) file
        optional: true
      side:
        doc: 5 or 3prime splice-site
    dependencies:
      conda:
        - pysam=0.15.2
        - python=3.8
    info:
      authors:
        - github: s6juncheng
          name: Jun Cheng
      doc: MaxEnt Splicing Model
      name: MaxEnt
      version: 0.1
    output_schema:
      inputs:
        associated_metadata: ranges
        doc: a junction (donor or acceptor) sequence
        name: seq
        shape: ()
        special_type: DNAStringSeq
      metadata:
        biotype:
          doc: gene biotype, can be used to filter protein coding genes for instance
          type: str
        geneID:
          doc: gene ID
          type: str
        order:
          doc: order of the donor site in the transcript, counted from 5' to 3'.
          type: int
        ranges:
          doc: ranges that the sequences were extracted
          type: GenomicRanges
        transcriptID:
          doc: transcript id
          type: str
      targets:
        doc: Predicted psi
        name: psi
        shape: (1,)
    """

    def __init__(self,
                 gtf_file,
                 fasta_file,
                 side='5prime',  # 5prime/3prime
                 target_file=None,
                 MISO_AS=False,
                 label_col='event_name'
                 ):
        from gtf_utils import loadgene

        self.genes = loadgene(gtf_file)
        self.fasta_file = fasta_file
        self.fasta = None  # open the file later

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
        if self.fasta is None:
            from fasta_utils import FastaFile
            self.fasta = FastaFile(self.fasta_file)

        out = {}
        if self.MISO_AS:
            gene = self.genes[idx]
            inputs, ranges = self.get_seq(gene)
            out['inputs'] = inputs
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
            out['metadata']['extracted_regions'] = ranges

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
        """ Get exon and intron sequences """
        exons = gene.get_all_exons()
        # N_exon = exons.shape[0]
        introns = gene.get_all_introns()

        # Take intron coordinate
        # Try both normal gtf and AS_gtf

        if self.side == "5prime":
            if gene.strand == "+":
                seq_ranges = exons[:-1, 1].reshape(-1, 1) + np.array([-self.overhang_l + 1, self.overhang_r])
            else:
                seq_ranges = exons[:-1, 0].reshape(-1, 1) + np.array([-self.overhang_r, self.overhang_l - 1])
        else:
            if gene.strand == "+":
                seq_ranges = exons[1:, 0].reshape(-1, 1) + np.array([-self.overhang_r, self.overhang_l - 1])
            else:
                seq_ranges = exons[1:, 1].reshape(-1, 1) + np.array([-self.overhang_l + 1, self.overhang_r])

        seq = [self.fasta.get_seq(gene.chrom,
                                  seq_range,
                                  gene.strand)
               for seq_range in seq_ranges]
        return np.array(seq), seq_ranges

    def _get_spliceSites(self, gene):
        ''' Get splice site sequence for all transcripts of a single gene.
        Applied for normal gtf annotation.
        '''
        spliceSites = []
        for transcript in gene.trans:
            exons = transcript.exons
            ind = np.lexsort((exons[:, 1], exons[:, 0]))
            if len(exons) > 1:
                if self.side == "5prime":
                    if gene.strand == "+":
                        seq_ranges = exons[:-1, 1].reshape(-1, 1) + np.array([-self.overhang_l + 1, self.overhang_r])
                    else:
                        ind = ind[::-1]
                        exons = exons[ind]
                        seq_ranges = exons[:-1, 0].reshape(-1, 1) + np.array([-self.overhang_r, self.overhang_l - 1])
                else:
                    if gene.strand == "+":
                        seq_ranges = exons[1:, 0].reshape(-1, 1) + np.array([-self.overhang_r, self.overhang_l - 1])
                    else:
                        ind = ind[::-1]
                        exons = exons[ind]
                        seq_ranges = exons[1:, 1].reshape(-1, 1) + np.array([-self.overhang_l + 1, self.overhang_r])

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
        except Exception:
            dim = self.target.shape
            return nans((dim[1:]))


def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a
