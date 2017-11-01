from kipoi.data import Dataset
from gtf_utils import loadgene
from fasta_utils import FastaFile
import numpy as np
import pandas as pd

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
                 target_file=None,
                 iscounts=False,
                 select_fn=None,
                 label_col='event_name',
                 sep_intron_exon=False,
                 overhang=80,  # intron + ~ bp exon from both side
                 ** kwargs):
        self.AS = loadgene(AS_gtf, **kwargs)
        self.fasta = FastaFile(fasta_file)
        self.sep_intron_exon = sep_intron_exon
        self.overhang = overhang
        self.iscounts = iscounts
        if target_file is not None:
            self.Y = Target(target_file, label_col, self.iscounts)
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
            out['target'] = self.Y.get_target(gene.geneName)
        else:
            out['target'] = np.nan
        # out['metadata'] = gene.__dict__  # gene is a Gene class instance, but torch Dataloader don't like it.
        # out['metadata']['trans'] = [i.__dict__ for i in gene.trans]
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
        introns = gene.get_all_introns()
        if self.sep_intron_exon:
            exon_seq = [self.fasta.get_seq(gene.chrom,
                                           exons[i, :],
                                           gene.strand)
                        for i in range(exons.shape[0])]
            intron_seq = [self.fasta.get_seq(gene.chrom,
                                             introns[i, :],
                                             gene.strand)
                          for i in range(introns.shape[0])]
            if genomic_reorder:
                if gene.strand == "-":
                    exon_seq = exon_seq[::-1]
                    intron_seq = intron_seq[::-1]

            return {"exons": exon_seq,
                    "introns": intron_seq}
        else:
            # seq_range = introns + np.array([-self.overhang, self.overhang])
            if gene.strand == "+":
                seq_range = exons[0:2,1] + np.array([-self.overhang+1, self.overhang])
            else:
                seq_range = exons[1:,0] + np.array([-self.overhang, self.overhang-1])
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


class Target(object):
    """ Read (miso) target file, counts or PSI
    """

    def __init__(self,
                 target_file,
                 label_col='event_name',
                 iscounts=True):
        self.label_col = label_col
        self.iscounts = iscounts
        self._read_target(target_file)

    def _read_target(self, target_file):
        dt = pd.read_csv(target_file, index_col=0)
        event_names = dt[self.label_col].tolist()
        self._index = event_names
        dt = dt.drop(self.label_col, axis=1)
        tissues = dt.columns
        if self.iscounts:
            tissues = tissues.str.replace("(k_|n_)", "").unique().tolist()
            try:
                counts = [dt[['k_' + ts, 'n_' + ts]].as_matrix() for ts in tissues]
            except:
                print("Check whether target_file indeed has counts as targets named as k_ or n_.")
                raise
            # pdb.set_trace()
            self.target = np.stack(counts, axis=2)
        else:
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
