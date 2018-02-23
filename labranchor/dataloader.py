import numpy as np
from kipoi.data import Dataset
import numpy as np
import itertools
from tqdm import tqdm
import inspect
import os
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))
import sys
sys.path.append(this_path)
from gtf_utils import loadgene
from fasta_utils import FastaFile


bases = ['A', 'C', 'G', 'T']


def onehot(seq):
    X = np.zeros((len(seq), len(bases)))
    for i, char in enumerate(seq):
        X[i, bases.index(char.upper())] = 1
    return X


class Branch(object):
    """ Region that consider to contains branchpoint branch
    """

    def __init__(self,
                 chrom,
                 start,
                 stop,
                 strand,
                 transcript_id,
                 gene_id,
                 biotype):

        self.chrom = chrom
        self.grange = (start, stop)
        self.strand = strand
        self.transcriptID = transcript_id
        self.geneID = gene_id
        self.biotype = biotype

    def get_branch_seq(self, fasta):
        """ Get branch sequence
        branch_seq: sequence of acceptor intron that potentially 
        has branchpoint
        """
        seq = fasta.get_seq(self.chrom,
                            self.grange,
                            self.strand)
        self.seq = onehot(seq)


class BranchPointDataset(Dataset):
    """
    Args:
    gtf_file: file path; Genome annotation GTF file.
    fasta_file: file path; Genome sequence
    length: length of branchpoint region, count upstream from 3' ss
    """

    def __init__(self,
                 gtf_file,
                 fasta_file,
                 length=70):
        self.genes = loadgene(gtf_file)
        self.fasta = FastaFile(fasta_file)
        self.length = length
        self.branches = self.get_branches()

    def __len__(self):
        return len(self.branches)

    def __getitem__(self, idx):
        """
        Return a list of Branch objects. They contain coordinates that can be
        written to bed files
        """
        out = {}
        out['inputs'] = {}
        branch = self.branches[idx]

        # input sequence
        out['inputs']['bidirectional_1_input'] = branch.seq

        # metadata
        out['metadata'] = {}
        out['metadata']['geneID'] = branch.geneID
        out['metadata']['transcriptID'] = branch.transcriptID
        out['metadata']['chrom'] = branch.chrom
        out['metadata']['strand'] = branch.strand
        out['metadata']['start'] = branch.grange[0] - 1  # use 0-base indexing
        out['metadata']['stop'] = branch.grange[1]
        out['metadata']['biotype'] = branch.biotype

        return out

    def _get_branches(self, gene):
        """ get_branches for a single gene
        """
        branchList = []
        for transcript in gene.trans:
            exons = transcript.exons
            if len(exons) > 1:
                if gene.strand == "+":
                    ss3 = exons[1:, 0]
                    ss3 = np.stack((ss3 - self.length, ss3 - 1), -1)
                else:
                    exons = exons[::-1]
                    ss3 = exons[:-1, 1]
                    ss3 = np.stack((ss3 + 1, ss3 + self.length), -1)
                for e in ss3:
                    try:
                        branch = Branch(gene.chrom,
                                        e[0],
                                        e[1],
                                        gene.strand,
                                        transcript.tranID,
                                        gene.geneID,
                                        gene.biotype)
                        branch.get_branch_seq(self.fasta)
                        branchList.append(branch)
                    except:
                        print("failed to append the brachpoint: {0}".format(branch))
                        print("{0}:{1}-{2} {3} {4}".format(gene.chrom, e[0], e[1], gene.strand, transcript.tranID))
        return branchList

    def get_branches(self):
        """ Get brachpoint regions of all acceptor exons into Branch object.
        First exon of the transcript is not an acceptor exon
        """
        # branches = [self.get_branches(g) for g in self.genes]
        branches = list(map(self._get_branches, self.genes))
        branches = list(itertools.chain.from_iterable(branches))
        return branches

