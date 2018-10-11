"""Dataloader
"""
import numpy as np
from kipoi.data import SampleIterator

import pickle
from pyfaidx import Fasta
import gffutils
from concise.preprocessing import encodeDNA
import warnings
from kipoi.metadata import GenomicRanges

from mmsplice import MMSplice
from mmsplice.vcf_dataloader import ExonInterval


class IntronDataLoader(SampleIterator):
    """
    Load genome annotation (gtf) file along with a vcf file, return wt sequence and mut sequence.
    Args:
    gtf: gtf file or pickled gtf IntervalTree.
    fasta_file: file path; Genome sequence
    vcf_file: file path; vcf file with variants to score
    """

    def __init__(self,
                 gtf_file,
                 fasta_file,
                 split_seq=True,
                 encode=True,
                 exon_cut_l=0,
                 exon_cut_r=0,
                 acceptor_intron_cut=6,
                 donor_intron_cut=6,
                 acceptor_intron_len=50,
                 acceptor_exon_len=3,
                 donor_exon_len=5,
                 donor_intron_len=13,
                 maxExonLength=200,
                 **kwargs
                 ):
        try:
            with open(gtf, 'rb') as f:
                self.exons = pickle.load(f)
        except:
            self.exonGenerator = self.GenerateExons(gtf_file, **kwargs)
        import six
        if isinstance(fasta_file, six.string_types):
            fasta = Fasta(fasta_file, as_raw=False)
        self.fasta = fasta
        
        self.encode = encode
        self.split_seq = split_seq
        self.exon_cut_l = exon_cut_l
        self.exon_cut_r = exon_cut_r
        self.acceptor_intron_cut = acceptor_intron_cut
        self.donor_intron_cut = donor_intron_cut
        self.acceptor_intron_len = acceptor_intron_len
        self.acceptor_exon_len = acceptor_exon_len
        self.donor_exon_len = donor_exon_len
        self.donor_intron_len = donor_intron_len
        self.maxExonLength = maxExonLength

    @staticmethod
    def GenerateExons(
        gtf_file,
        overhang=(100, 100),
        gtf_db_path=":memory:",
        disable_infer_transcripts = True,
        disable_infer_genes = True,
        firstLastNoExtend=True):
        ''' Generate EexonInterval objects from gtf file
        '''
        try:
            gtf_db = gffutils.interface.FeatureDB(gtf_db_path)
        except:
            gtf_db = gffutils.create_db(gtf_file,
                                      gtf_db_path,
                                      disable_infer_transcripts = disable_infer_transcripts,
                                      disable_infer_genes = disable_infer_genes)
        genes=gtf_db.features_of_type('gene')
        default_overhang = overhang
        for gene in genes:
            for exon in gtf_db.children(gene, featuretype = 'exon'): 
                isLast = False # track whether is last exon
                if firstLastNoExtend:
                    if (gene.strand == "+" and exon.end == gene.end) or (gene.strand == "-" and exon.start == gene.start):
                        overhang = (overhang[0],0)
                        isLast = True
                    elif (gene.strand == "+" and exon.start == gene.start) or (gene.strand == "-" and exon.end == gene.end):
                    #int(exon.attributes['exon_number'][0]) == 1:
                        overhang = (0,overhang[1])
                iv=ExonInterval.from_Feature(exon, overhang)
                iv.isLast = isLast
                overhang = default_overhang
                yield iv

    def __iter__(self):
        return self

    def __next__(self):
        ss = next(self.exonGenerator)
        out = {}
        out['inputs'] = {}
        seq = ss.get_seq(self.fasta).upper()
        if self.split_seq:
            seq = self.split(seq, ss.overhang)['acceptor'][0]
        out['inputs']['ss'] = seq
        
        out['metadata'] = {}
        out['metadata']['ranges'] = GenomicRanges(
            ss.chrom, 
            ss.Exon_Start,
            ss.Exon_End,
            ss.transcript_id,
            ss.strand)

        return out
    
    def batch_predict_iter(self, **kwargs):
        """Returns samples directly useful for prediction x["inputs"]
        Args:
          **kwargs: Arguments passed to self.batch_iter(**kwargs)
        """
        return (x for x in self.batch_iter(**kwargs))
    
    def split(self, x, overhang):
        ''' x: a sequence to split
        '''
        intronl_len, intronr_len = overhang
        lackl = self.acceptor_intron_len - intronl_len # need to pad N if left seq not enough long
        if lackl >= 0:
            x = "N"*(lackl+1) + x
            intronl_len += lackl+1
        lackr = self.donor_intron_len - intronr_len
        if lackr >= 0:
            x = x + "N"*(lackr+1)
            intronr_len += lackr + 1
        acceptor_intron = x[:intronl_len-self.acceptor_intron_cut]
        acceptor = x[(intronl_len-self.acceptor_intron_len) : (intronl_len+self.acceptor_exon_len)]
        exon = x[(intronl_len+self.exon_cut_l) : (-intronr_len-self.exon_cut_r)]
        donor = x[(-intronr_len-self.donor_exon_len) : (-intronr_len+self.donor_intron_len)]
        donor_intron = x[-intronr_len+self.donor_intron_cut:]
        if donor[self.donor_exon_len:self.donor_exon_len+2] != "GT":
            warnings.warn("None GT donor", UserWarning)
        if acceptor[self.acceptor_intron_len-2:self.acceptor_intron_len] != "AG":
            warnings.warn("None AG donor", UserWarning)

        if self.encode: 
            return {
                "acceptor_intron": encodeDNA([acceptor_intron]),
                "acceptor": encodeDNA([acceptor], maxlen=53),
                "exon": encodeDNA([exon]),
                "donor": encodeDNA([donor]),
                "donor_intron": encodeDNA([donor_intron])
            }
        else:
            return {
                "acceptor_intron": acceptor_intron,
                "acceptor": acceptor,
                "exon": exon,
                "donor": donor,
                "donor_intron": donor_intron
            }
