import numpy as np
from kipoi.data import SampleIterator

import pickle
from pyfaidx import Fasta
from cyvcf2 import VCF
from concise.preprocessing import encodeDNA
import warnings

from mmsplice import MMSplice
from mmsplice.vcf_dataloader import GenerateExonIntervalTree, VariantInterval, get_var_side

model = MMSplice(
    exon_cut_l=0,
    exon_cut_r=0,
    acceptor_intron_cut=6,
    donor_intron_cut=6,
    acceptor_intron_len=50,
    acceptor_exon_len=3,
    donor_exon_len=5,
    donor_intron_len=13)

class SplicingVCFDataloader(SampleIterator):
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
                 vcf_file,
                 split_seq=False,
                 encode=True,
                 exon_cut_l=0,
                 exon_cut_r=0,
                 acceptor_intron_cut=6,
                 donor_intron_cut=6,
                 acceptor_intron_len=50,
                 acceptor_exon_len=3,
                 donor_exon_len=5,
                 donor_intron_len=13,
                 **kwargs
                 ):
        try:
            with open(gtf, 'rb') as f:
                self.exons = pickle.load(f)
        except:
            self.exons = GenerateExonIntervalTree(gtf_file, **kwargs)
        import six
        if isinstance(fasta_file, six.string_types):
            fasta = Fasta(fasta_file, as_raw=False)
        self.fasta = fasta
        self.ssGenerator = self.spliceSiteGenerator(vcf_file, self.exons)
        
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

    @staticmethod
    def spliceSiteGenerator(vcf_file, exonTree, variant_filter=True):
        variants = VCF(vcf_file)
        for var in variants:
            if variant_filter and var.FILTER:
                next
            iv = VariantInterval.from_Variant(var)

            matches = map(lambda x: x.interval,
                          exonTree.intersect(iv, ignore_strand=True))

            for match in matches:
                side = get_var_side((
                    var.POS,
                    var.REF,
                    var.ALT,
                    match.Exon_Start,
                    match.Exon_End,
                    match.strand
                ))
                var = iv.to_Variant(match.strand, side)  # to my Variant class
                yield match, var


    def __iter__(self):
        return self

    def __next__(self):
        ss, var = next(self.ssGenerator)
        out = {}
        x = {}
        x['inputs'] = {}
        x['inputs_mut'] = {}
        seq = ss.get_seq(self.fasta).upper()
        mut_seq = ss.get_mut_seq(self.fasta, var).upper()
        if self.split_seq:
            seq = self.split(seq, ss.overhang)
            mut_seq = self.split(mut_seq, ss.overhang)
        x['inputs']['seq'] = seq
        x['inputs_mut']['seq'] = mut_seq
        x['inputs']['intronl_len'] = ss.overhang[0]
        x['inputs']['intronr_len'] = ss.overhang[1]
        x['inputs_mut']['intronl_len'] = ss.overhang[0]
        x['inputs_mut']['intronr_len'] = ss.overhang[1]
        
        out['inputs'] = np.concatenate([model.predict(x['inputs_mut']).values, model.predict(x['inputs']).values])
        
        out['metadata'] = {}
        out['metadata']['ranges'] = ss.grange
        out['metadata']['variant'] = var.to_dict
        out['metadata']['ExonInterval'] = ss.to_dict # so that np collate will work
        out['metadata']['annotation'] = str(ss)
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
                "acceptor": encodeDNA([acceptor]),
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
