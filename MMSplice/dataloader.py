from mmsplice.exon_dataloader import SeqSpliter
from mmsplice.vcf_dataloader import SplicingVCFDataloader as BaseVCFDl


class SplicingVCFDataloader(BaseVCFDl):

    def __init__(self, gtf, fasta_file, vcf_file,
                 split_seq=True, encode=True,
                 exon_cut_l=0, exon_cut_r=0,
                 acceptor_intron_cut=6, donor_intron_cut=6,
                 acceptor_intron_len=50, acceptor_exon_len=3,
                 donor_exon_len=5, donor_intron_len=13):
        seq_spliter = SeqSpliter(
            exon_cut_l=exon_cut_l, exon_cut_r=exon_cut_r,
            acceptor_intron_cut=acceptor_intron_cut,
            donor_intron_cut=donor_intron_cut,
            acceptor_intron_len=acceptor_intron_len,
            acceptor_exon_len=acceptor_exon_len,
            donor_exon_len=donor_exon_len,
            donor_intron_len=donor_intron_len)

        super().__init__(gtf, fasta_file, vcf_file,
                         split_seq=split_seq, encode=encode,
                         seq_spliter=seq_spliter)
