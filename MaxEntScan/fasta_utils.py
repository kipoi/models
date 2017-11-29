import pysam

class FastaFile:
    """docstring for FastaFile"""
    def __init__(self, fasta_file):
        self.f = pysam.FastaFile(fasta_file)

    def get_seq(self, chrom, start_stop, strand):
        """get the sequence in a given region, the start is from 1.
        The start and stop index may still need double check."""
        start, stop = start_stop
        seq = self.f.fetch(chrom, start-1, stop)
        if strand == "-":
            seq = self.rev_seq(seq)
        return seq
    

    def rev_seq(self, seq):
        """ reverse complement """
        _tmp = []
        _tmp[:] = seq
        for j in range(len(_tmp)):
            if _tmp[j] == "A": _tmp[j] = "T"
            elif _tmp[j] == "T": _tmp[j] = "A"
            elif _tmp[j] == "G": _tmp[j] = "C"
            elif _tmp[j] == "C": _tmp[j] = "G"
        RV = "".join(_tmp[::-1])
        return RV
