from kipoi.data import Dataset
import numpy as np
import pandas as pd
import sys
import pysam

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

class Transcript:
    def __init__(self, chrom, strand, start, stop, tran_id, tran_name="*",
                 biotype="*"):
        """a general purpose transcript object with the basic information.
        """
        self.chrom = chrom
        self.strand = strand
        self.start = int(start)
        self.stop = int(stop)
        self.tranID = tran_id
        self.exons = np.zeros((0, 2), "int")
        self.seglen = None
        self.tranL = 0
        self.exonNum = 0
        self.biotype = biotype
        self.tranName = tran_name
        self.introns = None

    def add_exon(self, chrom, strand, start, stop):
        if strand != self.strand or chrom != self.chrom:
            print("The exon has different chrom or strand to the transcript.")
            return
        _exon = np.array([start, stop], "int").reshape(1, 2)
        self.exons = np.append(self.exons, _exon, axis=0)
        self.exons = np.sort(self.exons, axis=0)
        self.tranL += abs(int(stop) - int(start) + 1)
        self.exonNum += 1

        self.seglen = np.zeros(self.exons.shape[0] * 2 - 1, "int")
        self.seglen[0] = self.exons[0, 1] - self.exons[0, 0] + 1
        for i in range(1, self.exons.shape[0]):
            self.seglen[i * 2 - 1] = self.exons[i, 0] - self.exons[i - 1, 1] - 1
            self.seglen[i * 2] = self.exons[i, 1] - self.exons[i, 0] + 1

        if ["-", "-1", "0", 0, -1].count(self.strand) > 0:
            self.seglen = self.seglen[::-1]

    def add_introns(self):
        if self.exons.any():
            left = self.exons[:-1, 1] + 1
            right = self.exons[1:, 0] - 1
            self.introns = np.stack([left, right], 1)


class Gene:
    def __init__(self, chrom, strand, start, stop, gene_id, gene_name="*",
                 biotype="*"):
        """
        """
        self.chrom = chrom
        self.strand = strand
        self.start = int(start)
        self.stop = int(stop)
        self.geneID = gene_id
        self.trans = []
        self.tranNum = 0
        self.biotype = biotype
        self.geneName = gene_name
        self.PSI = np.nan

    def add_transcipt(self, transcript):
        self.trans.append(transcript)
        self.tranNum += 1

    def get_gene_info(self):
        RV = [self.geneID, self.geneName, self.chrom, self.strand, self.start,
              self.stop, self.biotype]
        _trans = []
        for t in self.trans:
            _trans.append(t.tranID)
        RV.append(",".join(_trans))
        return RV

    def add_premRNA(self):
        _tran = Transcript(self.chrom, self.strand, self.start, self.stop,
                           self.geneID + ".p", self.geneName, self.biotype)
        _tran.add_exon(self.chrom, self.strand, self.start, self.stop)
        self.trans.append(_tran)
        self.tranNum += 1

    def get_exon_max_num(self):
        exonMax = 0
        for _tran in self.trans:
            exonMax = max(exonMax, _tran.exonNum)
        return exonMax

    def gene_ends_update(self):
        for t in self.trans:
            self.start = min(self.start, np.min(t.exons))
            self.stop = max(self.stop, np.max(t.exons))

    def get_all_exons(self):
        exons = np.vstack([i.exons for i in self.trans])
        exons = np.unique(exons, axis=0)
        # exons = np.sort(exons, axis=0) # TODO: use lexsort by 0 and 1 column
        ind = np.lexsort((exons[:, 1], exons[:, 0]))
        exons = exons[ind]
        return exons

    def get_all_introns(self):
        for j in range(len(self.trans)):
            self.trans[j].add_introns()
        introns = np.vstack([i.introns for i in self.trans])
        introns = np.unique(introns, axis=0)
        # introns = np.sort(introns, axis=0) # TODO: use lexsort by 0 and 1 column
        ind = np.lexsort((introns[:, 1], introns[:, 0]))
        introns = introns[ind]
        return introns

    @property
    def PSI(self):
        return self._PSI

    @PSI.setter
    def PSI(self, value):
        self._PSI = value


def parse_attribute(attStr, default="*",
                    ID_tags="ID,gene_id,transcript_id,mRNA_id",
                    Name_tags="Name,gene_name,transcript_name,mRNA_name",
                    Type_tags="Type,gene_type,gene_biotype,biotype",
                    Parent_tags="Parent"):
    """
    Parse attributes in GTF or GFF3
    Parameters
    ----------
    attStr: string
        String containing attributes either in GTF or GFF3 format.
    default: string
        default value for ID, Name, Type and Parent.
    ID_tags: string
        Multiple tags for ID. Use comma for delimit. 
        If multiple tags found, use the last one.
    Name_tags: string
        Multiple tags for Name. Use comma for delimit. 
        If multiple tags found, use the last one.
    Type_tags: string
        Multiple tags for Type. Use comma for delimit. 
        If multiple tags found, use the last one.
    Parent_tags: string
        Multiple tags for Parent. Use comma for delimit. 
        If multiple tags found, use the last one.
    Returns
    -------
    RV: library of string
        Library of all tags, always including ID, Name, Type, Parenet.
    """
    RV = {}
    RV["ID"] = default
    RV["Name"] = default
    RV["Type"] = default
    RV["Parent"] = default
    ID_tags = ID_tags.split(",")
    Name_tags = Name_tags.split(",")
    Type_tags = Type_tags.split(",")
    Parent_tags = Parent_tags.split(",")

    attList = attStr.rstrip().split(";")
    for att in attList:
        while len(att) > 0 and att[0] == " ":
            att = att[1:]
        if len(att) == 0:
            continue
        if att.find("=") > -1:
            _att = att.split("=")  # GFF3
        else:
            _att = att.split(" ")  # GTF

        if len(_att) < 2:
            print("Can't pase this attribute: %s" % att)
            continue

        if _att[1][0] == '"':
            _att[1] = _att[1].split('"')[1]

        if ID_tags.count(_att[0]) == 1:
            RV["ID"] = _att[1]
        elif Name_tags.count(_att[0]) == 1:
            RV["Name"] = _att[1]
        elif Type_tags.count(_att[0]) == 1:
            RV["Type"] = _att[1]
        elif Parent_tags.count(_att[0]) == 1:
            RV["Parent"] = _att[1]
        else:
            RV[_att[0]] = _att[1]

    return RV

def loadgene(anno_file, comments="#,>", geneTag="gene",
             tranTag="transcript,mRNA", exonTag="exon"):
    """
    Load genes from gtf or gff3 file.
    Parameters
    ----------
    anno_file: str
        path for the annotation file in GTF or GFF3 format.
    comments: string
        Multiple comments. Use comma for delimit. 
    geneTag: string
        Multiple tags for gene. Use comma for delimit. 
    tranTag: string
        Multiple tags for transcript. Use comma for delimit. 
    exonTag: string
        Multiple tags for exon. Use comma for delimit. 
    Return
    ------
    genes: list of ``pyseqlib.Gene``
        a list of loaded genes
    """

    fid = open(anno_file, "r")
    anno_in = fid.readlines()
    fid.close()

    geneTag = geneTag.split(",")
    tranTag = tranTag.split(",")
    exonTag = exonTag.split(",")
    comments = comments.split(",")

    genes = []
    _gene = None
    for _line in anno_in:
        if comments.count(_line[0]):
            continue

        aLine = _line.split("\t")
        if len(aLine) < 8:
            continue
        elif geneTag.count(aLine[2]) == 1:
            if _gene is not None:
                genes.append(_gene)

            RVatt = parse_attribute(aLine[8], ID_tags="ID,gene_id",
                                    Name_tags="Name,gene_name")
            _gene = Gene(aLine[0], aLine[6], aLine[3], aLine[4],
                         RVatt["ID"], RVatt["Name"], RVatt["Type"])

        elif tranTag.count(aLine[2]) == 1:
            RVatt = parse_attribute(aLine[8], ID_tags="ID,transcript_id,mRNA_id",
                                    Name_tags="Name,transcript_name,mRNA_name")
            _tran = Transcript(aLine[0], aLine[6], aLine[3], aLine[4],
                               RVatt["ID"], RVatt["Name"], RVatt["Type"])

            if _gene is not None:
                _gene.add_transcipt(_tran)
            else:
                print("Gene is not ready before transcript.")

        elif exonTag.count(aLine[2]) == 1:
            if aLine[0] != _gene.trans[-1].chrom:
                print("Exon from a different chrom of transcript.")
                continue
            if aLine[6] != _gene.trans[-1].strand:
                print("Exon from a different strand of transcript.")
                continue
            if _gene is not None and len(_gene.trans) > 0:
                _gene.trans[-1].add_exon(aLine[0], aLine[6], aLine[3], aLine[4])
                # _gene.gene_ends_update()
            else:
                print("Gene or transcript is not ready before exon.")

    if _gene is not None:
        genes.append(_gene)

    return genes


class FastaFile:
    """docstring for FastaFile"""

    def __init__(self, fasta_file):
        self.f = pysam.FastaFile(fasta_file)

    def get_seq(self, chrom, start_stop, strand):
        """get the sequence in a given region, the start is from 1.
        The start and stop index may still need double check."""
        start, stop = start_stop
        seq = self.f.fetch(chrom, start - 1, stop)
        if strand == "-":
            seq = self.rev_seq(seq)
        return seq

    def rev_seq(self, seq):
        """ reverse complement """
        _tmp = []
        _tmp[:] = seq
        for j in range(len(_tmp)):
            if _tmp[j] == "A":
                _tmp[j] = "T"
            elif _tmp[j] == "T":
                _tmp[j] = "A"
            elif _tmp[j] == "G":
                _tmp[j] = "C"
            elif _tmp[j] == "C":
                _tmp[j] = "G"
        RV = "".join(_tmp[::-1])
        return RV
