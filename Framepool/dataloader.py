import os

import pandas as pd
import numpy as np

from kipoi.metadata import GenomicRanges
from kipoi.data import Dataset

from kipoiseq.extractors import FastaStringExtractor
from kipoiseq.extractors import MultiSampleVCF, VariantSeqExtractor, SingleSeqVCFSeqExtractor
from kipoiseq.dataloaders.sequence import BedDataset
from kipoiseq.transforms.functional import resize_interval
from kipoiseq.utils import to_scalar

import pybedtools
from pybedtools import BedTool, Interval

class StrandedSequenceVariantDataloader(Dataset):
    """ This Dataloader requires the following input files:
        1. bed3+ where a specific user-specified column (>3, 1-based) of the bed denotes the strand
        and a specific user-specified column (>3, 1-based) of the bed denotes the transcript id
        (or some other id that explains which exons in the bed belong together to form one sequence).
        All columns of the bed, except the first three, the id and the strand, are ignored. 
        2. fasta file that provides the reference genome
        3. bgzip compressed (single sample) vcf that provides the variants
        4. A chromosome order file (such as a fai file) that specifies the order of chromosomes 
        (must be valid for all files)
        The bed and vcf must both be sorted (by position) and a tabix index must be present.
        (must lie in the same directory and have the same name + .tbi)
        The num_chr flag indicates whether chromosomes are listed numerically or with a chr prefix.
        This must be consistent across all input files!
        The dataloader finds all intervals in the bed which contain at least one variant in the vcf.
        It then joins intervals belonging to the same transcript, as specified by the id, to a single sequence.
        For these sequences, it extracts the reference sequence from the fasta file, 
        injects the applicable variants and reverse complements according to the strand information.
        This means that if a vcf mixes variants from more than one patient, the results will not be
        meaningful. Split the vcf by patient and run the predictions seperately in this case!
        Returns the reference sequence and variant sequence as 
        np.array([reference_sequence, variant_sequence]). 
        Region metadata is additionally provided
"""
    def __init__(self,
                 intervals_file,
                 fasta_file,
                 vcf_file,
                 chr_order_file,
                 vcf_file_tbi=None,
                 strand_column=6,
                 id_column=4,
                 num_chr=True
                ):

        # workaround for test
        if vcf_file_tbi is not None and vcf_file_tbi.endswith("vcf_file_tbi"):
            os.rename(vcf_file_tbi, vcf_file_tbi.replace("vcf_file_tbi", "vcf_file.tbi"))
        
        self.num_chr_fasta = num_chr
        self.intervals_file = intervals_file
        self.fasta_file = fasta_file
        self.vcf_file = vcf_file
        self.chr_order_file = chr_order_file
       
        self.strand_column = strand_column - 1
        self.id_column = id_column - 1

        self.force_upper = True
       
        # "Parse" bed file
        self.bed = BedDataset(self.intervals_file,
                              num_chr=self.num_chr_fasta,
                              bed_columns=3,
                              label_dtype=str,
                              ignore_targets=False)
        
        # Intersect bed and vcf using bedtools
        # bedtools c flag: for each bed interval, counts number of vcf entries it overlaps
        bed_tool = pybedtools.BedTool(self.intervals_file)
        intersect_counts = list(bed_tool.intersect(self.vcf_file, c=True, sorted=True, 
                                                   g=self.chr_order_file))
        intersect_counts = np.array([isect.count for isect in intersect_counts])
                
        # Retain only those transcripts that intersect a variant
        utr5_bed = self.bed.df
        id_col = utr5_bed.iloc[:,self.id_column]
        retain_transcripts = utr5_bed[intersect_counts > 0].iloc[:,self.id_column]
        utr5_bed = utr5_bed[utr5_bed.iloc[:,self.id_column].isin(retain_transcripts)]
        
        # Aggregate 5utr positions per transcript
        tuples = list(zip(utr5_bed.iloc[:,1], utr5_bed.iloc[:,2]))
        pos = [[x] for x in tuples]
        id_chr_strand = list(zip(utr5_bed.iloc[:,self.id_column], utr5_bed.iloc[:,0],
                          utr5_bed.iloc[:,self.strand_column]))
        utr5_bed_posaggreg = pd.DataFrame({"pos":pos, "id_chr_strand": id_chr_strand})
        utr5_bed_posaggreg = utr5_bed_posaggreg.groupby("id_chr_strand").agg({'pos': 'sum'})
        
        # Rebuild "bed"
        utr5_bed_posaggreg["id"] = [x[0] for x in utr5_bed_posaggreg.index]
        utr5_bed_posaggreg["chr"] = [x[1] for x in utr5_bed_posaggreg.index]
        utr5_bed_posaggreg["strand"] = [x[2] for x in utr5_bed_posaggreg.index]
        self.bed = utr5_bed_posaggreg.reset_index()[["id", "chr", "pos", "strand"]]
        
        self.fasta_extractor = None
        self.vcf = None
        self.vcf_extractor = None
        
    def __len__(self):
        return len(self.bed)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaStringExtractor(self.fasta_file, use_strand=True,
                                                         force_upper=self.force_upper)
        if self.vcf is None:
            self.vcf = MultiSampleVCF(self.vcf_file)
        if self.vcf_extractor is None:
            self.vcf_extractor = VariantSeqExtractor(self.fasta_file)
        
        entry = self.bed.iloc[idx]
        entry_id = entry["id"]
        entry_chr = entry["chr"]
        entry_pos = entry["pos"]
        entry_strand = entry["strand"]
        
        ref_exons = []
        var_exons = []
        exon_pos_strings = []
        exon_var_strings = []
        for exon in entry_pos:
            # We get the interval
            interval = pybedtools.Interval(to_scalar(entry_chr), to_scalar(exon[0]), 
                                           to_scalar(exon[1]), strand=to_scalar(entry_strand))
            exon_pos_strings.append("%s-%s" % (str(exon[0]),str(exon[1])))

            # We get the reference sequence
            ref_seq = self.fasta_extractor.extract(interval)

            # We get the variants, insert them and also save them as metadata
            variants = list(self.vcf.fetch_variants(interval))
            if len(variants) == 0:
                ref_exons.append(ref_seq)
                var_exons.append(ref_seq)
            else:
                var_seq = self.vcf_extractor.extract(interval, variants=variants,
                    anchor=0, fixed_len=False)
                var_string = ";".join([str(var) for var in variants])

                ref_exons.append(ref_seq)
                var_exons.append(var_seq)
                exon_var_strings.append(var_string)
        
        # Combine
        if entry_strand == "-":
            ref_exons.reverse()
            var_exons.reverse()
        ref_seq = "".join(ref_exons)
        var_seq = "".join(var_exons)
        pos_string = ";".join(exon_pos_strings)
        var_string = ";".join(exon_var_strings)
        
        return {
            "inputs": np.array([ref_seq, var_seq]),
            "metadata": {
                "id": entry_id,
                "chr": entry_chr,
                "exon_positions": pos_string,
                "strand": entry_strand,
                "variants": var_string
            }
        }

    

    


"""    def __del__(self):
        for file in self.tmp_files:
            os.remove(file)
            
    def create_tbi_index(self):
        tabix_path = self.vcf_file + ".tbi"
        if not os.path.isfile(tabix_path):
            try:
                subprocess.check_output(['tabix','-p','vcf',self.vcf_file])
                self.tmp_files.append(tabix_path)
            except: 
                raise ValueError("Failed to create tabix index. Possible reasons: \
                                 vcf corrupt; vcf not bgzipped; \
                                 vcf not sorted by position")
    
    def presort_bed(self):
        tmp = tempfile.mkstemp(suffix=".bed")[1]
        pybedtools.BedTool(self.intervals_file).sort(output=tmp)
        self.intervals_file = tmp
        self.tmp_files.add(tmp)
        
    def presort_vcf(self):
        tmp = tempfile.mkstemp(suffix=".vcf.gz")[1]
        pybedtools.BedTool(self.vcf_file).sort(output=tmp)
        self.vcf_file = tmp
        self.tmp_files.add(tmp)
"""
