import gffutils
import pybedtools
from kipoi.data import Dataset
from kipoiseq.transforms import OneHot
from kipoiseq.extractors import FastaStringExtractor
from kipoi.metadata import GenomicRanges


def get_upstream(feature, n_bases):
    """Get the upstream interval of the genomic feature
    Args:
      feature: gffutils.Feature
      n_bases

      [  interval  ][>>>feature>>>>]
       <- n_bases->
    """
    if feature.strand == '-':
        start = feature.end
        end = feature.end + n_bases
    else:
        start = feature.start - n_bases
        end = feature.start
    return pybedtools.create_interval_from_list([feature.chrom, start, end,
                                                 feature.id, feature.score, feature.strand])


class FixedSeq5UtrDl(Dataset):

    n_upstream = 50

    def __init__(self, gtf_file, fasta_file,
                 disable_infer_transcripts=True,
                 disable_infer_genes=True):
        self.gtf_file = gtf_file
        self.fasta_file = fasta_file

        self.fasta_extractor = None

        self.db = gffutils.create_db(gtf_file, ":memory:",
                                     disable_infer_transcripts=disable_infer_transcripts,
                                     disable_infer_genes=disable_infer_genes)
        self.start_codons = list(self.db.features_of_type("start_codon"))
        self.input_transform = OneHot()

    def __len__(self):
        return len(self.start_codons)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaStringExtractor(self.fasta_file,
                                                        use_strand=True,
                                                        force_upper=True)
        feature = self.start_codons[idx]
        interval = get_upstream(feature, self.n_upstream)
        seq = self.fasta_extractor.extract(interval)
        seq_one_hot_encoded = self.input_transform(seq)
        return {
            "inputs": seq_one_hot_encoded,
            "metadata": {
                "ranges": GenomicRanges.from_interval(interval),
                "gene_id": feature.attributes.get('gene_id', [""])[0],
                "transcript_id": feature.attributes.get('transcript_id', [""])[0],
                "gene_biotype": feature.attributes.get('gene_biotype', [""])[0]
            }
        }