"""Basenji dataloader
"""
# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool
from genomelake.extractors import FastaExtractor, BigwigExtractor
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges
import linecache
from six.moves.urllib.request import urlretrieve

# Get the local path
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
this_dir = os.path.dirname(os.path.abspath(filename))
# --------------------------------------------


class BedToolLinecache(BedTool):
    """Faster BedTool accessor by Ziga Avsec
    Normal BedTools loops through the whole file to get the
    line of interest. Hence the access it o(n)
    Note: this might load the whole bedfile into memory
    """

    def __getitem__(self, idx):
        line = linecache.getline(self.fn, idx + 1)
        return pybedtools.create_interval_from_list(line.strip().split("\t"))


class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
    """

    SEQ_WIDTH = 1002

    def __init__(self,
                 intervals_file,
                 fasta_file,
                 dnase_file,
                 {%- if needs_cell_line %}
                 cell_line=None,
                 {%- endif %}
                 {%- if needs_rnaseq %}
                 RNAseq_PC_file=None,
                 {%- endif %}
                 {%- if needs_mappability %}
                 mappability_file=None,
                 {%- endif %}
                 {%- if needs_gencode %}
                 GENCODE_dir=None,
                 {%- endif %}
                 use_linecache=True):

        # intervals
        if use_linecache:
            linecache.clearcache()
            BT = BedToolLinecache
        else:
            BT = BedTool

        self.bt = BT(intervals_file)

        # Fasta
        self.fasta_extractor = FastaExtractor(fasta_file)

        # DNase
        self.dnase_extractor = BigwigExtractor(dnase_file)

        {%- if needs_mappability %}
        # mappability
        if mappability_file is None:
        # download the mappability file if not existing
            mappability_file = os.path.join(this_dir, "../../template/dataloader_files",
          "wgEncodeDukeMapabilityUniqueness35bp.bigWig")
            if not os.path.exists(mappability_file):
                print("Downloading the mappability file")
                urlretrieve("http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDukeMapabilityUniqueness35bp.bigWig", mappability_file)
                print("Download complete")

            {# mappability_file = os.path.join(this_dir, #}
            {#                                 "dataloader_files", #}
            {#                                 "wgEncodeDukeMapabilityUniqueness35bp.bigWig") #}
        self.mappability_extractor = BigwigExtractor(mappability_file)
        {%- endif %}

        {%- if needs_gencode %}
        # Gencode features
        if GENCODE_dir is None:
            gp = os.path.join(this_dir, "dataloader_files/gencode_features/")
        else:
            gp = GENCODE_dir
        self.gencode_beds = [
            ("cpg", BedTool(gp + '/cpgisland.bed.gz')),
            ("cds", BedTool(gp + '/wgEncodeGencodeBasicV19.cds.merged.bed.gz')),
            ("intron", BedTool(gp + '/wgEncodeGencodeBasicV19.intron.merged.bed.gz')),
            ("promoter", BedTool(gp + '/wgEncodeGencodeBasicV19.promoter.merged.bed.gz')),
            ("utr5", BedTool(gp + '/wgEncodeGencodeBasicV19.utr5.merged.bed.gz')),
            ("utr3", BedTool(gp + '/wgEncodeGencodeBasicV19.utr3.merged.bed.gz')),
        ]
        # Overlap beds - could be done incrementally
        print("Overlapping all the bed-files")
        # The BT() and .fn are there in order to leverage BedToolLinecache
        overlap_beds = [(b, BT(self.bt.intersect(v, wa=True, c=True).fn))
                        for b, v in self.gencode_beds]
        print("Assesing the file")
        assert len(overlap_beds[1][1]) == len(self.bt)
        {%- endif %}

        {%- if needs_rnaseq %}
        # Get the metadata features
        if cell_line is None:
            if RNAseq_PC_file is None:
                raise ValueError("RNAseq_PC_file has to be specified when cell_line=None")
            assert os.path.exists(RNAseq_PC_file)
        else:
            # Using the pre-defined cell-line
            rp = os.path.join(this_dir, "dataloader_files/RNAseq_features/")
            RNAseq_PC_file = os.path.join(rp, cell_line, "meta.txt")
        self.meta_feat = pd.read_csv(RNAseq_PC_file,
                                     sep="\t", header=None)[0].values
        {%- endif %}

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        # Get the interval
        interval = self.bt[idx]
        if interval.stop - interval.start != self.SEQ_WIDTH:
            raise ValueError("Expected the interval to be {0} wide. Recieved stop - start = {1}".
                             format(self.SEQ_WIDTH, interval.stop - interval.start))

        {%- if needs_gencode %}
        # Get the gencode features
        gencode_counts = np.array([v[idx].count for k, v in self.gencode_beds],
                                  dtype=bool)
        {%- endif %}

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]), axis=0)
        seq_rc = seq[::-1, ::-1]

        # Dnase
        dnase = np.squeeze(self.dnase_extractor([interval], axis=0))[:, np.newaxis]
        dnase[np.isnan(dnase)] = 0  # NA fill
        dnase_rc = dnase[::-1]

        bigwig_list = [seq, dnase]
        bigwig_rc_list = [seq_rc, dnase_rc]

        {%- if needs_mappability %}
        mappability = np.squeeze(self.mappability_extractor([interval], axis=0))[:, np.newaxis]
        mappability[np.isnan(mappability)] = 0  # NA fill
        mappability_rc = mappability[::-1]
        bigwig_list.append(mappability)
        bigwig_rc_list.append(mappability_rc)
        {%- endif %}

        ranges = GenomicRanges.from_interval(interval)
        ranges_rc = GenomicRanges.from_interval(interval)
        ranges_rc.strand = "-"

        return {
            "inputs": [
                np.concatenate(bigwig_list, axis=-1),  # stack along the last axis
                np.concatenate(bigwig_rc_list, axis=-1),  # RC version
                {%- if needs_rnaseq and needs_gencode %}
                np.append(self.meta_feat, gencode_counts)
                {%- elif needs_rnaseq and not needs_gencode %}
                self.meta_feat
                {%- elif not needs_rnaseq and needs_gencode %}
                gencode_counts
                {%- endif %}
            ],
            "targets": {},  # No Targets
            "metadata": {
                "ranges": ranges,
                "ranges_rc": ranges_rc
            }
        }
