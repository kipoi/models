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
from kipoi.specs import RemoteFile
from kipoi.external.torchvision.dataset_utils import download_url
from kipoi.utils import makedir_exist_ok
import linecache
from six.moves.urllib.request import urlretrieve

# Get the local path
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
this_dir = os.path.dirname(os.path.abspath(filename))
# --------------------------------------------

# TODO - include this files also on Zenodo
def download_gencode_dir(output_dir):
    """Download all the required gencode files
    """
    makedir_exist_ok(output_dir)
    url_template = ("https://s3.eu-central-1.amazonaws.com/kipoi-models/"
                    "dataloader_files/FactorNet/dataloader_files/gencode_features/{}")

    # url_template = "https://github.com/uci-cbcl/FactorNet/blob/master/resources/{}?raw=true"
    fnames = [('cpgisland.bed.gz', 'ac7dc007d7019c05adb7a331d1d6721d'),
              ('wgEncodeGencodeBasicV19.cds.merged.bed.gz', '4ec9883932932efe87e4adc6c84ced1c'),
              ('wgEncodeGencodeBasicV19.intron.merged.bed.gz', 'd2db7e3255323d2b5b04e1c0c59ecd2d'),
              ('wgEncodeGencodeBasicV19.promoter.merged.bed.gz', '48fe1ab3aa0e9f5d11f3e5dfedbd47b6'),
              ('wgEncodeGencodeBasicV19.utr5.merged.bed.gz', 'de87c14d4ff055226afeb01446aba6e6'),
              ('wgEncodeGencodeBasicV19.utr3.merged.bed.gz', '8bbe08f5fba86306dfbef56d756856f1')]
    for fname, md5 in fnames:
        output_file = os.path.join(output_dir, fname)
        rf = RemoteFile(url=url_template.format(fname), md5=md5)
        if not os.path.exists(output_file) or not rf.validate(output_file):
            rf.get_file(output_file)


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
                 use_linecache=True):

        # intervals
        if use_linecache:
            linecache.clearcache()
            BT = BedToolLinecache
        else:
            BT = BedTool

        self.bt = BT(intervals_file)

        # Fasta
        self.fasta_file = fasta_file
        self.fasta_extractor = None  # initialize later
        # DNase
        self.dnase_file = dnase_file
        self.dnase_extractor = None

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            # Fasta
            self.fasta_extractor = FastaExtractor(self.fasta_file)
            # DNase
            self.dnase_extractor = BigwigExtractor(self.dnase_file)

        # Get the interval
        interval = self.bt[idx]
        if interval.stop - interval.start != self.SEQ_WIDTH:
            center = (interval.start + interval.stop) // 2
            interval.start = center - self.SEQ_WIDTH // 2
            interval.end = center + self.SEQ_WIDTH // 2 + self.SEQ_WIDTH % 2

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]), axis=0)
        seq_rc = seq[::-1, ::-1]

        # Dnase
        dnase = np.squeeze(self.dnase_extractor([interval], axis=0))[:, np.newaxis]
        dnase[np.isnan(dnase)] = 0  # NA fill
        dnase_rc = dnase[::-1]

        bigwig_list = [seq]
        bigwig_rc_list = [seq_rc]
        bigwig_list.append(dnase)
        bigwig_rc_list.append(dnase_rc)

        ranges = GenomicRanges.from_interval(interval)
        ranges_rc = GenomicRanges.from_interval(interval)
        ranges_rc.strand = "-"

        return {
            "inputs": [
                np.concatenate(bigwig_list, axis=-1),  # stack along the last axis
                np.concatenate(bigwig_rc_list, axis=-1),  # RC version
            ],
            "targets": {},  # No Targets
            "metadata": {
                "ranges": ranges,
                "ranges_rc": ranges_rc
            }
        }
