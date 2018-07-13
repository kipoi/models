"""

Steps:
1. Pre-compute the dataloader
2. loop variant by variant
  - get all the scores
3. Return a rich dictionary of predictionsn
targets:
  - predictions
  - exons
inputs:
  - dictionary with all the different scores (nicely organized)

# Improvement ideas:
- store the intermediary arrays using hdf5/zarr
"""
import tempfile
import kipoi
import kipoi.postprocessing.variant_effects.snv_predict as sp
import shutil
from kipoi.postprocessing.variant_effects import score_variants, Diff, Logit
import json
# Allow different python modules to be used here
from sklearn.base import BaseEstimator, TransformerMixin
import inspect
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
import os
import sys
from kipoi.utils import read_txt
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))
sys.path.append(this_path)

# custom scripts
from gather import gather_vcfs


class ZeroImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X).fillna(0).copy()


def preproc(X, features):
    """Pre-processing pipeline
    """
    mapper = DataFrameMapper([
        (features, [ZeroImputer()]),
    ])
    return mapper.fit_transform(X)


def test():
    vcf_file = "example_files/example.vcf"
    gtf_file = "example_files/hg19.chr22.gtf"
    fasta_file = "example_files/hg19.chr22.fa"
    add_conservation = False
    batch_size = 32
    num_workers = 0
    tmpdir = '/tmp/KipoiSplice/'
    this_path = "."


def ensure_dirs(fname):
    """Ensure that the basepath of the given file path exists.

    Args:
      fname: (full) file path
    """
    required_path = "/".join(fname.split("/")[:-1])
    if not os.path.exists(required_path):
        os.makedirs(required_path)


def load_data(vcf_file, gtf_file, fasta_file,
              batch_size=32,
              num_workers=0,
              tmpdir='/tmp/KipoiSplice/'):
    """
    Args:
      vcf_file: Path to the input vcf file
      fasta_file: reference genome fasta file
      gtf_file: path to the GTF file required by the models (Ensemble)
      batch_size: batch size to use with all the models
      num_workers: number of workers to use for each model
      tmpdir (optional): path to the temporary directory where to store the predictions
    """
    #contains_conservation  is not optional here
    contains_conservation=True

    MODELS = ["MaxEntScan/3prime", "MaxEntScan/5prime", "HAL", "labranchor"]
    features = read_txt(os.path.join(this_path, "features.txt"))

    # Could also be generated on the fly from "MODELS"
    with open(os.path.join(this_path, "model_table_cols.json"), "r") as ifh:
        model_output_col_names = json.load(ifh)

    os.makedirs(tmpdir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(dir=tmpdir)

    # Generate a vcf file for each model
    for model in MODELS:
        # One could even parallelize here using joblib for example
        out_vcf_fpath = os.path.join(tmpdir, model + ".vcf")
        ensure_dirs(out_vcf_fpath)
        dataloader_arguments = {"gtf_file": os.path.abspath(gtf_file),
                                "fasta_file": os.path.abspath(fasta_file)}
        if "rbp_eclip" in model:
            dataloader_arguments["use_linecache"] = True
        sel_scores = ["ref", "alt", "diff"]
        if model == "labranchor":
            sel_scores += ["logit_ref", "logit_alt"]
        score_variants(model,
                       dl_args=dataloader_arguments,
                       input_vcf=os.path.abspath(vcf_file),
                       output_vcf=out_vcf_fpath,
                       scores=sel_scores)

    # Gather the predictions from all the vcf files
    conservation_vcf = None
    if contains_conservation:
      conservation_vcf = vcf_file
    df = gather_vcfs(MODELS, tmpdir, max(num_workers, 1),
                     model_output_col_names,
                     conservation_vcf = conservation_vcf)

    # impute zeros, convert the pandas dataframe to the array
    X = preproc(df, features).astype(float)

    try:
        shutil.rmtree(tmpdir)
    except:
        pass

    return {
        "inputs": X,
        "metadata": {
            "variant": {
                "id": df["variant_id"].values,  # have the variant ID
                "chr": df["variant_chr"].values.astype(str),  # get the chromosome
                "pos": df["variant_pos"].values,  # get the position
                "ref": df["variant_ref"].values,  # get the reference allele
                "alt": df["variant_alt"].values,  # get the alternative allele
            }
        }
    }
