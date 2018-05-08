"""Helper functions to gather the variants
"""

import os
from tqdm import tqdm
from collections import OrderedDict
from glob import glob
import pandas as pd
import numpy as np
from kipoi.postprocessing.variant_effects import KipoiVCFParser

import pandas as pd


def refmt_col(col, model_name, col_types):
    for ct in col_types:
        if ct in col:
            col = model_name + ct.lower()
            break
    return col


def average_labranchor(df, model_name, col_types):
    import numpy as np
    # choose the maximum diff
    diff_cols = df.columns.values[df.columns.str.contains("DIFF")]
    model_outputs = [int(el.split("_")[-1]) for el in diff_cols]
    model_outputs_order = np.argsort(model_outputs)
    # select the model output tha gives the maximum absolute difference
    max_col_id = df[diff_cols[model_outputs_order]].abs().values.argmax(axis=1)
    #
    # just to be sure it will work:
    assert np.all(df[diff_cols[model_outputs_order]].abs().values[np.arange(len(max_col_id)), max_col_id] == df[diff_cols].abs().max(axis=1).values)
    #
    averaged = {}
    usable_columns = df.columns.tolist()
    for ct in col_types:
        col_sel = [col for col in usable_columns if ct in col]
        usable_columns = [col for col in usable_columns if col not in col_sel]
        if len(col_sel) == 0:
            break
        # average
        model_outputs = [int(el.split("_")[-1]) for el in col_sel]
        model_outputs_order = np.argsort(model_outputs)
        # use the column selection from before
        keep_vals = df[np.array(col_sel)[model_outputs_order]].values[np.arange(len(max_col_id)), max_col_id]
        averaged[model_name + ct.lower()] = keep_vals
    #
    return pd.DataFrame(averaged, index=df.index)


def deduplicate_vars(df):
    diff_cols = df.columns.values[df.columns.str.contains("diff")]
    assert len(diff_cols) == 1
    return df.groupby(df.index).apply(lambda x: x.iloc[np.argmax(x[diff_cols[0]].values), :])


# Modify here: add the _isna column and average labranchor if needed also clump the variants together.
def get_df(vcf_file, model_name):
    df = pd.DataFrame(list(KipoiVCFParser(vcf_file)))
    df.index = df["variant_id"]
    obsolete_variant_columns = ["variant_chr", "variant_pos", "variant_ref", "variant_alt", "variant_id"]
    df = df[[col for col in df.columns if col not in obsolete_variant_columns]]
    df = df[[col for col in df.columns if "rID" not in col]]
    col_types = ["_LOGIT_REF", "_LOGIT_ALT", "_REF", "_ALT", "_DIFF", "_LOGIT"]
    if model_name == "labranchor":
        df = average_labranchor(df, model_name, col_types)
    else:
        df.columns = [refmt_col(col, model_name, col_types) for col in df.columns]
    # clump variants together
    df = deduplicate_vars(df)
    return df


def gather_vcfs(models, base_path, ncores=16, model_df_colnames = None):
    """
    Args:
        models: list of model names
        base_path: base path of the directory for storing vcfs: {base_path}/{model}.vcf
        ncores: number of cores used to read the data in paralell
        model_df_colnames: Model column names present after loading annotated VCF.
            Required if for a model all variants were NA and no VCF had been generated.
    """
    vcf_fnames = [(m, os.path.join(base_path, "{}.vcf".format(m))) for m in models]

    # Check which VCFs are missing because no overlaps between GTF and VCF existed.
    all_na_models = []
    if model_df_colnames is None:
        for m, fname in vcf_fnames:
            if not os.path.exists(fname):
                raise Exception("If model_df_colnames is not set then all VCFs have to exist.")
    else:
        vcf_fnames_exists = []
        for m, fname in vcf_fnames:
            if os.path.exists(fname):
                vcf_fnames_exists.append((m, fname))
            else:
                all_na_models.append(m)
        vcf_fnames = vcf_fnames_exists


    dfs = {}
    from joblib import Parallel, delayed
    import threading
    threading.current_thread().name = 'MainThread'

    dfs = Parallel(n_jobs=ncores)(delayed(get_df)(vcf_file, model_name) for model_name, vcf_file in vcf_fnames)

    for model in all_na_models:
        dfs.append(pd.DataFrame(columns = model_df_colnames[model]))

    merged_dfs = pd.concat(dfs, axis=1)
    for m in models:
        merged_dfs[m + "_isna"] = merged_dfs[m + "_diff"].isnull()

    # now remove variants for which there are no splicing model predictions:
    merged_dfs_filtered = merged_dfs.loc[merged_dfs[[m + "_isna" for m in models]].sum(axis=1) != len(models), :]
    return merged_dfs_filtered.reset_index().rename(columns={"index": "variant_id"})
