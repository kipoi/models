### Check deepsea against
import kipoi
import pandas as pd
import numpy as np
import deepdish.io
import copy
import cyvcf2
import h5py
import os
import pytest
from tqdm import tqdm

model_root = "./"

def array_to_seq(in_array, nuc_dim, alphabet = ["A", "C", "G", "T"]):
    import copy
    in_dim = list(in_array.shape)
    in_dim[nuc_dim] = 1
    out_array = np.ndarray(tuple(in_dim), dtype = "S1")
    slices = [slice(None) for _ in range(len(in_dim))]
    out_slices = copy.copy(slices)
    out_slices[nuc_dim] = 0
    for i, base in enumerate(alphabet):
        in_slices = copy.copy(slices)
        in_slices[nuc_dim] = slice(i, i+1)
        sel = in_array[in_slices] == 1
        out_array[sel] = base
    return out_array



def test_ref_seq():
    model_root = "./"
    model_dir = model_root+"./"
    model = kipoi.get_model(model_dir, source="dir")

    ifh = h5py.File("test_files/encode_roadmap_short.h5", "r")
    ref_in  = ifh["test_in"][:]
    ifh.close()
    ref_in = np.swapaxes(ref_in, 2,3)
    res_orig = pd.read_csv("/nfs/research1/stegle/users/rkreuzhu/deeplearning/Basset/data/encode_roadmap_short_pred.txt", "\t", header=None)

    for i in tqdm(range(int(np.ceil(ref_in.shape[0]/32)))):
        irange = [i*32, min((i+1)*32, ref_in.shape[0])]
        preds = model.predict_on_batch(ref_in[irange[0]:irange[1],...])
        assert np.isclose(preds, res_orig.values[irange[0]:irange[1],...], atol=1e-3).all()



