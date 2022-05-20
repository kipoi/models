import kipoi
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

model_root = "./"

def test_ref_seq():
    # Get pure fasta predictions
    model_dir = model_root + "./"
    model = kipoi.get_model(model_dir, source="dir")
    # The preprocessor
    dataloader_arguments = {
            "fasta_file": "/home/yuanh/programs/genomes/hg19/hg19.fa",
        "intervals_file": "../test_peaks.bed"
    }
    preds = model.pipeline.predict(dataloader_arguments)
    counts = np.load('../test_m.npy')
    res = roc_auc_score(counts, preds)
    assert np.isclose(res, 0.74, 1e-3)
