# Check deepsea against
import kipoi
import pandas as pd
import numpy as np
import os
import pytest


model_root = "./"


def test_ref_seq():
    # Get pure fasta predictions
    model_dir = model_root + "./"
    model = kipoi.get_model(model_dir, source="dir")
    # The preprocessor
    Dataloader = kipoi.get_dataloader_factory(model_dir, source="dir")
    dataloader_arguments = {
        "fasta_file": "/nfs/research1/stegle/users/rkreuzhu/opt/manuscript_code/data/raw/dataloader_files/shared/hg19.fa",
        "intervals_file": "test_files/test_encode_roadmap.bed"
    }
    # predict using results
    preds = model.pipeline.predict(dataloader_arguments)
    #
    res_orig = pd.read_csv("/nfs/research1/stegle/users/rkreuzhu/deeplearning/Basset/data/test_encode_roadmap_short_pred.txt", "\t", header=None)
    assert np.isclose(preds, res_orig.values, atol=1e-3).all()
