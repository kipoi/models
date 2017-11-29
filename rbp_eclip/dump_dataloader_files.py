"""Dump the RBP data
"""
import os
import numpy as np
import dill as pickle
from kipoi.utils import load_module
from concise.preprocessing.splines import encodeSplines
from concise.utils.position import ALL_LANDMARKS
from glob import glob
rbp = "UPF1"
pipeline_obj_path = "dataloader_files/position_transformer.pkl"
# -----------------

# load the data module from eclip
data = load_module(os.path.expanduser("~/projects-work/code_spline_trans/Scripts/RBP/Eclip/predictive_models/data.py"))
DIR_ROOT = data.DIR_ROOT
PROC_DIR = DIR_ROOT + "/processed"
RBP_ALL = [os.path.basename(x).replace(".csv", "")
           for x in glob(PROC_DIR + "/design_matrix/train/*.csv") if "extended" not in x]


def dump_position_transformer(rbp):
    out_path = "{0}/dataloader_files/position_transformer.pkl".format(rbp)
    print("Dumping for rbp: {rbp} to path: {out_path}".format(rbp=rbp, out_path=out_path))
    train, valid, test = data.data_extended(rbp)
    pipeline_obj = train[3:]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # save to pkl
    with open(out_path, "wb") as f:
        pickle.dump(pipeline_obj, f)

    print("Done!")


class PositionTransformer:

    def __init__(self, pos_features, pipeline_obj_path):
        """
        Args:
          pos_features: list of positional features to use
          pipeline_obj_path: path to the serialized pipeline obj_path
        """
        self.pos_features = pos_features
        self.pipeline_obj_path = pipeline_obj_path

        # deserialize the pickle file
        with open(self.pipeline_obj_path, "rb") as f:
            pipeline_obj = pickle.load(f)
        self.POS_FEATURES = pipeline_obj[0]
        self.preproc_pipeline = pipeline_obj[1]
        self.imp = pipeline_obj[2]

        # for simplicity, assume all current pos_features are the
        # same as from before
        assert self.POS_FEATURES == self.pos_features

    def transform(self, x):
        # impute missing values and rescale the distances
        xnew = self.preproc_pipeline.transform(self.imp.transform(x))

        # convert distances to spline bases
        dist = {"dist_" + k: encodeSplines(xnew[:, i, np.newaxis], start=0, end=1)
                for i, k in enumerate(self.POS_FEATURES)}
        return dist


def test_transformer(rbp):
    """Test the transformer function
    """
    posTransf = PositionTransformer(ALL_LANDMARKS, "{0}/dataloader_files/position_transformer.pkl".format(rbp))
    x = np.arange(len(ALL_LANDMARKS)).reshape((1, -1)).astype(float)
    x[0, 1] = np.nan
    res = posTransf.transform(x)
    assert set(res.keys()) == set(["dist_" + x for x in ALL_LANDMARKS])
    print("Test passed!")


if __name__ == "__main__":
    for rbp in RBP_ALL:
        dump_position_transformer(rbp)
        test_transformer(rbp)
