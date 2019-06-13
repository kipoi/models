"""Merged model, loads all the other ones
"""
import os
import kipoi
import numpy as np
from kipoi.model import BaseModel
from glob import glob


def dump_models():
    """Dump column names
    """
    model_names = [os.path.basename(os.path.dirname(x))
                   for x in glob("../*/model.yaml")
                   if "template" not in x and "merged" not in x]
    os.makedirs("model_files", exist_ok=True)
    with open("models.txt", "w") as f:
        f.write("\n".join(model_names))


def read_txt(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [l.strip() for l in lines]


class MergedModel(BaseModel):

    def __init__(self):

        from keras import backend as K
        K.clear_session()

        self.model_names = read_txt("models.txt")
        # hard-code the path to this models
        # if we'd use `source='dir'`, then the models wouldn't
        # be updated
        self.models = [kipoi.get_model("CpGenie/{0}".format(m), source='kipoi',
                                       with_dataloader=False)
                       for m in self.model_names]

    def predict_on_batch(self, x):
        return np.stack([model.predict_on_batch(x)[:, 0]
                         # [:, 0] -> take only the positive probability
                         for model in self.models],
                        axis=1)
