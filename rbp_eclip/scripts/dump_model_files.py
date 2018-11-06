"""Dump model files
"""
import numpy as np
import pandas as pd
from shutil import copyfile
from glob import glob
from concise.hyopt import CMongoTrials, get_data
import os
from pprint import pprint
from kipoi.utils import load_module

# src
from dump_dataloader_files import RBP_ALL, DIR_ROOT

DB_NAME = "RBP__Eclip_p"
HOST = "ouga03"


def get_models_overall(exp_name, rbp):
    print("RBP: " + rbp)

    out_h5 = "{rbp}/model_files/model.h5".format(rbp=rbp)
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)

    trials = CMongoTrials(DB_NAME, exp_name + "_" + rbp, ip=HOST)

    # no trials yet - return None
    if trials.n_ok() == 0:
        trials = CMongoTrials(DB_NAME[:-2], exp_name + "_" + rbp, ip=HOST)
        if trials.n_ok() == 0:
            raise Exception("No trials")
    print("N trials: {0}".format(trials.n_ok()))

    # get best trial parameters
    tid = trials.best_trial_tid()
    model_path = trials.get_trial(tid)["result"]["path"]["model"]
    copyfile(model_path, out_h5)


if __name__ == "__main__":
    EXPERIMENT = "DeepNN_scalar_position_ext_gam"

    for rbp in RBP_ALL:
        get_models_overall(EXPERIMENT, rbp)
