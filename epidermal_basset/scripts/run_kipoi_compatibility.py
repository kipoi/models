#!/bin/env python

import os
import subprocess

import tensorflow as tf
import pandas as pd


def get_md5sum(filename):
    """get md5sum
    """
    md5sum_val = subprocess.check_output(["md5sum", filename]).decode("ascii")
    md5sum_val = md5sum_val.split()[0]
    
    return md5sum_val


def adjust_meta_file(old_meta_file, new_meta_file):
    """properly insert placeholder into the graph so that Kipoi
    can utilize its own dataloader
    """
    # set up graph with placeholders to feed into model
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32,
                            shape=(None, 1, 1000, 4),
                            name="inputs")

    # load metagraph
    loaded_meta = tf.train.import_meta_graph(
        old_meta_file,
        input_map={
            "map/TensorArrayStack/TensorArrayGatherV3": inputs})
    #ops = tf.get_default_graph().get_operations()
    #print(ops)

    # save out with placeholders
    out = tf.train.export_meta_graph(filename=new_meta_file, as_text=True)

    return


def main():
    """take TF models and make kipoi compatible
    Notes: key thing is to get into the TF meta file and figure out
    which ops are the ones to adjust. Can use import_meta_graph with
    get_operations (see below) to take a look.

    only replaced the input side
    """
    # work dir
    new_model_dir = "models.kipoi_compatible"
    os.system("mkdir -p {}".format(new_model_dir))
    
    # models
    orig_model_table = "models_orig.tsv"
    models = pd.read_csv(orig_model_table, sep="\t")

    # go through each model to update
    for orig_model_idx in range(models.shape[0]):
        model_info = models.iloc[orig_model_idx]
        
        # copy over data
        model_data_name = "{}/{}.model.data-00000-of-00001".format(
            new_model_dir, model_info["names"])
        os.system("cp {} {}".format(model_info["args_data_url"], model_data_name))
        models.loc[orig_model_idx, "args_data_url"] = model_data_name
        
        # copy over index
        model_index_name = "{}/{}.model.index".format(
            new_model_dir, model_info["names"])
        os.system("cp {} {}".format(model_info["args_index_url"], model_index_name))
        models.loc[orig_model_idx, "args_index_url"] = model_index_name
        
        # adjust meta and get new md5 sum
        model_meta_name = "{}/{}.model.meta".format(
            new_model_dir, model_info["names"])
        adjust_meta_file(model_info["args_meta_url"], model_meta_name)
        models.loc[orig_model_idx, "args_meta_url"] = model_meta_name
        md5sum_val = get_md5sum(model_meta_name)
        models.loc[orig_model_idx, "args_meta_md5"] = md5sum_val

    # save out to models.tsv
    models.to_csv(
        "models.tsv", sep="\t", index=False, header=True)

    return


main()
