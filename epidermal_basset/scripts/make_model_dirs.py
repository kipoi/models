
import os

import pandas as pd


def main():
    """for whatever reason, testing on models.tsv doesn't work
    so just generating model dirs for each model
    """
    # set up
    out_dir = "models"
    os.system("mkdir -p {}".format(out_dir))
    
    # read in models
    models_file = "models.tsv"
    models = pd.read_csv(models_file, sep="\t")
    
    for model_idx in range(models.shape[0]):
        model_info = models.iloc[model_idx]

        # set up model dir
        model_dir = "{}/{}".format(out_dir, model_info["model"])
        os.system("mkdir -p {}".format(model_dir))
        os.system("cp model.yaml {}/model.TMP.yaml".format(model_dir))

        # adjust model yaml
        old_yaml = "{}/model.TMP.yaml".format(model_dir)
        new_yaml = "{}/model.yaml".format(model_dir)
        with open(new_yaml, "w") as out:
            with open(old_yaml, "r") as fp:
                for line in fp:
                    if "{{ args_meta_url }}" in line:
                        line = "      url: {}\n".format(model_info["args_meta_url"])
                    if "{{ args_meta_md5 }}" in line:
                        line = "      md5: {}\n".format(model_info["args_meta_md5"]) 
                    if "{{ args_index_url }}" in line:
                        line = "      url: {}\n".format(model_info["args_index_url"])
                    if "{{ args_index_md5 }}" in line:
                        line = "      md5: {}\n".format(model_info["args_index_md5"])
                    if "{{ args_data_url }}" in line:
                        line = "      url: {}\n".format(model_info["args_data_url"])
                    if "{{ args_data_md5 }}" in line:
                        line = "      md5: {}\n".format(model_info["args_data_md5"])

                    out.write(line)

        # clean up
        os.system("rm {}".format(old_yaml))

    return


main()
