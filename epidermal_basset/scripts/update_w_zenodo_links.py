
import re

import pandas as pd

def main():
    """update models.tsv with zenodo links
    """
    # zenodo record
    zenodo_url_prefix = "https://zenodo.org/record/4777310/files"
    zenodo_url_suffix = "?download=1"
    
    # read in models
    old_models_file = "models_kipoi_local.tsv"
    models = pd.read_csv(old_models_file, sep="\t")

    models["args_meta_url"] = [
        "{}/{}".format(
            re.sub("models.kipoi_compatible", zenodo_url_prefix, url), zenodo_url_suffix)
        for url in models["args_meta_url"]]

    models["args_index_url"] = [
        "{}/{}".format(
            re.sub("models.kipoi_compatible", zenodo_url_prefix, url), zenodo_url_suffix)
        for url in models["args_index_url"]]

    models["args_data_url"] = [
        "{}/{}".format(
            re.sub("models.kipoi_compatible", zenodo_url_prefix, url), zenodo_url_suffix)
        for url in models["args_data_url"]]

    # save out
    models.to_csv(
        "models.tsv", sep="\t", header=True, index=False)
    
    return


main()
