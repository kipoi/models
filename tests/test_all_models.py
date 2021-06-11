"""Test all models

1. [x] list all models to test (for model-groups, take the first one)
2. Write functions to test a single models
  - Create a new conda environment
  - activate the environment
  - run `kipoi test model --source=kipoi` in that environment
"""

import pytest
import subprocess
import kipoi
import logging
from kipoi_conda import get_kipoi_bin, env_exists, remove_env
from kipoi.cli.env import conda_env_name
from kipoi_utils.utils import list_files_recursively, read_txt


def models_to_test(src):
    """Returns a list of models to test

    By default, this method returns all the model. In case a model group has a `test_subset.txt`
    file present in the group directory, then testing is only performed for models
    listed in `test_subset.txt`.

    Args:
      src: Model source
    """
    import os
    txt_files = list_files_recursively(src.local_path, "test_subset", "txt")

    exclude = []
    include = []
    for x in txt_files:
        d = os.path.dirname(x)
        exclude += [d]
        include += [os.path.join(d, l) for l in read_txt(os.path.join(src.local_path, x))]

    # try to load every model
    for m in include:
        src.get_model_descr(m)

    models = src.list_models().model
    for excl in exclude:
        models = models[~models.str.startswith(excl)]
    return list(models) + include


@pytest.mark.parametrize("model_name", models_to_test(kipoi.get_source("kipoi")))
def test_model(model_name, caplog):
    """kipoi test ...
    """
    caplog.set_level(logging.INFO)

    source_name = "kipoi"
    assert source_name == "kipoi"

    env_name = conda_env_name(model_name, model_name, source_name)
    env_name = "test-" + env_name  # prepend "test-"

    # if environment already exists, remove it
    if env_exists(env_name):
        print("Removing the environment: {0}".format(env_name))
        remove_env(env_name)

    # create the model test environment
    args = ["kipoi", "env", "create",
            "--source", source_name,
            "--env", env_name,
            model_name]
    returncode = subprocess.call(args=args)
    assert returncode == 0

    if model_name == "basenji":
        batch_size = str(2)
    else:
        batch_size = str(4)

    # run the tests in the environment
    args = [get_kipoi_bin(env_name), "test",
            "--batch_size", batch_size,
            "--source", source_name,
            model_name]
    returncode = subprocess.call(args=args)
    assert returncode == 0

    for record in caplog.records:
        # there shoudn't be any warning
        assert record.levelname not in ['WARN', 'WARNING', 'ERROR', 'CRITICAL']
