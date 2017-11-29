"""link files
"""
import os
from dump_dataloader_files import RBP_ALL
import errno


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def softlink_files(rbp):
    print("Softlinking: {0}".format(rbp))
    symlink_force("../template/dataloader_template.yaml",
                  "{0}/dataloader.yaml".format(rbp))
    symlink_force("../template/model_template.yaml",
                  "{0}/model.yaml".format(rbp))
    symlink_force("../template/dataloader.py",
                  "{0}/dataloader.py".format(rbp))
    symlink_force("../template/custom_keras_objects.py",
                  "{0}/custom_keras_objects.py".format(rbp))
    symlink_force("../template/example_files",
                  "{0}/example_files".format(rbp))


if __name__ == "__main__":

    for rbp in RBP_ALL:
        softlink_files(rbp)
