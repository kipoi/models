
import os
import glob


def main():
    """test all models in their folders
    """
    model_yamls = sorted(glob.glob("models/*/model.yaml"))
    for model_yaml in model_yamls:
        model_dir = os.path.dirname(model_yaml)
        
        # test
        kipoi_cmd = "kipoi test {}/".format(model_dir)
        print(kipoi_cmd)
        os.system(kipoi_cmd)
        
        # clean up
        clean_cmd = "rm -r {}/downloaded/".format(model_dir)
        print(clean_cmd)
        os.system(clean_cmd)

    return


main()
