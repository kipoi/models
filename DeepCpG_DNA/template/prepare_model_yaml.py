import yaml
import keras
import json
import shutil
import os
from deepcpg.utils import make_dir, to_list
from deepcpg.models.utils import decode_replicate_names, encode_replicate_names, get_sample_weights 

##### This function is needed to extract info on model architecture so that the output can be generated correctly.
def data_reader_config_from_model(model, config_out_fpath = None, replicate_names=None):
    """Return :class:`DataReader` from `model`.
    Builds a :class:`DataReader` for reading data for `model`.
    Parameters
    ----------
    model: :class:`Model`.
        :class:`Model`.
    outputs: bool
        If `True`, return output labels.
    replicate_names: list
        Name of input cells of `model`.
    Returns
    -------
    :class:`DataReader`
        Instance of :class:`DataReader`.
    """
    use_dna = False
    dna_wlen = None
    cpg_wlen = None
    output_names = None
    encode_replicates = False
    #
    input_shapes = to_list(model.input_shape)
    for input_name, input_shape in zip(model.input_names, input_shapes):
        if input_name == 'dna':
            # Read DNA sequences.
            use_dna = True
            dna_wlen = input_shape[1]
        elif input_name.startswith('cpg/state/'):
            # DEPRECATED: legacy model. Decode replicate names from input name.
            replicate_names = decode_replicate_names(input_name.replace('cpg/state/', ''))
            assert len(replicate_names) == input_shape[1]
            cpg_wlen = input_shape[2]
            encode_replicates = True
        elif input_name == 'cpg/state':
            # Read neighboring CpG sites.
            if not replicate_names:
                raise ValueError('Replicate names required!')
            if len(replicate_names) != input_shape[1]:
                tmp = '{r} replicates found but CpG model was trained with' \
                    ' {s} replicates. Use `--nb_replicate {s}` or ' \
                    ' `--replicate_names` option to select {s} replicates!'
                tmp = tmp.format(r=len(replicate_names), s=input_shape[1])
                raise ValueError(tmp)
            cpg_wlen = input_shape[2]
    output_names = model.output_names
    config = {"output_names":output_names,
                      "use_dna":use_dna,
                      "dna_wlen":dna_wlen,
                      "cpg_wlen":cpg_wlen,
                      "replicate_names":replicate_names,
                      "encode_replicates":encode_replicates}
    if config_out_fpath is not None:
        with open(config_out_fpath, "w") as ofh:
            json.dump(config, ofh)
    return config


def make_model_yaml(template_yaml, model_json, output_yaml_path):
    #
    with open(template_yaml, 'r') as f:
        model_yaml = yaml.load(f)
    #
    # get the model config:
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    #
    model_yaml["schema"]["targets"] = []
    for oname, oshape in zip(loaded_model.output_names, loaded_model.output_shape):
        append_el ={"name":oname , "shape":str(oshape)#replace("None,", "")
        , "doc":"Methylation probability for %s"%oname}
        model_yaml["schema"]["targets"].append(append_el)
    #
    with open(output_yaml_path, 'w') as f:
        yaml.dump(model_yaml, f, default_flow_style=False)

def make_secondary_dl_yaml(template_yaml, model_json, output_yaml_path):
    with open(template_yaml, 'r') as f:
        model_yaml = yaml.load(f)
    #
    # get the model config:
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    #
    model_yaml["output_schema"]["targets"] = []
    for oname, oshape in zip(loaded_model.output_names, loaded_model.output_shape):
        append_el ={"name":oname , "shape":str(oshape)#replace("None,", "")
        , "doc":"Methylation probability for %s"%oname}
        model_yaml["output_schema"]["targets"].append(append_el)
    #
    with open(output_yaml_path, 'w') as f:
        yaml.dump(model_yaml, f, default_flow_style=False)


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

def softlink_files(bpath, model_name):
    print("Softlinking: {0}".format(model_name))
    symlink_force(bpath+"template/dataloader.yaml",bpath+"{0}/dataloader.yaml".format(model_name))
    #symlink_force("../template/model.yaml","{0}/model.yaml".format(model_name))
    symlink_force(bpath+"template/dataloader.py",bpath+"{0}/dataloader.py".format(model_name))
    symlink_force(bpath+"template/example_files",bpath+"{0}/example_files".format(model_name))


# prepare DeepCpG
deepcpg_bdir = "/nfs/research2/stegle/users/rkreuzhu/deepcpg/deepcpg-1.0.4/scripts/"
output_dir = "/nfs/research2/stegle/users/rkreuzhu/kipoi_models_fork/models/DeepCpG"


models = ["Hou2016_HepG2_dna", "Hou2016_HCC_dna", "Hou2016_mESC_dna", "Smallwood2014_serum_dna", "Smallwood2014_2i_dna"]


for model in models:
    in_dir = os.path.join(deepcpg_bdir, model)
    out_dir = os.path.join(output_dir, model)
    model_files = os.path.join(out_dir, "model_files")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(model_files):
        os.makedirs(model_files)
    shutil.copy(os.path.join(in_dir, "model.json"), model_files)
    shutil.copy(os.path.join(in_dir, "model_weights.h5"), model_files)
    make_model_yaml(os.path.join(output_dir, "template", 'model_template.yaml'), os.path.join(model_files, "model.json"), os.path.join(out_dir, 'model.yaml'))
    make_secondary_dl_yaml(os.path.join(output_dir, "template", 'dataloader_m_template.yaml'), os.path.join(model_files, "model.json"), os.path.join(out_dir, 'dataloader_m.yaml'))
    try:
        os.unlink(output_dir+ "/" + model + "/dataloader_m.py")
    except:
        pass
    shutil.copy(output_dir+ "/"+"template/dataloader_m.py",output_dir+ "/"+model)
    softlink_files(output_dir+ "/", model)
    #
    # generate the model config file:
    json_file = open(os.path.join(model_files, "model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    data_reader_config_from_model(loaded_model, os.path.join(out_dir, 'model_config.json'), replicate_names=None)



for model in models:
    out_dir = os.path.join(output_dir, model)
    if os.path.isdir(out_dir):
        command = "python /nfs/research2/stegle/users/rkreuzhu/opt/model-zoo/kipoi/__main__.py  test %s"%out_dir
        ret =os.system(command)
        assert(ret==0)



## test with custom dataloader:

import kipoi
model = kipoi.get_model(out_dir, source="dir") 
Dl = kipoi.get_dataloader_factory(out_dir + "/dataloader_m.yaml", source="dir") # fails



import os
os.chdir(out_dir)

import keras
import kipoi
from dataloader import *

from keras.models import load_model
from dataloader_m import Dataloader


samples = ["example_files/BS27_1_SER.tsv", "example_files/BS27_3_SER.tsv", "example_files/BS27_5_SER.tsv", "example_files/BS27_6_SER.tsv", "example_files/BS27_8_SER.tsv"]
ref = "example_files/mm10"


model = kipoi.get_model("./", source="dir") 
data_loader = Dataloader(samples, ref, outputs = True)

# the inputs, outputs, weights can then be returned from the dataloader...
ret = data_loader.__next__()

for inputs, outputs, weights in data_loader:
    preds = to_list(model.model.predict(inputs))




