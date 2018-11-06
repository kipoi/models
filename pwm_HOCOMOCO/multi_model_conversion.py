# pwm_HOCOMOCO move:
import os
import yaml
import pandas as pd

original_group_name = "pwm_HOCOMOCO"
original_group_new = "pwm_HOCOMOCO_redesign"
model_info = []
listen_to = ["args.weights"]

def recursive_extractor(yaml_dict, listen_to):
    cats = listen_to.split(".")
    k = cats[0]
    if k not in yaml_dict:
        return None
    if len(cats) == 1:
        return {k:yaml_dict[k]}
    else:
        ret_val = recursive_extractor(yaml_dict[k], ".".join(cats[1:]))
        return {".".join([k, k2]):v for k2, v in ret_val.items()}

for root, dirs, files in os.walk(original_group_name):
    if "model.yaml" in files and "template" not in root:
        root_no_base = "/".join(root.split("/")[len(original_group_name.rstrip("/").split("/")):])
        this_model = {}
        this_model['model_name'] = root_no_base
        # derive the model file from the yaml
        with open(os.path.join(root, "model.yaml"), "r") as yfh:
            model_yaml = yaml.load(yfh)
        for k in listen_to:
            ret = recursive_extractor(model_yaml, k)
            if ret is not None:
                this_model.update({k:os.path.join(root_no_base, v) for k, v in ret.items()})
        model_info.append(this_model)

def rename_fn(old_fn):
    return "model_files/" + old_fn.replace("/model_files/model", "").replace("/", "-")

model_info_pd = pd.DataFrame(model_info)
model_info_pd['args.weights_new'] = model_info_pd['args.weights'].apply(rename_fn)


for _, row in model_info_pd.iterrows():
    out_path = os.path.join(original_group_new, row['args.weights_new'])
    bpath = "/".join(out_path.split("/")[:-1])
    if not os.path.exists(bpath):
        os.makedirs(bpath)
    ret = os.system("cp {0} {1}".format(os.path.join(original_group_name, row['args.weights']), out_path))
    assert ret == 0 

model_info_pd_out = model_info_pd[['model_name', 'args.weights_new']]
model_info_pd_out.columns = ['model', 'args_weights']
model_info_pd_out.to_csv(os.path.join(original_group_new, "models.tsv"), sep="\t", index=None)



