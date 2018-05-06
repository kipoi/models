"""Parse the model directory and write the jinja template
"""

# Script to generate <cell line>/model.yaml from template_model.yaml
from kipoi.utils import read_txt
import os
from jinja2 import Template

GENERATE_FILES = ["model.yaml", "dataloader.py", "dataloader.yaml"]


def render_template(template_path, output_path, context, mkdir=False):
    """Render template with jinja

    Args:
      template_path: path to the jinja template
      output_path: path where to write the rendered template
      context: Dictionary containing context variable
    """
    if mkdir:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(template_path, "r") as f:
        template = Template(f.read())
    out = template.render(**context)
    with open(output_path, "w") as f:
        f.write(out)


def parse_context(orig_dir):
    """Parses the context for each model
    """
    bigwig = read_txt(os.path.join(orig_dir, "bigwig.txt"))
    tasks = read_txt(os.path.join(orig_dir, "chip.txt"))
    features = read_txt(os.path.join(orig_dir, "feature.txt"))
    meta_fname = os.path.join(orig_dir, "meta.txt")
    if os.path.exists(meta_fname):
        meta = read_txt(meta_fname)
        n_meta_features = len(meta)
        assert n_meta_features == 8
    else:
        meta = None
        n_meta_features = 0

    needs_gencode = "gencode" in features
    if needs_gencode:
        n_meta_features += 6
    seq_n_channels = 4 + len(bigwig)

    return {"bigwig": bigwig,
            "tasks": tasks,
            "features": features,
            "meta": meta,
            "needs_mappability": "Unique35" in bigwig,
            "needs_rnaseq": "meta" in features,
            "needs_gencode": needs_gencode,
            "needs_cell_line": ["bigwig"] != features,
            "needs_meta_features": n_meta_features > 0,
            "seq_n_channels": seq_n_channels,
            "n_meta_features": n_meta_features,
            }


def write_templates(tf, model_subtype):
    """For a particular cell_line:
    - Generate `{cell_line}/model.yaml`
    """
    ctx = parse_context(os.path.join("FactorNet/models", tf, model_subtype))

    # perform some assertions
    assert "DGF" in ctx['bigwig']
    if ctx['meta'] is not None:
        assert ctx['meta'] == ["GEPC{0}".format(i + 1) for i in range(8)]
    if "meta" in ctx['features']:
        assert ctx['meta'] is not None
    assert tf in ctx['tasks']

    ctx = {**ctx,
           # additional features
           **{"tf": tf,
              "model_subtype": model_subtype,

              }}

    # render the templates
    for gen_file in GENERATE_FILES:
        render_template(os.path.join("template", "template_" + gen_file),
                        os.path.join(tf, model_subtype, gen_file),
                        context=ctx,
                        mkdir=True)
