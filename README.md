## Kipoi models

> [!WARNING]
> ### **Kipoi Project - Sunset Announcement**
> 
> After several impactful years, we have made the decision to **archive the Kipoi repositories and end active maintenance** of the project.
> 
> This is a bittersweet moment. While it’s always a little sad to sunset a project, the field of machine learning in genomics has evolved rapidly, with new technologies and platforms emerging that better meet current needs. Kipoi played an important role in its time, helping researchers **share, reuse, and benchmark trained models** in regulatory genomics. We’re proud of what it accomplished and grateful for the strong community support that made it possible.
> 
> Kipoi’s impact continues, however:
> 
> *   [The Kipoi webinar series](seminar.html) will carry on, supporting discussions around model reuse and interpretability.
> *   [Kipoiseq](https://github.com/kipoi/kipoiseq), our standard set of data-loaders for sequence-based modeling, also remains active and relevant.
> 
> Thanks to everyone who contributed, used, or supported Kipoi. It’s been a fantastic journey, and we're glad the project helped shape how models are shared in the field.
> 
> \- The Kipoi Team


[![CircleCI](https://circleci.com/gh/kipoi/models.svg?style=svg&circle-token=ee92a92acb288e17399660e66603f700737e7382)](https://circleci.com/gh/kipoi/models) [![DOI](https://zenodo.org/badge/103403966.svg)](https://zenodo.org/badge/latestdoi/103403966)

This repository hosts predictive models for genomics and serves as a model source for [Kipoi](https://github.com/kipoi/kipoi). Each folder containing `model.yaml` is considered to be a single model.

### Contributing models

1. Install kipoi:
```shell
pip install kipoi
```

2. Run `kipoi ls`. This will checkout the `kipoi/models` repo to `~/.kipoi/models`)


3. Follow the instructions on [contributing/Getting started](https://kipoi.org/docs/contributing/01_Getting_started/).

### Using models (to predict, score variants, build new models)

To explore available models, visit [http://kipoi.org/models](http://kipoi.org/groups/). See [kipoi/README.md](https://github.com/kipoi/kipoi) and [docs/using getting started](http://kipoi.org/docs/using/01_Getting_started/) for more information on how to programatically access the models from this repository using CLI, python or R.

#### Configuring local storage location

This model source (https://github.com/kipoi/models) is included in the Kipoi config file (`~/.kipoi/config.yaml`) by default:

```yaml
# ~/.kipoi/config.yaml
model_sources:
    kipoi:
        type: git-lfs
        remote_url: git@github.com:kipoi/models.git
        local_path: ~/.kipoi/models/
        auto_update: True
```

If you wish to keep the models stored elsewhere, edit the `local_path` accordingly.
