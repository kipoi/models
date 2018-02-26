## Kipoi models

[![CircleCI](https://circleci.com/gh/kipoi/models.svg?style=svg&circle-token=ee92a92acb288e17399660e66603f700737e7382)](https://circleci.com/gh/kipoi/models)

This repository hosts predictive models for genomics and serves as a model source for [Kipoi](https://github.com/kipoi/kipoi). Each folder containing the following files is considered to be a single model:

```
├── dataloader.py     # implements the dataloader
├── dataloader.yaml   # describes the dataloader
├── dataloader_files/      #/ files required by the dataloader
│   ├── ...
│   └── y_transfomer.pkl # example
├── model.yaml        # describes the model
├── model_files/           #/ files required by the model
│   ├── model.json # example
│   └── weights.h5 # example
└── example_files/         #/ small example files used to test the model
    ├── features.csv # example
    └── targets.csv # example
```

Folders named `*_files` are tracked by Git Large File Storage (LFS). New models are added by simply submitting a pull-request to <https://github.com/kipoi/models>.

### Contributing models

#### 1. Install Kipoi

1. Install git-lfs
    - `conda install -c conda-forge git-lfs` (alternatively see <https://git-lfs.github.com/>)
2. Install kipoi
    - `pip install kipoi`
3. Run `kipoi ls` (this will checkout the `kipoi/models` repo to `~/.kipoi/models`)

#### 2. Add the model

0. `cd ~/.kipoi/models`
1. [Write the model](#how-to-write-the-model): Create a new folder `<my new model>` containing all the required files
    - Option 1: Copy the existing model: `cp -R <existing model> <my new model>` & edit the copied files
	- Option 2: Run `kipoi init`, answer the questions & edit the created files
	- Option 3: `mkdir <my new model>` & write all the files from scratch
2. [Test the model](#how-to-test-the-model)
    - Step 1: `kipoi test ~/.kipoi/models/my_new_model`
	- Step 2: `kipoi test-source kipoi --all -k my_new_model`
3. Commit your changes
    - `cd ~/.kipoi/models && git commit -m "Added <my new model>"`

#### 3. Submit the pull-request

1. [Fork](https://guides.github.com/activities/forking/) the <https://github.com/kipoi/models> repo on github (click on the Fork button)
2. Add your fork as a git remote to `~/.kipoi/models`
    - `cd ~/.kipoi/models && git remote add fork https://github.com/<username>/models.git`
3. Push to your fork
    - `git push fork master`
4. Submit a pull-request (click the [New pull request](https://help.github.com/articles/creating-a-pull-request/) button on your github fork - `https://github.com/<username>/models>`)

See [docs/contributing getting started](http://ec2-54-87-147-83.compute-1.amazonaws.com/docs/contributing/01_Getting_started/) and [docs/tutorials/contributing/models](http://ec2-54-87-147-83.compute-1.amazonaws.com/docs/tutorials/contributing_models/) for more information.

### Using models (to predict, score variants, build new models)

To explore available models, visit [http://kipoi.org/models](http://ec2-54-87-147-83.compute-1.amazonaws.com/groups/). See [kipoi/README.md](https://github.com/kipoi/kipoi) and [docs/using getting started](http://ec2-54-87-147-83.compute-1.amazonaws.com/docs/using/01_Getting_started/) for more information on how to programatically access the models from this repository using CLI, python or R.

#### Configuring local storage location

This model source (https://github.com/kipoi/models) is included in the Kipoi config file (`~/.kipoi/config.yaml`) by default:

```yaml
# ~/.kipoi/config.yaml
model_sources:
    kipoi:
        type: git-lfs
        remote_url: git@github.com:kipoi/models.git
        local_path: ~/.kipoi/models/
```

If you wish to keep the models stored elsewhere, edit the `local_path` accordingly.
