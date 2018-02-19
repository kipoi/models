# Kipoi models

[![CircleCI](https://circleci.com/gh/kipoi/models.svg?style=svg&circle-token=ee92a92acb288e17399660e66603f700737e7382)](https://circleci.com/gh/kipoi/models)

**Note:** See [kipoi/README.md](https://github.com/kipoi/kipoi) for more information on how to install `kipoi` and access the models from this repository using CLI, python or R.

## Contributing to the Kipoi model zoo

1. Fork this repository
2. Clone your repository fork, ignore all the git lfs files
    - `git lfs clone git@github.com:<username>/models.git '-I /'`
3. Create a new folder `<mynewmodel>` containing all the model files in the repostiory root
    - See how to structure a model directory in [kipoi/kipoi/nbs/contributing_models.ipynb](https://github.com/kipoi/kipoi/blob/master/nbs/contributing_models.ipynb)
	- put all the non-code files (serialized models, test data) into a `*files` directory, where `*` can be anything.
	  - Examples: `model_files`, `dataloader_files`
4. Test your repository locally:
    - `kipoi test <mynewmodel>`
5. Commit, push to your forked remote, submit a pull request
