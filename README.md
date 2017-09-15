# models

Repository hosting Kipoi models

## Setup - Install git lfs

```bash
# on Ubuntu
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
git-lfs install
```

## Contributing to the model zoo

1. Fork this repository
2. Clone your repository fork, ignore all the git lfs files
  - `git lfs clone git@github.com:<username>/models.git '-I /'`
3. Create a new folder `<mynewmodel>` containing all the model files in the repostiory root
  - See how to contribute models in [kipoi/model-zoo/docs/contributing_models.md](https://github.com/kipoi/model-zoo/blob/master/docs/contributing_models.md)
4. Test your repository locally:
  - `modelzoo test <mynewmodel>`
5. Commit, push to your forked remote, submit a pull request


## TODO

### Open questions

- [ ] shall the folder structure be: <github username>__model? (not concerning about the implementation)

## Docs

- put all the non-code files (serialized models, test data) into a `*files` directory, where `*` can be anything.
  - Examples:
    - `model_files`
	- `dataloader_files`
