#!/bin/bash

# Adopted from the Bioconda CircleCI setup:
# https://github.com/bioconda/bioconda-recipes/blob/master/.circleci/setup.sh

set -eu

WORKSPACE=`pwd`

# TODO - test with specific Kipoi version
# - have a config in the models file storing the current Kipoi version to use


# -----------  Not needed now
# Common definitions from latest bioconda-utils master have to be downloaded before setup.sh is executed.
# This file can be used to set BIOCONDA_UTILS_TAG and MINICONDA_VER.
# source .circleci/common.sh
# --------------------------------------------

# Make sure the CircleCI config is up to date.
# add upstream as some semi-randomly named temporary remote to diff against
UPSTREAM_REMOTE=__upstream__$(tr -dc a-z < /dev/urandom | head -c10)
git remote add -t master $UPSTREAM_REMOTE https://github.com/kipoi/models.git
git fetch $UPSTREAM_REMOTE
if ! git diff --quiet HEAD...$UPSTREAM_REMOTE/master -- .circleci/; then
    echo 'The CI configuration is out of date.'
    echo 'Please merge in models:master.'
    exit 1
fi
git remote remove $UPSTREAM_REMOTE


# if ! type kipoi > /dev/null; then
echo "Configure Kipoi"
mkdir -p ~/.kipoi
echo "
model_sources:
  kipoi:
    type: git-lfs
    remote_url: git@github.com:kipoi/models.git
    local_path: /root/repo/
" > ~/.kipoi/config.yaml

# ---

echo "Installing Kipoi."
# install kipoi and others from the master branch for now
pip install git+https://github.com/kipoi/kipoi-utils
pip install git+https://github.com/kipoi/kipoi-conda
pip install git+https://github.com/kipoi/kipoi.git@h5-chunk-size
# pip install kipoi

# fi

# Fetch the master branch for comparison (this can fail locally, if git remote 
# is configured via ssh and this is executed in a container).
git fetch origin +master:master || true
