#!/bin/bash

# Script to go from: https://github.com/uci-cbcl/FactorNet/ to kipoi models
# Use snakemake

# rule git_clone - 0. git clone the permalink to the repo 

# 1. copy over the files
# 2. bigwig.txt - check the bigwigs
# - DGF (DNase)
# - Unique35 (mappability track)
# 3. chip.txt (tasks)
# - [ ] save to column names
# 4. feature.txt (input features)
# - softlink differnet dataloaders for it
# 5. meta.txt assert it's allways the same

# Assert DGF is always one of the bigwigs

# QUESTION: can we always just softlink the dataloaders?
