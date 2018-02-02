# DeepSEA

## Prediction of epigenetic features from DNA sequence


## Description

This CNN is based on the DeepSEA model from Zhou and Troyanskaya (2015). It categorically predicts 918 cell type-specific epigenetic features from DNA sequence. The model is trained on publicly available ENCODE and Roadmap Epigenomics data and on DNA sequences of size 1000bp. The input of the tensor has to be (N, 1000, 4) for N samples, 1000bp window size and 4 nucleotides. Per sample, 918 probabilities of showing a specific epigentic feature will be predicted. 
