## Modular Modeling of Splicing (MMSplice)

MMSplice predicts variant effect with 5 modules scoring exon, donor, acceptor, 3' intron and 5' intron. Modular predictions are combined with a linear model to predict $\Delta logit(\Psi)$ or witha logistic regression model to predict variant pathogenicity.

This repository hosts following models:

`deltaLogitPSI`: predict $\Delta logit(\Psi)$. Returns one prediction per variant-exon pair. 

`pathogenicity`: predict variant pathogenicity. Returns one prediction per variant.

`modularPredictions`: the raw predictions from the five modules for reference sequence and alternative sequence. Returns a vector of length 10 for each variant-exon pair. 

`deltaLogitPSI` and `pathogenicity` differ by the last modular combination model. The model modularPredictions returns.

## Use example
Run deltaLogitPSI prediction on example files from the command line with `kipoi predict`, other models follow the same logic:

```bash
kipoi predict -k MMSplice/deltaLogitPSI --dataloader_args='{'fasta_file': 'deltaLogitPSI/example_files/hg19.nochr.chr17.fa', 'gtf_file': 'deltaLogitPSI/example_files/test.gtf', 'vcf_file': 'deltaLogitPSI/example_files/test.vcf.gz'}' -o test.tsv

```