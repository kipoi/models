## Modular Modeling of Splicing (MMSplice)

MMSplice predicts variant effect with 5 modules scoring exon, donor, acceptor, 3' intron and 5' intron. Modular predictions are combined with a linear model to predict $\Delta logit(\Psi)$ or witha logistic regression model to predict variant pathogenicity.

This repository hosts following models:

`deltaLogitPSI`: predict $\Delta logit(\Psi)$. Returns one prediction per variant-exon pair. 

`pathogenicity`: predict variant pathogenicity. Returns one prediction per variant.

`modularPredictions`: the raw predictions from the five modules for reference sequence and alternative sequence. Returns a vector of length 10 for each variant-exon pair. 

`deltaLogitPSI` and `pathogenicity` differ by the last modular combination model. The model modularPredictions returns.

## Use example
------

### Preparation
------

#### 1. Prepare annotation (gtf) file
Standard human gene annotation file in GTF format can be downloaded from ensembl or gencode.
`MMSplice` can work directly with those files, however, some filtering is higly recommended.

- Filter for protein coding genes.
- Filter out duplicated exons. The same exon can be annotated multiple times if it appears in multiple transcripts. 
  This will cause duplicated predictions.

We provide a filtered version [here](https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/master/data/shared/Homo_sapiens.GRCh37.75.chr.uniq_exon.gtf.gz). 
Note this version has chromosome names in the format `chr*`. You may need to remove them to match the chromosome names in your fasta file.

#### 2. Prepare variant (VCF) file
A correctly formatted VCF file with work with `MMSplice`, however the following steps will make it less prone to false positives:

- Quality filtering. Low quality variants leads to unreliable predictions.
- Avoid presenting multiple variants in one line by splitting them into multiple lines. Example code to do it:
  ```bash
  bcftools norm -m-both -o out.vcf in.vcf.gz
  ```
- Left-normalization. For instance, GGCA-->GG is not left-normalized while GCA-->G is. Details for unified representation of genetic variants see [Tan et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4481842/)
  ```bash
  bcftools norm -f reference.fasta -o out.vcf in.vcf
  ```
  
#### 3. Prepare reference genome (fasta) file
Human reference fasta file can be downloaded from ensembl/gencode. Make sure the chromosome name matches with GTF annotation file you use.

### Example code
------
Run deltaLogitPSI prediction on example files from the command line with `kipoi predict`, other models follow the same logic:

```bash
kipoi predict -k MMSplice/deltaLogitPSI --dataloader_args='{'fasta_file': 'deltaLogitPSI/example_files/hg19.nochr.chr17.fa', 'gtf_file': 'deltaLogitPSI/example_files/test.gtf', 'vcf_file': 'deltaLogitPSI/example_files/test.vcf.gz'}' -o test.tsv

```