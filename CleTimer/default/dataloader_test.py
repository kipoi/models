import dataloader
import os
import inspect
this_file_path = os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
this_dir = os.path.dirname(this_file_path)
ds = dataloader.IntronsDataset(fasta_file = "example_files/chr21.fa",\
                               gtf_file = "example_files/hg38.chr21_UCSC_allGenes.gtf",\
                               create_introns = True)
print(ds.introns[20].attributes)
print(ds.introns[20].chrom)
print(ds.introns[20].strand)
print(ds.introns[20].start)
print(ds.introns[20].end)
