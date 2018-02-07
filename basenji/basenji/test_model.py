from model import Basenji

import tensorflow as tf
from concise.preprocessing import encodeDNA
tf.reset_default_graph()

# TODO - unbalanced...
b = Basenji()

x = encodeDNA(["A" * b.seq_len])
x.shape
out = b.predict_on_batch(x)
out.min()
out.max()


x = encodeDNA(["C" * b.seq_len])
x.shape
out = b.predict_on_batch(x)
out.min()
out.max()

x = encodeDNA(["G" * b.seq_len])
x.shape
out = b.predict_on_batch(x)
out.min()
out.max()


x = encodeDNA(["T" * b.seq_len])
x.shape
out = b.predict_on_batch(x)
out.min()
out.max()


from pysam import FastaFile

fa =FastaFile("example_files/hg19.ml.fa")

from pybedtools import Interval


interval = Interval("chr1", start=10000, end=10000 + b.seq_len, strand="+")

seq = fa.fetch(str(interval.chrom), interval.start, interval.stop)

x = encodeDNA(["A" * b.seq_len])
x.shape
out = b.predict_on_batch(x)
out.min()
out.max()
