import h5py
import pandas as pd
from concise.preprocessing import encodeDNA

df = pd.read_pickle("human_utrs_result.pkl")

top_n = 2000
inputs = encodeDNA(df.utr)[:top_n]
preds = df.retrained_pred.values.reshape((-1, 1))[:top_n]

fw = h5py.File("expect.human_utrs.h5", 'w')
fw.create_dataset('/inputs', data=inputs)
fw.create_dataset('/preds', data=preds)
fw.flush()
fw.close()