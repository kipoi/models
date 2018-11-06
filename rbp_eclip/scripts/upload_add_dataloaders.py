"""Add dataloaders to models.tsv
"""
import os
import hashlib
import pandas as pd


def get_md5(fpath):
    assert os.path.isfile(fpath)
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    return md5o.hexdigest()


if __name__ == '__main__':
    dpath = '/tmp/rbpeclip_dataloaders'
    df = pd.read_table("models.tsv")

    df.head()

    assert len(df) == 112

    if "dl_md5" not in df:
        df['dl_md5'] = df.model.map(lambda m: get_md5(os.path.join(dpath, m + ".dataloader.pkl")))
    if "dl_url" not in df:
        df['dl_url'] = df.model.map(lambda m: f"https://sandbox.zenodo.org/record/248594/files/{m}.dataloader.pkl?download=1")
    # write to a file
    df.to_csv('models.tsv', sep='\t', index=None)
