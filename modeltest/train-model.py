import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from sklearn import datasets
from keras.models import Model
import keras.layers as kl
import pickle

iris = datasets.load_iris()


y_transformer = LabelBinarizer().fit(iris["target"])
x_transformer = StandardScaler().fit(iris["data"])

x = x_transformer.transform(iris["data"])
y = y_transformer.transform(iris["target"])


inp = kl.Input(shape=(4, ), name="features")
out = kl.Dense(units=3)(inp)
model = Model(inp, out)
model.compile("adam", "categorical_crossentropy")

model.fit(x, y, verbose=0)

with open("tmp/y_transformer.pkl", "wb") as f:
    pickle.dump(y_transformer, f, protocol=2)

with open("tmp/x_transformer.pkl", "wb") as f:
    pickle.dump(x_transformer, f, protocol=2)

# Architecture
with open("tmp/model.json", "w") as f:
    f.write(model.to_json())

# Weights
model.save_weights("tmp/weights.h5")    

# select first 20 rows of the iris dataset
X = pd.DataFrame(iris["data"][:20], columns=iris["feature_names"])
y = pd.DataFrame({"class": iris["target"][:20]})
# store the model input features and targets as csv files with column names:
X.to_csv("tmp/example_features.csv", index=False)
y.to_csv("tmp/example_targets.csv", index=False)

model_yaml = """
defined_as: kipoi.model.KerasModel  # use `kipoi.model.KerasModel`
args:  # arguments of `kipoi.model.KerasModel`
    arch: tmp/model.json
    weights: tmp/weights.h5
default_dataloader: . # path to the dataloader directory. Here it's defined in the same directory
info: # General information about the model
    authors: 
        - name: Your Name
          github: your_github_username
          email: your_email@host.org
    doc: Model predicting the Iris species
    cite_as: https://doi.org:/... # preferably a doi url to the paper
    trained_on: Iris species dataset (http://archive.ics.uci.edu/ml/datasets/Iris) # short dataset description
    license: MIT # Software License - defaults to MIT
dependencies:
    conda: # install via conda
      - python=3.5
      - h5py
      # - soumith::pytorch  # specify packages from other channels via <channel>::<package>      
    pip:   # install via pip
      - keras>=2.0.4
      - tensorflow>=1.0
schema:  # Model schema
    inputs:
        features:
            shape: (4,)  # array shape of a single sample (omitting the batch dimension)
            doc: "Features in cm: sepal length, sepal width, petal length, petal width."
    targets:
        shape: (3,)
        doc: "One-hot encoded array of classes: setosa, versicolor, virginica."
"""
with open("model.yaml", "w") as ofh:
    ofh.write(model_yaml)


dataloader_yaml = """
type: Dataset
defined_as: dataloader.MyDataset
args:
    features_file:
        # descr: > allows multi-line fields
        doc: >
          Csv file of the Iris Plants Database from
          http://archive.ics.uci.edu/ml/datasets/Iris features.
        type: str
        example: tmp/example_features.csv  # example files
    x_transformer:
        default: tmp/x_transformer.pkl
        #default:
        #      url: https://github.com/kipoi/kipoi/raw/57734d716b8dedaffe460855e7cfe8f37ec2d48d/example/models/sklearn_iris/dataloader_files/x_transformer.pkl
        #      md5: bc1bf3c61c418b2d07506a7d0521a893
    y_transformer:
        default: tmp/y_transformer.pkl
    targets_file:
        doc: >
          Csv file of the Iris Plants Database targets.
          Not required for making the prediction.
        type: str
        example: tmp/example_targets.csv
        optional: True  # if not present, the `targets` field will not be present in the dataloader output

info:
    authors: 
        - name: Your Name
          github: your_github_account
          email: your_email@host.org
    version: 0.1
    doc: Model predicting the Iris species
dependencies:
    conda:
      - python=3.5
      - pandas
      - numpy
      - sklearn
output_schema:
    inputs:
        features:
            shape: (4,)
            doc: "Features in cm: sepal length, sepal width, petal length, petal width."
    targets:
        shape: (3, )
        doc: "One-hot encoded array of classes: setosa, versicolor, virginica."
    metadata:  # field providing additional information to the samples (not directly required by the model)
        example_row_number:
            doc: Just an example metadata column
"""
with open("dataloader.yaml", "w") as ofh:
    ofh.write(dataloader_yaml)


dataloader_py = """
import pickle
from kipoi.data import Dataset
import pandas as pd
import numpy as np

def read_pickle(f):
    with open(f, "rb") as f:
        return pickle.load(f)

class MyDataset(Dataset):

    def __init__(self, features_file, targets_file=None, x_transformer=None, y_transformer=None):
        self.features_file = features_file
        self.targets_file = targets_file

        self.y_transformer = read_pickle(y_transformer)
        self.x_transformer = read_pickle(x_transformer)

        self.features = pd.read_csv(features_file)
        if targets_file is not None:
            self.targets = pd.read_csv(targets_file)
            assert len(self.targets) == len(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x_features = np.ravel(self.x_transformer.transform(self.features.iloc[idx].values[np.newaxis]))
        if self.targets_file is None:
            y_class = {}
        else:
            y_class = np.ravel(self.y_transformer.transform(self.targets.iloc[idx].values[np.newaxis]))
        return {
            "inputs": {
                "features": x_features
            },
            "targets": y_class,
            "metadata": {
                "example_row_number": idx
            }
        }
"""
with open("dataloader.py", "w") as ofh:
    ofh.write(dataloader_py)
