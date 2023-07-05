from bpnet.seqmodel import SeqModel
from keras.models import load_model
import numpy as np
import bpnet
import tensorflow as tf
from bpnet.functions import softmax
import keras.backend as K
import keras.layers as kl
from kipoi.model import BaseModel

def profile_contrib(p):
    return kl.Lambda(lambda p:
                     K.mean(K.sum(K.stop_gradient(tf.nn.softmax(p, dim=-2)) * p, axis=-2), axis=-1)
                     )(p)


class BPNetOldSeqModel(BaseModel, SeqModel):

    preact_tensor_names = ['reshape_2/Reshape:0',
                           'dense_1/BiasAdd:0',
                           'reshape_4/Reshape:0',
                           'dense_2/BiasAdd:0',
                           'reshape_6/Reshape:0',
                           'dense_3/BiasAdd:0',
                           'reshape_8/Reshape:0',
                           'dense_4/BiasAdd:0',
                           'reshape_10/Reshape:0',
                           'dense_5/BiasAdd:0',
                           'reshape_12/Reshape:0',
                           'dense_6/BiasAdd:0'
                           ]

    bottleneck_name = 'add_9/add:0'

    target_names = ['Bcd/profile',
                    'Bcd/counts',
                    'Cad/profile',
                    'Cad/counts',
                    'Dl/profile',
                    'Dl/counts',
                    'GAF/profile',
                    'GAF/counts',
                    'Twi/profile',
                    'Twi/counts',
                    'Zld/profile',
                    'Zld/counts']

    seqlen = 1000

    tasks = ['Bcd', 'Cad', 'Dl', 'GAF', 'Twi', 'Zld']

    postproc_fns = [softmax, None] * 6

    def __init__(self, model_file):
        self.model_file = model_file
        K.clear_session()  # restart session
        self.model = load_model(model_file, compile=False)
        self.contrib_fns = {}

    def predict_on_batch(self, seq):
        preds = self.model.predict_on_batch(seq)
        pred_dict = {target: preds[i] for i, target in enumerate(self.target_names)}
        return {task: softmax(pred_dict[f'{task}/profile']) * np.exp(pred_dict[f'{task}/counts'][:, np.newaxis])
                for task in self.tasks}

    def get_intp_tensors(self, preact_only=True, graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        intp_targets = []
        for head_name, tensor_name in zip(self.target_names, self.preact_tensor_names):
            tensor = graph.get_tensor_by_name(tensor_name)
            if head_name.endswith("/profile"):
                tensor = profile_contrib(tensor)
            intp_targets.append((head_name, tensor))
        return intp_targets