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
                           'dense_3/BiasAdd:0',
                           'reshape_6/Reshape:0',
                           'dense_5/BiasAdd:0',
                           'reshape_8/Reshape:0',
                           'dense_7/BiasAdd:0'
                           ]

    bottleneck_name = 'add_9/add:0'

    target_names = ['Oct4/profile',
                    'Oct4/counts',
                    'Sox2/profile',
                    'Sox2/counts',
                    'Nanog/profile',
                    'Nanog/counts',
                    'Klf4/profile',
                    'Klf4/counts']

    seqlen = 1000

    tasks = ['Oct4', 'Sox2', 'Nanog', 'Klf4']

    postproc_fns = [softmax, None] * 4

    def __init__(self, model_file):
        self.model_file = model_file
        K.clear_session()  # restart session
        self.model = load_model(model_file, compile=False)
        self.contrib_fns = {}

    def predict_on_batch(self, seq):
        preds = self.model.predict_on_batch({"seq": seq, **self.neutral_bias_inputs(len(seq), seqlen=seq.shape[1])})
        pred_dict = {target: preds[i] for i, target in enumerate(self.target_names)}
        return {task: softmax(pred_dict[f'{task}/profile']) * np.exp(pred_dict[f'{task}/counts'][:, np.newaxis])
                for task in self.tasks}

    def neutral_bias_inputs(self, length, seqlen):
        """Compile a set of neutral bias inputs
        """
        return dict([('bias/' + target, np.zeros((length, seqlen, 4))
                      if target.endswith("/profile")
                      else np.zeros((length, 2)))
                     for target in self.target_names])

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
