"""Basenji model
"""
import numpy as np
import tensorflow as tf
from kipoi.utils import merge_dicts
import six

import tensorflow as tf
from concise.preprocessing import encodeDNA
tf.reset_default_graph()


def dump_const_feed_dict():
    pkl_file = "model_files/const_feed_dict.pkl"
    const_feed_dict={"Placeholder_" + str(i): 0 for i in range(1, 15)}  # cnn_dropout_ph
    const_feed_dict["Placeholder"] = np.zeros((2, 1024)).astype(bool)  # targets_na
    const_feed_dict["Placeholder_14"] = False  # is_training
    const_feed_dict["Placeholder_15"] = np.zeros((2, 960, 4229))  # preds_adhoc
    import pickle
    with open(pkl_file, "wb") as f:
        pickle.dump(const_feed_dict, f, protocol=2)

# input matp

def network_surgery():
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32,
                            shape=(None, 131072, 4),
                            name='inputs')
    targets = tf.placeholder(tf.float32, shape=(None, 1024, 4229),
                             name='targets')
    targets_na = tf.placeholder(tf.bool, shape=(None, 1024), name="targets_na")
    preds_adhoc = tf.placeholder(tf.float32, shape=(None, 960, 4229), name="Placeholder_15")


    saver = tf.train.import_meta_graph("model_files/model.tf.meta",
                                       input_map={'Placeholder_15:0': preds_adhoc,
                                                  'Placeholder:0': targets_na,
                                                  'inputs:0': inputs,
                                                  'targets:0': targets
                                       })

    ops = tf.get_default_graph().get_operations()

    out = tf.train.export_meta_graph(filename='model_files/model.tf-modified.meta', as_text=True)

    ops[:15]

def test_tf_model():
    tf.reset_default_graph()
    input_nodes = "inputs"
    target_nodes = "preds"
    meta_graph = "model_files/model.tf.meta"
    # meta_graph = 'model_files/model.tf-modified.meta'
    checkpoint = "model_files/model.tf"
    index = "model_files/model.tf.index"
    pkl_file = "model_files/const_feed_dict.pkl"

    from kipoi.model import TensorFlowModel

    m = TensorFlowModel(input_nodes="inputs",
                        target_nodes="preds",
                        meta_graph=meta_graph,
                        checkpoint=checkpoint,
                        const_feed_dict_pkl=pkl_file)
    ops = tf.get_default_graph().get_operations()



    # TODO - modify the 
    out = tf.train.export_meta_graph(filename='model_files/model.tf-modified.meta', as_text=True)
    ops[0].outputs[0].shape[0] = None

    pops = [op.outputs[0] for op in ops
           if op.type == "Placeholder" and
           op.name.startswith("Placeholder")]
    
    m.input_ops  # view shapes of the data
    m.target_ops
    
    from concise.preprocessing import encodeDNA

    x = encodeDNA(["T" * m.input_ops.shape[1].value]*2).astype("float32")
    out = m.predict_on_batch(x)

    # pred_shape = (batch_size, seq_length // target_pool,
    #               num_targets)

    # # (self.batch_size, self.seq_length, self.seq_depth)

    # # batches
    # self.inputs = tf.placeholder(
    #     tf.float32,
    #     shape=(self.batch_size, self.seq_length, self.seq_depth),
    #     name='inputs')
    # if self.target_classes == 1:
    #     self.targets = tf.placeholder(
    #         tf.float32,
    #         shape=(self.batch_size, self.seq_length // self.target_pool,
    #                self.num_targets),
    #         name='targets')
    # else:
    #     self.targets = tf.placeholder(
    #         tf.int32,
    #         shape=(self.batch_size, self.seq_length // self.target_pool,
    #                self.num_targets),
    #         name='targets')
    # self.targets_na = tf.placeholder(
    #     tf.bool, shape=(self.batch_size, self.seq_length // self.target_pool))
