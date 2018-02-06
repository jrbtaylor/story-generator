"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import json
import os

import tensorflow as tf

from model import GRU, Dense, orthogonal
from pipeline import Vector_Pipeline
from train import fit


def train_model(exp_name, train_tfrecord, val_tfrecord, dictionary_file,
                n_hidden, learn_rate, batch_size, patience=10, max_epochs=200,
                sample_length=16, resume=False):

    exp_dir = os.path.join(os.path.expanduser('~/experiments/story-gen/'),
                           exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    with open(dictionary_file,'r') as f:
        reverse_dict = json.load(f)  # word -> int
    reverse_dict = {v:k for k,v in reverse_dict.items()}  # int -> word
    dict_size = len(reverse_dict)

    if not resume:
        pipeline = Vector_Pipeline(train_tfrecord, val_tfrecord, batch_size)
        init_train, init_val = pipeline.init_train, pipeline.init_val
        # tf.scan dims: time x batch x feature
        int_input = tf.transpose(pipeline.output, [1,0,2])
        model_input = tf.placeholder_with_default(int_input, [None,], 'input')

        training_toggle = tf.placeholder(tf.int32, name='training_toggle')
        # model part1: embedding
        embedding = orthogonal([dict_size,n_hidden], 'embedding')
        embedded_input = tf.nn.embedding_lookup(embedding, model_input)
        int_label = model_input[1:]
        embedded_input = embedded_input[:-1]
        # model part2: stacked GRUs
        gru0 = GRU(embedded_input, n_hidden, training_toggle, name='gru0')
        gru1 = GRU(gru0.output, n_hidden, training_toggle, name='gru1')
        # model part3: dropout and dense layer
        dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
        dropped = tf.nn.dropout(gru1.output, 1-dropout_rate)
        dense = Dense(dropped, dict_size)
        model_output = tf.identity(dense.output, 'output')

        # cross-entropy loss
        # note: sequences padded with -1, mask these entries
        mask = tf.not_equal(int_label, -1)
        # swap -1's to avoid error in loss fcn, even though we're ignoring these
        int_label = tf.where(mask, int_label, tf.zeros_like(int_label))
        # mean over entries with mask==1
        mask = tf.cast(mask, dtype=tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=int_label, logits=model_output)
        loss = tf.reduce_sum(mask*loss)/tf.reduce_sum(mask)

        train_step = tf.train.AdamOptimizer(learn_rate).minimize(
            loss,name='train_step')
    else:
        (model_input, training_toggle, dropout_rate, train_step,
         init_train, init_val, loss) = reload_graph(exp_dir)

    # TODO: add callback (w/ input arg sess) to sample sentence after each epoch, define here and add to train.py --------------------------------------------------------

    fit(training_toggle, dropout_rate, train_step, init_train, init_val, loss,
        patience, max_epochs, exp_dir, resume)



def reload_graph(exp_dir):
    model_file = [f for f in os.listdir(exp_dir) if f.endswith('.meta')]
    if len(model_file)!=1:
        raise ValueError
    model_file = os.path.join(exp_dir,model_file[0])

    tf.train.import_meta_graph(model_file)
    graph = tf.get_default_graph()
    nodes = [node for node in graph.as_graph_def().node]

    def get_tensor(name):
        name = next(node.name for node in nodes if name == node.name)+':0'
        return graph.get_tensor_by_name(name)

    def get_op(name):
        name = next(node.name for node in nodes if node.name.endswith(name))
        return graph.get_operation_by_name(name)

    model_input = get_tensor('input')
    training_toggle = get_tensor('training_toggle')
    dropout_rate = get_tensor('dropout_rate')
    train_step = get_op('train_step')
    init_train = get_op('init_train')
    init_val = get_op('init_val')
    loss = get_tensor('loss')

    return (model_input, training_toggle, dropout_rate, train_step,
            init_train, init_val, loss)

