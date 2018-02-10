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
                n_hidden, learn_rate, batch_size, decouple_split=100,
                patience=10, max_epochs=200, sample_length=16, resume=False):

    exp_dir = os.path.join(os.path.expanduser('~/experiments/story-gen/'),
                           exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    with open(dictionary_file,'r') as f:
        reverse_dict = json.load(f)  # word -> int
    reverse_dict = {v+1:k for k,v in reverse_dict.items()}  # int -> word
    # note: sequences are padded with zero, add to dict_size (for embedding)
    reverse_dict[0] = '_END_'  # this should be removed from sampled output
    dict_size = max(reverse_dict.keys())+1

    if not resume:
        pipeline = Vector_Pipeline(train_tfrecord, val_tfrecord, batch_size)
        init_train, init_val = pipeline.init_train, pipeline.init_val
        model_input = tf.placeholder_with_default(pipeline.output[:-1],
                                                  [None, None], 'input')

        # Embedding
        embedding = orthogonal([dict_size, n_hidden], 'embedding')
        embedded_input = tf.nn.embedding_lookup(embedding, model_input)
        int_label = pipeline.output[1:]

        # TODO: add decoupled neural interfaces to speed this up ----------------------------------
        # 1. reshape input to be slow_time x batch x fast_time
        # 2. define decoupled neural interface as a gru
        # 3. add dni loss printout to training

        # Split to subsequences, reshape to slow_time x batch x fast_time x feat
        seq_len = tf.shape(embedded_input)[1]
        # pad so sequence length is divisible by subsequence length
        decouple_split = 200
        pad_len = decouple_split-tf.mod(seq_len,tf.constant(decouple_split))
        embedded_input = tf.pad(embedded_input, [[0,0], [0,pad_len], [0,0]],
                                mode='CONSTANT', constant_values=0)
        int_label = tf.pad(int_label, [[0,0], [0,pad_len]])
        # batch x features x time
        dni_input = tf.transpose(embedded_input, [0,2,1])
        # batch x features x slow_time x fast_time
        dni_input = tf.reshape(
            dni_input,
            [-1, n_hidden, tf.floor_div(seq_len,decouple_split),decouple_split])
        # fast_time x features x batch x slow_time
        dni_input = tf.transpose(dni_input, [3,1,0,2])
        # fast_time x features x (batch x slow_time)
        dni_input = tf.reshape(dni_input, [decouple_split, n_hidden, -1])
        # (batch x slow_time) x fast_time x features
        dni_input = tf.transpose(dni_input, [2,0,1])
        # (batch x slow_time) x (fast_time x features)
        dni_input = tf.reshape(dni_input, [tf.shape(dni_input)[0],-1])

        # Decoupled neural interface: simplify to single dense layer
        dni = Dense(dni_input, n_hidden, tf.nn.relu, name='dni', init='uniform',
                    n_in=n_hidden*decouple_split)

        # Reshape DNI out & embedded_input to new_batch x fast_time for main GRU
        gru_hidden = tf.reshape(dni.output, [-1, n_hidden])
        embedded_input = tf.reshape(embedded_input,
                                    [-1, decouple_split, n_hidden])
        int_label = tf.reshape(int_label, [-1, decouple_split])

        # model part2: GRU
        # transpose: tf.scan needs time x batch x features
        embedded_input = tf.transpose(embedded_input, [1,0,2])
        training_toggle = tf.placeholder(tf.int32, name='training_toggle')
        gru = GRU(embedded_input, n_hidden, training_toggle, h0=gru_hidden,
                  name='gru')
        # model part3: dropout and dense layer
        dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
        dropped = tf.nn.dropout(gru.output, 1-dropout_rate)
        dense = Dense(dropped, dict_size)
        model_output = tf.identity(dense.output, 'output')

        # decoupled neural interface loss
        dni_label = tf.stop_gradient(gru.output)
        dni_loss = tf.reduce_mean(tf.square(dni_label-dni.output),
                                  name='dni_loss')

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
            loss+dni_loss,name='train_step')
    else:
        (model_input, training_toggle, dropout_rate, train_step,
         init_train, init_val, loss, dni_loss) = reload_graph(exp_dir)

    # TODO: add callback (w/ input arg sess) to sample sentence after each epoch, define here and add to train.py --------------------------------------------------------

    fit(training_toggle, dropout_rate, train_step, init_train, init_val, loss,
        dni_loss, patience, max_epochs, exp_dir, resume)


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
    dni_loss = get_tensor('dni_loss')

    return (model_input, training_toggle, dropout_rate, train_step,
            init_train, init_val, loss, dni_loss)


def generate(sess, model_input, hidden_states, model_output, reverse_dictionary,
             sequence_length=16):
    """
    Generate a story or sample

    :param sess: tf.Session object to run in
    :param model_input: placeholder for model input
    :param hidden_states: list of placeholders/tensors for all the hidden states
                          to allow for sampled output to be fed back in without
                          re-scanning the entire sequence each time
    :param model_output: tensor for model output
    :param reverse_dictionary: dict mapping int to word
    :param sequence_length: number of tokens to sample
    :return: a string sampled from the model output
    """



