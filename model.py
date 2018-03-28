"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import numpy as np
import tensorflow as tf


def orthogonal(shape, name=None):
    assert len(shape)>1

    shape0 = shape
    if len(shape)>2:
        shape = [shape[0],np.prod(shape[1:])]
    w = np.random.randn(*2*[np.max(shape)])
    u,*_ = np.linalg.svd(w)
    u = u[:shape[0],:shape[1]]
    u = np.reshape(u,shape0)
    return tf.Variable(u.astype('float32'), name=name)


def uniform(shape, name=None):
    limit = shape[-1]/6
    return tf.Variable(tf.random_uniform(shape,-limit,limit), name=name)


def bias(shape, init_value=0., name=None):
    if not isinstance(shape,list):
        shape = [shape]
    return tf.Variable(tf.constant(init_value, shape=shape), name=name)


class GRU(object):
    def __init__(self, input, n_hidden, training=None, zoneout=0.5, h0=None,
                 name='gru'):
        """
        A layer of gated recurrent units

        :param input: input tensor (time x batch x n_in)
        :param n_hidden: number of hidden units
        :param training: placeholder to toggle training/validation
        :param zoneout: zoneout rate (training toggles functionality)
        :param h0: placeholder for initial hidden state
                   note: can be fed in from decoupled neural interface
        :param name: layer name to prepend to variable names in the graph
        """
        assert len(input.shape.as_list())==3
        assert isinstance(n_hidden, int)
        assert isinstance(zoneout, (int,float))
        assert 0<=zoneout<1

        seq_len, _, n_in = input.shape.as_list()

        if training is None:
            self.training = tf.placeholder(tf.int32, name=name+'_training')
        else:
            self.training = training
        is_training = tf.equal(self.training, 1)

        # Initial state
        if h0 is None:
            self.h0_default = tf.Variable(tf.random_normal([1, n_hidden]))
            self.h0 = tf.placeholder_with_default(self.h0_default,
                                                  [None, n_hidden],
                                                  name=name+'_h0')
        else:
            self.h0 = tf.placeholder_with_default(h0, [None, n_hidden],
                                                  name=name+'_h0')

        # Zoneout
        bern = tf.distributions.Bernoulli(zoneout, dtype=tf.float32)
        def zoneout_train(h_tm1, h_t):
            zoneout_mask = bern.sample(n_hidden)
            return zoneout_mask*h_tm1+(1-zoneout_mask)*h_t
        def zoneout_test(h_tm1, h_t):
            return zoneout*h_tm1+(1-zoneout)*h_t

        # Initialize
        w_g = orthogonal([n_in, 2*n_hidden], name+'_wg')
        w_h = orthogonal([n_in, n_hidden], name+'_wh')
        u_g = orthogonal([n_hidden, 2*n_hidden], name+'_ug')
        u_h = orthogonal([n_hidden, n_hidden], name+'_uh')
        # initialize reset bias high: avoid resetting lots early in learning
        b_r = bias(n_hidden, 1., name+'_br')
        b_z = bias(n_hidden, 0., name+'_bz')
        b_g = tf.concat([b_r, b_z], axis=0)
        b_h = bias(n_hidden, 0., name+'_bh')

        def timestep(h,x):
            g = tf.nn.sigmoid(tf.matmul(x,w_g)+tf.matmul(h,u_g)+b_g)
            r,z = tf.split(g, num_or_size_splits=2, axis=1)
            h = z*h+(1-z)*tf.nn.tanh(tf.matmul(x,w_h)+tf.matmul(r*h,u_h)+b_h)
            return h

        if zoneout>0:
            def train(h_tm1, x):
                return zoneout_train(h_tm1, timestep(h_tm1, x))

            def val(h_tm1, x):
                return zoneout_test(h_tm1, timestep(h_tm1, x))
        else:
            train = timestep
            val = timestep

        self.output = tf.cond(
            is_training,
            lambda: tf.scan(train, input, initializer=self.h0),
            lambda: tf.scan(val, input, initializer=self.h0),
            name=name+'_output')


class Dense(object):
    def __init__(self, input, n_out, activation=None, name='dense',
                 init='orthogonal', n_in='auto'):
        """
        A single fully-connected layer

        :param input: input tensor
        :param n_out: output dimensionality
        :param activation: activation function to apply
        :param name: layer name to prepend to variable names in the graph
        """
        assert isinstance(n_out, int)

        if n_in=='auto':
            n_in = input.shape.as_list()[-1]

        if init=='orthogonal':
            w = orthogonal([n_in,n_out], name+'_w')
        else:
            w = uniform([n_in,n_out], name+'_w')
        b = bias(n_out, 0., name+'_b')

        if activation is None:
            activation = tf.identity

        self.output = activation(tf.tensordot(input, w, axes=[[-1],[0]])+b,
                                 name=name+'_output')