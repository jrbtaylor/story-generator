"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import json
import os
import time

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar
import tensorflow as tf


def fit(training_toggle, dropout_rate, train_step, init_train, init_val, loss,
        patience, max_epochs, exp_dir, resume=False):

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    stats_file = os.path.join(exp_dir,'stats.json')
    best_checkpoint = os.path.join(exp_dir,'best')
    last_checkpoint = os.path.join(exp_dir,'last')

    def epoch(sess,epoch_num,training):
        bar = ProgressBar()
        losses = []
        if training:
            sess.run(init_train)
            training = 1
            name = '  training'
        else:
            sess.run(init_val)
            training = 0
            name = 'validation'
        start_time = time.time()
        while bar(True):
            try:
                _, batch_loss = sess.run([train_step, loss],
                                         feed_dict={training_toggle:training,
                                                    dropout_rate:0.5*training})
                losses += batch_loss
            except tf.errors.OutOfRangeError:
                break
        clearline()
        mean_loss = np.mean(losses)
        total_time = time.time()-start_time
        print('Epoch %4i - %s:    loss = %.6f    time = %.4f (%.4f s/example)'
              %(epoch_num, name, mean_loss, total_time, total_time/len(losses)))
        return mean_loss

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        if resume:
            with open(stats_file, 'r') as js:
                stats = json.load(js)
            best_val = np.min(stats['loss']['val'])
            stall = len(stats['loss']['val'])-np.argmin(stats['loss']['val'])-1
            start_epoch = len(stats['loss']['val'])-1
            print('Resuming from epoch %i'%start_epoch)
            saver.restore(sess,last_checkpoint)
        else:
            stats = {'loss': {'train': [], 'val': []}}
            if os.path.isfile(meta_file):
                os.remove(meta_file)
            best_val = np.inf
            stall = 0
            start_epoch = 0
            sess.run(tf.local_variables_initializer())

        for epoch_number in range(start_epoch,max_epochs):
            stats['loss']['train'].append(epoch(sess,epoch_number,True))
            stats['loss']['val'].append(epoch(sess,epoch_number,False))

            # save checkpoint & stats, update plots
            meta_saved = os.path.isfile(last_checkpoint+'.meta')
            saver.save(sess, last_checkpoint,
                       write_meta_graph=not meta_saved)
            with open(stats_file,'w') as f:
                json.dump(stats,f)
            plot_stats(stats, exp_dir)

            # early stopping
            if stats['loss']['val'][-1]<best_val:
                best_val = stats['loss']['val'][-1]
                stall = 0
                meta_saved = os.path.isfile(best_checkpoint+'.meta')
                saver.save(sess, best_checkpoint,
                           write_meta_graph=not meta_saved)
            else:
                stall += 1
            if stall >= patience:
                break


def clearline():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)


def plot_stats(stats, save_dir):
    """
    Make all the plots in stats. Stats can be a dict or a path to json (str)
    """
    if type(stats) is str:
        assert os.path.isfile(stats)
        with open(stats,'r') as sf:
            stats = json.load(sf)
    assert type(stats) is dict

    assert type(save_dir) is str
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    def _plot(y,title):
        plt.Figure()
        if type(y) is list:
            plt.plot(range(1,len(y)+1),y)
        elif type(y) is dict:
            for key,z in y.items():
                plt.plot(range(1,len(z)+1),z,label=key)
            plt.legend()
        else:
            raise ValueError
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(title)
        plt.savefig(os.path.join(save_dir,title.replace(' ','_')+'.png'))
        plt.close()

    # Loop over stats dict and plot. Dicts within stats get plotted together
    for key,value in stats.items():
        _plot(value,key)