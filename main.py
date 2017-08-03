# encoding: utf-8

'''

@author: ZiqiLiu


@file: main.py

@time: 2017/5/18 上午11:03

@desc:
'''

# -*- coding:utf-8 -*-
# !/usr/bin/python

import os
import sys
import time
import pickle
import signal
import traceback
from args import parse_args
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.python.framework import graph_util
from reader import read_dataset
from utils.common import check_dir, path_join
from utils.prediction import evaluate, ctc_predict, ctc_decode
from octbit.octbit_graph import GraphRewriter

from utils.wer import WERCalculator

DEBUG = False


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.wer_cal = WERCalculator([0, -1])

    def run(self, TrainingModel):

        graph = tf.Graph()
        with graph.as_default(), tf.Session() as sess:

            self.data = read_dataset(self.config)

            if config.mode == 'train':
                print('building training model....')
                with tf.variable_scope("model"):
                    self.train_model = TrainingModel(self.config,
                                                     self.data.batch_input_queue(),
                                                     is_train=True)
                    self.train_model.config.show()
                print('building valid model....')
                with tf.variable_scope("model", reuse=True):
                    self.valid_model = TrainingModel(self.config,
                                                     self.data.valid_queue(),
                                                     is_train=False)
            else:
                with tf.variable_scope("model", reuse=False):
                    self.valid_model = TrainingModel(self.config,
                                                     self.data.valid_queue(),
                                                     is_train=False)
            saver = tf.train.Saver()

            # restore from stored models
            files = glob(path_join(self.config.model_path, '*.ckpt.*'))

            if len(files) > 0:

                saver.restore(sess, path_join(self.config.model_path,
                                              self.config.model_name))
                print(('Model restored from:' + self.config.model_path))
            else:
                print("Model doesn't exist.\nInitializing........")
                sess.run(tf.global_variables_initializer())

            sess.run(tf.local_variables_initializer())
            tf.Graph.finalize(graph)

            st_time = time.time()
            if os.path.exists(path_join(self.config.model_path, 'best.pkl')):
                with open(path_join(self.config.model_path, 'best.pkl'),
                          'rb') as f:
                    best_miss, best_false = pickle.load(f)
                    print('best miss', best_miss, 'best false', best_false)
            else:
                print('best not exist')

            check_dir(self.config.model_path)

            if self.config.mode == 'train':
                best_miss = 1
                best_false = 1
                accu_loss = 0
                epoch_step = config.tfrecord_size * self.data.train_file_size // config.batch_size

                if self.config.reset_global:
                    sess.run(self.train_model.reset_global_step)

                def handler_stop_signals(signum, frame):
                    global run
                    run = False
                    if not DEBUG:
                        print(
                            'training shut down, total setp %s, the model will be save in %s' % (
                                step, self.config.model_path))
                        saver.save(sess, save_path=(
                            path_join(self.config.model_path, 'latest.ckpt')))
                        print('best miss rate:%f\tbest false rate %f' % (
                            best_miss, best_false))
                    sys.exit(0)

                signal.signal(signal.SIGINT, handler_stop_signals)
                signal.signal(signal.SIGTERM, handler_stop_signals)

                best_list = []
                best_threshold = 0.08
                best_count = 0
                # (miss,false,step,best_count)

                last_time = time.time()

                try:
                    sess.run([self.data.noise_stage_op,
                              self.data.noise_filequeue_enqueue_op,
                              self.train_model.stage_op,
                              self.train_model.input_filequeue_enqueue_op,
                              self.valid_model.stage_op,
                              self.valid_model.input_filequeue_enqueue_op])

                    va = tf.trainable_variables()
                    for i in va:
                        print(i.name)
                    while self.epoch < self.config.max_epoch:

                        _, _, _, _, _, l, lr, step, grads = sess.run(
                            [self.train_model.train_op,
                             self.data.noise_stage_op,
                             self.data.noise_filequeue_enqueue_op,
                             self.train_model.stage_op,
                             self.train_model.input_filequeue_enqueue_op,
                             self.train_model.loss,
                             self.train_model.learning_rate,
                             self.train_model.global_step,
                             self.train_model.grads
                             ])
                        epoch = step // epoch_step
                        accu_loss += l
                        if epoch > self.epoch:
                            self.epoch = epoch
                            print('accumulated loss', accu_loss)
                            saver.save(sess, save_path=(
                                path_join(self.config.model_path,
                                          'latest.ckpt')))
                            print('latest.ckpt save in %s' % (
                                path_join(self.config.model_path,
                                          'latest.ckpt')))
                            accu_loss = 0
                        if step % config.valid_step == 0:
                            print('epoch time ', (time.time() - last_time) / 60)
                            last_time = time.time()

                            miss_count = 0
                            false_count = 0
                            target_count = 0
                            wer = 0
                            valid_batch = self.data.valid_file_size * config.tfrecord_size // config.batch_size
                            text = ""
                            for i in range(valid_batch):
                                softmax, correctness, labels, _, _ = sess.run(
                                    [self.valid_model.softmax,
                                     self.valid_model.correctness,
                                     self.valid_model.labels,
                                     self.valid_model.stage_op,
                                     self.valid_model.input_filequeue_enqueue_op])
                                np.set_printoptions(precision=4,
                                                    threshold=np.inf,
                                                    suppress=True)

                                decode_output = [ctc_decode(s) for s in softmax]
                                for i in decode_output:
                                    text += str(i) + '\n'
                                    text += str(labels) + '\n'
                                    text += '=' * 20 + '\n'
                                result = [ctc_predict(seq, config.label_seqs)
                                          for seq in
                                          decode_output]
                                miss, target, false_accept = evaluate(
                                    result, correctness.tolist())

                                miss_count += miss
                                target_count += target
                                false_count += false_accept

                                wer += self.wer_cal.cal_batch_wer(labels,
                                                                  decode_output).sum()
                                # print(miss_count, false_count)
                            with open('./valid.txt', 'w') as f:
                                f.write(text)

                            miss_rate = miss_count / target_count
                            false_accept_rate = false_count / (
                                self.data.validation_size - target_count)
                            print('--------------------------------')
                            print('epoch %d' % self.epoch)
                            print('training loss:' + str(l))
                            print('learning rate:', lr, 'global step', step)
                            print('miss rate:' + str(miss_rate))
                            print('flase_accept_rate:' + str(false_accept_rate))
                            print(miss_count, '/', target_count)
                            print('wer', wer / self.data.validation_size)

                            if miss_rate + false_accept_rate < best_miss + best_false:
                                best_miss = miss_rate
                                best_false = false_accept_rate
                                saver.save(sess,
                                           save_path=(path_join(
                                               self.config.model_path,
                                               'best.ckpt')))
                                with open(path_join(
                                        self.config.model_path, 'best.pkl'),
                                        'wb') as f:
                                    best_tuple = (best_miss, best_false)
                                    pickle.dump(best_tuple, f)
                            if miss_rate + false_accept_rate < best_threshold:
                                best_count += 1
                                print('best_count', best_count)
                                best_list.append((miss_rate,
                                                  false_accept_rate, step,
                                                  best_count))
                                saver.save(sess,
                                           save_path=(path_join(
                                               self.config.model_path,
                                               'best' + str(
                                                   best_count) + '.ckpt')))

                    print(
                        'training finished, total epoch %d, the model will be save in %s' % (
                            self.epoch, self.config.model_path))
                    saver.save(sess, save_path=(
                        path_join(self.config.model_path, 'latest.ckpt')))
                    print('best miss rate:%f\tbest false rate"%f' % (
                        best_miss, best_false))

                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                finally:
                    with open('best_list.pkl', 'wb') as f:
                        pickle.dump(best_list, f)
                    print('total time:%f hours' % (
                        (time.time() - st_time) / 3600))
                    # When done, ask the threads to stop.

            else:
                with open(
                                        config.rawdata_path + 'valid/' + "ctc_valid.pkl.sorted",
                        'rb') as f:
                    pkl = pickle.load(f)
                miss_count = 0
                false_count = 0
                target_count = 0

                valid_batch = self.data.valid_file_size * config.tfrecord_size // config.batch_size
                _, _ = sess.run([self.valid_model.stage_op,
                                 self.valid_model.input_filequeue_enqueue_op])
                for i in range(valid_batch):
                    # if i > 7:
                    #     break

                    softmax, ctc_input, correctness, labels, _, _ = sess.run(
                        [self.valid_model.softmax,
                         self.valid_model.nn_outputs,
                         self.valid_model.correctness,
                         self.valid_model.labels,
                         self.valid_model.stage_op,
                         self.valid_model.input_filequeue_enqueue_op
                         ])
                    np.set_printoptions(precision=4,
                                        threshold=np.inf,
                                        suppress=True)

                    correctness = correctness.tolist()
                    decode_output = [ctc_decode(s) for s in softmax]
                    result = [ctc_predict(seq, config.label_seqs)
                              for seq in
                              decode_output]
                    for k, r in enumerate(result):
                        if r == 1 and correctness[k] == 0:
                            # if r != correctness[k]:
                            name = pkl[i * config.batch_size + k][0]
                            # print("scp liuziqi@10.8.0.62:/ssd/keyword_raw/valid/%s ./"%name)
                            print(pkl[i * config.batch_size + k])
                            print(decode_output[k])
                            print(labels[k])
                            with open('logits.txt', 'w') as f:
                                f.write(str(ctc_input[k]))

                    miss, target, false_accept = evaluate(
                        result, correctness)

                    miss_count += miss
                    target_count += target
                    false_count += false_accept

                print('--------------------------------')
                print('miss rate: %d/%d' % (miss_count, target_count))
                print('flase_accept_rate: %d/%d' % (
                    false_count, self.data.validation_size - target_count))

    def build_graph(self, DeployModel):
        check_dir(self.config.graph_path)
        config_path = path_join(self.config.graph_path, 'config.pkl')
        graph_path = path_join(self.config.graph_path, self.config.graph_name)
        import pickle
        pickle.dump(self.config, open(config_path, 'wb'))

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as session:
            with tf.variable_scope("model"):
                model = DeployModel(config=config)

            print('Graph build finished')
            variable_names = [n.name for n in
                              tf.get_default_graph().as_graph_def().node]
            for n in variable_names:
                print(n)

            saver = tf.train.Saver()
            saver.restore(session, save_path=path_join(self.config.model_path,
                                                       'latest.ckpt'))
            print("model restored from %s" % config.model_path)

            frozen_graph_def = graph_util.convert_variables_to_constants(
                session, session.graph.as_graph_def(),
                ['model/inputX', 'model/rnn_initial_states',
                 'model/rnn_states', 'model/softmax', 'model/logit'])
            tf.train.write_graph(
                frozen_graph_def,
                self.config.graph_path,
                self.config.graph_name,
                as_text=False,
            )
            try:
                tf.import_graph_def(frozen_graph_def, name="")
            except Exception as e:
                print("!!!!Import graph meet error: ", e)
                exit()
            print('graph saved in %s' % graph_path)

            print("====octbit graph====")
            rewriter = GraphRewriter(frozen_graph_def, mode="octbit")
            octbit_graph = rewriter.rewrite(
                ['model/inputX', 'model/rnn_initial_states',
                 'model/rnn_states', 'model/softmax', 'model/logit'])
            try:
                tf.import_graph_def(octbit_graph, name="")
            except Exception as e:
                print("!!!!Import octbit graph meet error: ", e)
                exit()
            tf.train.write_graph(
                octbit_graph,
                self.config.graph_path,
                self.config.graph_name[:-3] + '_octbit.pb',
                as_text=False,
            )


if __name__ == '__main__':

    config, model = parse_args()

    if model == 'rnn':
        from models import rnn_ctc

        TrainingModel = rnn_ctc.GRU
        DeployModel = rnn_ctc.DeployModel
    elif model == 'attention':
        from models import attention_ctc

        TrainingModel = attention_ctc.Attention
        DeployModel = attention_ctc.DeployModel
    else:
        raise Exception('model %s not defined!' % model)

    runner = Runner(config)
    if config.mode == 'build':
        runner.build_graph(DeployModel)
    else:
        runner.run(TrainingModel)
