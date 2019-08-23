from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import math
import time
from absl import flags
from progressbar import ProgressBar
import absl.logging as _logging  # pylint: disable=unused-import
import csv

import tensorflow as tf
import model
import data_utils
from vocabulary import Vocab
from gpu_utils import assign_to_gpu, average_grads_and_vars
from postprocess import top_one_result, top_n_prob, gen_on_keyword, gen_diversity
import numpy as np
import pandas as pd
import multiprocessing
import random

# GPU config
flags.DEFINE_integer("num_core_per_host", default=8, help="Number of cores per host")
flags.DEFINE_integer("multiprocess", default=2, help="Number of processes")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("corpus_info_path", default="", help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default=None, help="Estimator model_dir.")
flags.DEFINE_string("dataset", "tmall", help="Dataset name.")
flags.DEFINE_string("input_file_dir", default=None, help="Input file_dir.")
flags.DEFINE_string("output_file_dir", default=None, help="Output file_dir.")
flags.DEFINE_bool("do_sent_gen", default=False, help="Whether to generate sentence.")
flags.DEFINE_bool("do_sent_ppl_pred", default=False, help="Whether to predict sentence log probability.")
flags.DEFINE_integer("limit_len", default=50, help="Limited length of input sentence.")
flags.DEFINE_integer("gen_len", default=30, help="Number of token to generate.")

# Model config
flags.DEFINE_integer("mem_len", default=10, help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False, help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")

flags.DEFINE_integer("n_layer", default=6, help="Number of layers.")
flags.DEFINE_integer("d_model", default=500, help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500, help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10, help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50, help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000, help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False, help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_integer("div_val", default=1, help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
                  help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True, help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"], help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02, help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01, help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1, help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS

def sent_gen(tmp_Vocab, input_txt, n_token, cutoffs, ps_device):

    test_list = tf.placeholder(tf.int64, shape=[1, None])
    dataset = tf.data.Dataset.from_tensors(test_list)
    # dataset = dataset.batch(1, drop_remainder=True)

    iterator = dataset.make_initializable_iterator()
    input_feed = iterator.get_next()

    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)

    per_core_bsz = 1
    tower_mems, tower_losses, tower_new_mems = [], [], []
    tower_output = []
    tower_mems_id = []
    tower_new_mems_id = []
    tower_attn_prob = []

    for i in range(FLAGS.num_core_per_host):
        with tf.device(assign_to_gpu(i, ps_device)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            mems_i = [tf.placeholder(tf.float32,
                                     [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]

            mems_i_id = [tf.placeholder(tf.int64,
                                        [FLAGS.mem_len, per_core_bsz])
                         for _ in range(FLAGS.n_layer)]

            new_mems_i, output_i, new_mems_i_id, attn_prob_i = single_core_graph_for_inference(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=False,
                inp=inputs[i],
                mems=mems_i,
                mems_id=mems_i_id)

            tower_mems.append(mems_i)
            tower_new_mems.append(new_mems_i)
            tower_output.append(output_i)
            tower_mems_id.append(mems_i_id)
            tower_new_mems_id.append(new_mems_i_id)
            tower_attn_prob.append(attn_prob_i)

    # Evaluation loop
    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    tower_mems_id_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)

        saver.restore(sess, eval_ckpt_path)

        if input_txt == "":
            txt_gen = tmp_Vocab.get_sym(random.randint(3, len(tmp_Vocab.idx2sym) - 1))
        else:
            txt_gen = input_txt

        fetches = [tower_new_mems,
                   tower_output,
                   tower_new_mems_id,
                   tower_attn_prob,
                   'transformer/adaptive_embed/lookup_table:0']

        encoded_input = tmp_Vocab.encode_sents(txt_gen, ordered=True)

        progress = ProgressBar()
        for _ in progress(range(FLAGS.gen_len)):
            time.sleep(0.01)
            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

                for id, id_np in zip(tower_mems_id[i], tower_mems_id_np[i]):
                    feed_dict[id] = id_np

            sess.run(iterator.initializer, feed_dict={test_list: [encoded_input]})
            fetched = sess.run(fetches, feed_dict=feed_dict)

            tower_mems_np, output = fetched[:2]

            tower_mems_id_np = fetched[2]

            tmp_list = output[0][-1][0]
            tmp_list = tmp_list.tolist()

            index = top_one_result(tmp_list)

            txt_gen += tmp_Vocab.get_sym(index)
            if tmp_Vocab.get_sym(index) == "<eos>":
                break
            else:
                encoded_input = [index]

        return txt_gen


def sent_ppl(input_txt_list, n_token, cutoffs, ps_device):

    test_list = tf.placeholder(tf.int64, shape=[1, None])
    dataset = tf.data.Dataset.from_tensors(test_list)
    # dataset = dataset.batch(1, drop_remainder=True)

    iterator = dataset.make_initializable_iterator()
    input_feed = iterator.get_next()

    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)

    per_core_bsz = 1
    tower_mems, tower_losses, tower_new_mems = [], [], []
    tower_output = []
    tower_mems_id = []
    tower_new_mems_id = []
    tower_attn_prob = []

    for i in range(FLAGS.num_core_per_host):
        with tf.device(assign_to_gpu(i, ps_device)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            mems_i = [tf.placeholder(tf.float32,
                                     [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]

            mems_i_id = [tf.placeholder(tf.int64,
                                        [FLAGS.mem_len, per_core_bsz])
                         for _ in range(FLAGS.n_layer)]

            new_mems_i, output_i, new_mems_i_id, attn_prob_i = single_core_graph_for_inference(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=False,
                inp=inputs[i],
                mems=mems_i,
                mems_id=mems_i_id)

            tower_mems.append(mems_i)
            tower_new_mems.append(new_mems_i)
            tower_output.append(output_i)
            tower_mems_id.append(mems_i_id)
            tower_new_mems_id.append(new_mems_i_id)
            tower_attn_prob.append(attn_prob_i)

    # Evaluation loop
    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    tower_mems_id_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    saver = tf.train.Saver()

    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True  # 按需分配内存
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 限制单进程只能占用GPU显存一定比例
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())

        eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)

        saver.restore(sess, eval_ckpt_path)

        fetches = [tower_new_mems,
                    tower_output,
                    tower_new_mems_id,
                    tower_attn_prob,
                    'transformer/adaptive_embed/lookup_table:0']

        sent_ppl_list = []

        def _cal_ppl(log_prob, sent_len):
            ppl = pow(math.exp((-1)*log_prob), 1/(sent_len-1))

            return ppl

        for i in range(len(input_txt_list)):
            #tf.logging.info('#time: {}'.format(time.time()))
            input_txt = input_txt_list[i]

            tower_mems_np = [
                [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
                 for layer in range(FLAGS.n_layer)]
                for core in range(FLAGS.num_core_per_host)
            ]

            tower_mems_id_np = [
                [np.zeros([FLAGS.mem_len, per_core_bsz], dtype=np.float32)
                 for layer in range(FLAGS.n_layer)]
                for core in range(FLAGS.num_core_per_host)
            ]

            #print("Encoded Input:", encoded_input)

            log_prob = 0

            for token in range(1, len(input_txt)):
                tf.logging.info('#time: {}'.format(time.time()))
                feed_dict = {}
                for i in range(FLAGS.num_core_per_host):
                    for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                        feed_dict[m] = m_np

                    for id, id_np in zip(tower_mems_id[i], tower_mems_id_np[i]):
                        feed_dict[id] = id_np

                sess.run(iterator.initializer, feed_dict={test_list: [[input_txt[token-1]]]})
                fetched = sess.run(fetches, feed_dict=feed_dict)

                tower_mems_np, output = fetched[:2]

                tower_mems_id_np = fetched[2]

                tmp_list = output[0][-1][0]
                tmp_list = tmp_list.tolist()

                e_sum = sum([math.exp(i) for i in tmp_list])
                log_prob_list = [math.log(math.exp(i)) - math.log(e_sum) for i in tmp_list]

                log_prob = log_prob + log_prob_list[input_txt[token]]
            
            sent_ppl_list.append(_cal_ppl(log_prob, len(input_txt)))
        
        return sent_ppl_list
                        

def single_core_graph_for_inference(n_token, cutoffs, is_training, inp, mems, mems_id):
    model_fn = get_model_fn_for_inference(
        n_token=n_token,
        cutoffs=cutoffs)

    model_ret = model_fn(
        inp=inp,
        mems=mems,
        mems_id=mems_id,
        is_training=is_training)

    return model_ret


def get_model_fn_for_inference(n_token, cutoffs):
    def model_fn(inp, mems, mems_id, is_training):
        inp = tf.transpose(inp, [1, 0])

        if FLAGS.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range,
                maxval=FLAGS.init_range,
                seed=None)
        elif FLAGS.init == "normal":
            initializer = tf.initializers.random_normal(
                stddev=FLAGS.init_std,
                seed=None)
            proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std,
                seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]
        if FLAGS.proj_share_all_but_first:
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True
        new_mems, output, new_mems_id, attn_prob = model.transformer_inference(
            dec_inp=inp,
            mems=mems,
            mems_id=mems_id,
            n_token=n_token,
            n_layer=FLAGS.n_layer,
            d_model=FLAGS.d_model,
            d_embed=FLAGS.d_embed,
            n_head=FLAGS.n_head,
            d_head=FLAGS.d_head,
            d_inner=FLAGS.d_inner,
            dropout=FLAGS.dropout,
            dropatt=FLAGS.dropatt,
            initializer=initializer,
            proj_initializer=proj_initializer,
            is_training=is_training,
            mem_len=FLAGS.mem_len,
            cutoffs=cutoffs,
            div_val=FLAGS.div_val,
            tie_projs=tie_projs,
            input_perms=None,
            target_perms=None,
            head_target=None,
            same_length=FLAGS.same_length,
            clamp_len=FLAGS.clamp_len,
            use_tpu=False,
            untie_r=FLAGS.untie_r,
            proj_same_dim=FLAGS.proj_same_dim)

        # number of parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        return new_mems, output, new_mems_id, attn_prob

    return model_fn

def cut_pad(num_list, r_len, pad_value=0):
    if len(num_list) <= r_len:
        return num_list
    else:
        return num_list[:r_len]

def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
    n_token = corpus_info["vocab_size"]
    cutoffs = corpus_info["cutoffs"][1:-1]
    tf.logging.info("n_token {}".format(n_token))

    tmp_Vocab = Vocab(special=["<bos>", "<eos>", "<UNK>"])
    tmp_Vocab.count_file("../data/{}/train.txt".format(FLAGS.dataset), add_eos=False)
    tmp_Vocab.build_vocab()

    if FLAGS.do_sent_ppl_pred:
        encoded_txt_input = []
        txt_input = []
        input_csv = []
        with open(FLAGS.input_file_dir, "r") as read_file:
            csv_reader = csv.reader(read_file)
            for line in csv_reader:
                if line[0].strip() != 0:
                    input_csv.append(line)
            
            for i in range(1, len(input_csv)):
                txt_input.append(input_csv[i][0].strip())
                encoded_txt_input.append(list(tmp_Vocab.encode_sents(input_csv[i][0].strip(), \
                    add_eos=True, ordered=True)))

        encoded_txt_input = [line[:FLAGS.limit_len] if len(line) > FLAGS.limit_len else line for line in encoded_txt_input]
        encoded_txt_input = np.array(encoded_txt_input)

        input_csv[0].append("ppl")

        pool = multiprocessing.Pool(FLAGS.multiprocess)
        
        parti_len = len(encoded_txt_input)//FLAGS.multiprocess
        pro_res_l = []

        for i in range(FLAGS.multiprocess):
            print("Setting process-%s" % i)
            ### 有空这里要写一个控制使用gpu:xx的步骤(gpu:1满了就用下一个)

            if i+1 == FLAGS.multiprocess:
                end = len(encoded_txt_input)
            else:
                end = (i+1)*parti_len
            pro_res_l.append(pool.apply_async(sent_ppl, \
                args=(encoded_txt_input[i*parti_len:end], n_token, cutoffs, "/gpu:1")))
            
        res_l = []

        for i in range(len(pro_res_l)):
            proc_i_res = pro_res_l[i].get()
            res_l.extend(proc_i_res)

        pool.close()
        pool.join()
        print('All subprocesses done.')

        tf.logging.info('#time: {}'.format(time.time()))

        for i in range(1, len(input_csv)):
            input_csv[i].append(res_l[i-1])
        output_df = pd.DataFrame(input_csv[1:], columns=input_csv[0])
        output_df.to_csv(FLAGS.output_file_dir, sep=",", index=False, encoding="utf-8-sig")

        with open("non_batch_ref_output.txt", "w") as write_res:
            for i in range(len(txt_input)):
                write_res.write(txt_input[i] + " " + str(encoded_txt_input[i]) + " " + str(res_l[i]) + "\n")
        
        # Check whether the length of result is right; Make sure multiprocess work well
        print(len(res_l))

    elif FLAGS.do_sent_gen:
        txt_gen_list = []
        with open(FLAGS.input_txt_dir, "r") as read_txt:
            for input_txt in read_txt:
                if len(input_txt.strip()) != 0:
                    txt_gen_list.append(sent_gen(tmp_Vocab, input_txt.strip(), n_token, cutoffs, "/gpu:1"))
        
        with open("sent_generation.txt", "w") as write_res:
            for line in txt_gen_list:
                write_res.write(line + "\n")

if __name__ == "__main__":
    tf.app.run()
