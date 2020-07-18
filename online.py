#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  :  Li Yan

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import os
import importlib
from tensorflow.python.util import nest
import numpy as np

try:
  trt = importlib.import_module("tensorflow.contrib.tensorrt")
except:
  trt = None

from nltool.helper import pad_multi_lines

from aitranslation.backend.utils.tensorflow import *
from aitranslation.config import *
from model_config import *
from tensor2tensorextend.utils.text_encoder import SPMTextEncoder

class Ctriptranslator(TranslatorBase):
  def __init__(self, deploy_hparam, model_config):
    super(Ctriptranslator, self).__init__(deploy_hparam, model_config)

  def init(self, deploy_hparam, model_config):
      self.is_deploy = deploy_hparam.get('is_deploy') or False
      self.is_gpu = deploy_hparam.get('is_gpu') or False
      self.model_config = model_config
      self.checkpoint_dir = os.path.expanduser(
        os.path.join(deploy_hparam.get('tpath') or "./translation_model", self.model_config.checkpoint_dir))
      print("**checkpoint****: ", self.checkpoint_dir)
      self.vocab_path = os.path.join(self.checkpoint_dir, self.model_config.vocab_path)
      self.langs = self.model_config.langs
      self.encoders = SPMTextEncoder(self.vocab_path)
      logger.info("{} - Initializing...".format(self.info))
      self.warmuped = False
      logger.info("{} - Building...".format(self.info))
      self.ckpt_path = tf.train.latest_checkpoint(self.checkpoint_dir)
      logger.info("{} - Checkpoint:{}".format(self.info, self.ckpt_path))
      self.build()
      logger.info("{} - Initialization Done...".format(self.info))

  def build(self):
    '''
    Build Model
    :return: None
    '''
    # Get Hyper Parameters by Name
    device = "" if not self.is_gpu else "GPU"
    logger.info("{} - Using device: {}".format(self.info, device))
    with tf.Graph().as_default() as graph:
      with tf.device(device):
        logger.info("{} - Building Input Placeholders...".format(self.info))
        self.graph = graph
        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        source_lang = tf.placeholder(tf.int32, [None], "source_lang")
        target_lang = tf.placeholder(tf.int32, [None], "target_lang")
        task_space = tf.placeholder(tf.int32, [None, None], "task_space")
        beam_size = tf.placeholder(tf.int32, (), "beam_size")
        alpha = tf.placeholder(tf.float32, [], "alpha")
        beta = tf.placeholder(tf.float32, [], "beta")
        decode_length_scale = tf.placeholder(tf.float32, [], 'decode_length_scale')
        max_decode_length = tf.placeholder(tf.int32, [], 'max_decode_length')
        self.translate_inputs = TFTranslateInput(inputs, decode_length_scale, max_decode_length,
                                                 source_lang, target_lang, task_space, beam_size, alpha, beta)

        def _build_greedy_model(inputs, source_lang, target_lang, task_space):
          tf.get_variable_scope()._reuse = tf.AUTO_REUSE
          features = {}
          features["target_lang_id"] = target_lang
          features["inputs"] = inputs
          features["targets"] = inputs
          translate_batch_size = tf.shape(inputs)[0]
          inputs_mask = tf.cast(tf.not_equal(inputs, tf.constant(self.model_config.PAD, tf.int32)), tf.float32)
          decode_length = tf.reduce_sum(inputs_mask, 1)
          translate_dynamic_decode_length = tf.cast(decode_length * decode_length_scale, tf.int32)
          translate_max_decode_length = tf.ones([translate_batch_size], tf.int32) * max_decode_length
          translate_decode_length = tf.where(translate_dynamic_decode_length < translate_max_decode_length,
                                             translate_dynamic_decode_length, translate_max_decode_length,
                                             name="final_greedy_max_decode_length")
          # Greedy Graph
          logger.info("{} - Building Greedy Translation Graph...".format(self.info))
          translate_outputs = self.model_infer(features,translate_decode_length)
          return translate_outputs


        output = _build_greedy_model(inputs, source_lang, target_lang,task_space)
        

        self.session()

        self.translate_outputs =  output
        print(output)
                                      

        
  def session(self, graph=True):
    logger.info("{} - Building Session...".format(self.info))
    config = {}
    self.sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, **config)
    #self.sess_config.gpu_options.allow_growth = True
    self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95
    self.sess = tf.Session(config=self.sess_config, graph=self.graph if graph else None)
    if self.model_config.mode == Mode.Ckpt:
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())
      # Restore Parameters in Model
      self.saver = tf.train.Saver()
      self.saver.restore(self.sess, self.ckpt_path)
    logger.info("{} - Building Session Done...".format(self.info))

  @property
  def info(self):
    return "Backend:Tensor2tensor Model:{}...".format("transformer")

  def translate(self, inputs, lang1, lang2, task_space, max_decode_length=None, decode_scale=None, beam_size=None,
                alpha=0., beta=0.):
    '''
    Translate the input sentence
    '''
    try:
      lang1 = [self.model_config.langs.index(l) for l in lang1]
      lang2 = [self.model_config.langs.index(l) for l in lang2]
      inputs_idxs = [self.encoders.encode(input_) + [self.model_config.EOS] for input_ in inputs]
      inputs_lists = [self.encoders.decode_list(encoded_input_idxs, True) for encoded_input_idxs in inputs_idxs]

      padded_inputs_idxs = pad_multi_lines(inputs_idxs, self.model_config.PAD)
      #print("input: ", padded_inputs_idxs)
    except:
      return [
        TranslationCandResult([], inp, None, None, None, None,
                              TranslationSetting(beam_size, alpha if beam_size else None,
                                                 beta if beam_size else None, None), None, 0.,
                              Status.report(Status.BackendError, "Translator.translate Prepare Error"))
        for inp, ts in zip(inputs, task_space)
      ]
    try:
      t0 = time.time()
      options = tf.RunOptions()
      if self.warmuped:
        options.timeout_in_ms = switch.get_back_switch("timeout", None, 15000)
      tf_result = self.sess.run(self.translate_outputs,#.outputs_dict(beam_size),
                                feed_dict=self.translate_inputs.feed_dict(
                                  padded_inputs_idxs,
                                  lang1, lang2, task_space, max_decode_length,
                                  decode_scale, beam_size, alpha, beta), options=options)
      tf_result = np.transpose(tf_result)
      elapsed = time.time() - t0
    except tf.errors.DeadlineExceededError:
      return [
        TranslationCandResult([], inp, None, inp_l, None, None,
                              TranslationSetting(beam_size, alpha if beam_size else None,
                                                 beta if beam_size else None, None), None, 0.,
                              Status.report(Status.TimeoutError, "Translator.translate Session Timeout"))
        for inp, inp_l, ts in zip(inputs, inputs_lists, task_space)
      ]
    except:
      return [
        TranslationCandResult([], inp, None, inp_l, None, None,
                              TranslationSetting(beam_size, alpha if beam_size else None,
                                                 beta if beam_size else None, None), None, 0.,
                              Status.report(Status.BackendError, "Translator.translate Session Run Error"))
        for inp, inp_l, ts in zip(inputs, inputs_lists, task_space)
      ]
    try:
      targets_idxs = [tf_result]
      #print(targets_idxs)
      targsts = []
      targsts_lists = []
      for targets_beam_idxs in targets_idxs:
        targsts_beam = []
        targsts_lists_beam = []
        for target_idx in targets_beam_idxs:
          target = self.encoders.decode(target_idx, True)
          target_list = self.encoders.decode_list(target_idx, True)
          if len(target_list) == 0 or target == 0: continue
          targsts_beam.append(target)
          targsts_lists_beam.append(target_list)
        if len(targsts_beam) == 0:
          targsts_beam = [""]
          targsts_lists_beam = [[]]
        targsts.append(targsts_beam)
        targsts_lists.append(targsts_lists_beam)
      #print(targsts)
      return targsts
    except:
      return [
        TranslationCandResult([], inp, None, inp_l, None, None,
                              TranslationSetting(beam_size, alpha if beam_size else None,
                                                 beta if beam_size else None, None), None, 0.,
                              Status.report(Status.BackendError, "Translator.translate Result Extraction Error"))
        for inp, inp_l, ts in zip(inputs, inputs_lists, task_space)
      ]
    return [
      TranslationCandResult([], inp, trgs, inp_l, trgs_l, score,
                            TranslationSetting(beam_size, alpha if beam_size else None, beta if beam_size else None,
                                               ml), None, elapsed)
      for inp, inp_l, trgs, trgs_l, score, ml, ts in
      zip(inputs, inputs_lists, targsts, targsts_lists, scores, max_length, task_space)
    ]

  def model_infer(self, features, translate_decode_length):
    def transformer_prepare_encoder(features):
      encoder_input = features["inputs_emb"]
      # get bias
      encoder_padding = tf.to_float(tf.equal(tf.reduce_sum(tf.abs(encoder_input), axis=-1), 0.0))
      ignore_padding = tf.expand_dims(tf.expand_dims(encoder_padding * -1e9, axis=1), axis=1)
      encoder_self_attention_bias = ignore_padding
      encoder_decoder_attention_bias = ignore_padding

      # add target_lang_emb
      ishape_static = encoder_input.shape.as_list()
      target_lang_emb = embedding(features["target_lang_id"], len(self.langs), ishape_static[-1],
                                  name="body/encoder_target_lang_embedding")
      target_lang_emb = tf.expand_dims(target_lang_emb, 1)
      encoder_input += target_lang_emb

      return encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias

    def ffn(x, num_units, filter_size, nonpad_ids=None,dim_origin=None):
      original_shape = shape_list(x)
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      if nonpad_ids is not None:
        x = tf.expand_dims(remove(x,nonpad_ids), axis=0)

      x = dense(x, filter_size, use_bias=True, activation=tf.nn.relu, name="conv1")
      output = dense(x,num_units,activation=None,use_bias=True, name="conv2")

      if nonpad_ids is not None:
        output = tf.reshape(restore(tf.squeeze(output, axis=0),nonpad_ids,dim_origin), original_shape)
      output = tf.reshape(output, original_shape)
      return output

    def encode(features):
      encoder_input, self_attention_bias, encoder_decoder_attention_bias = transformer_prepare_encoder(features)
      x = encoder_input
      attention_bias = self_attention_bias
      padding = tf.squeeze(tf.to_float(tf.less(attention_bias, -1)), axis=[1, 2])
      nonpad_ids,dim_origin = padremover(padding)
      with tf.variable_scope("body/encoder"):
        for layer in range(self.model_config.num_layers):
          with tf.variable_scope("layer_%d" % layer):
            with tf.variable_scope("self_attention"):
              y = multihead_attention(
                layer_preprocess(x),
                None,
                self_attention_bias,
                self.model_config.num_units,
                self.model_config.num_units,
                self.model_config.num_units,
                self.model_config.num_heads,
                attention_type="dot_product_relative",
                max_relative_position=self.model_config.max_relative_position,
              )
              x = layer_postprocess(x, y)
            with tf.variable_scope("ffn"):
              y = ffn(
                layer_preprocess(x),
                self.model_config.num_units,
                self.model_config.filter_size,
                nonpad_ids=nonpad_ids,
                dim_origin=dim_origin
              )
              x = layer_postprocess(x, y)
        
        return layer_preprocess(x), encoder_decoder_attention_bias

    def decode(language_id, encoder_output, encoder_decoder_attention_bias):
      units = 1024
      max_length_decode = 256;
      para = [[] for i in range(self.model_config.num_layers)]
      with tf.variable_scope("body/decoder"):
        for layer in range(self.model_config.num_layers):
          layer_name = "layer_%d" % layer
          with tf.variable_scope(layer_name):
            with tf.variable_scope("self_attention"):
              self_scale = tf.get_variable("layer_prepostprocess/layer_norm/layer_norm_scale", [units])
              para[layer].append(self_scale)
              self_bias = tf.get_variable("layer_prepostprocess/layer_norm/layer_norm_bias", [units])
              para[layer].append(self_bias)
              self_q = tf.get_variable("multihead_attention/q/kernel", [units, units])
              para[layer].append(self_q)
              self_k = tf.get_variable("multihead_attention/k/kernel", [units, units])
              para[layer].append(self_k)
              self_v = tf.get_variable("multihead_attention/v/kernel", [units, units])
              para[layer].append(self_v)
              self_last = tf.get_variable("multihead_attention/output_transform/kernel", [units, units])
              para[layer].append(self_last)
              self_position_key = tf.get_variable("multihead_attention/dot_product_attention_relative/relative_positions_keys/embeddings", [41,64])
              para[layer].append(self_position_key)
              self_position_value = tf.get_variable("multihead_attention/dot_product_attention_relative/relative_positions_values/embeddings", [41,64])
              para[layer].append(self_position_value)

            with tf.variable_scope("encdec_attention"):
              encdec_scale = tf.get_variable("layer_prepostprocess/layer_norm/layer_norm_scale", [units])
              para[layer].append(encdec_scale)
              encdec_bias = tf.get_variable("layer_prepostprocess/layer_norm/layer_norm_bias", [units])
              para[layer].append(encdec_bias)
              encdec_q = tf.get_variable("multihead_attention/q/kernel", [units, units])
              para[layer].append(encdec_q)
              encdec_k = tf.get_variable("multihead_attention/k/kernel", [units, units])
              para[layer].append(encdec_k)
              encdec_v = tf.get_variable("multihead_attention/v/kernel", [units, units])
              para[layer].append(encdec_v)
              encdec_last = tf.get_variable("multihead_attention/output_transform/kernel", [units, units])
              para[layer].append(encdec_last)
              para[layer].append(encoder_decoder_attention_bias)

            with tf.variable_scope("ffn"):
              ffn_scale = tf.get_variable("layer_prepostprocess/layer_norm/layer_norm_scale", [units])
              para[layer].append(ffn_scale)
              ffn_bias = tf.get_variable("layer_prepostprocess/layer_norm/layer_norm_bias", [units])
              para[layer].append(ffn_bias)
              first_weight = tf.get_variable("conv1/kernel", [units, 4*units])
              para[layer].append(first_weight)
              first_bias = tf.get_variable("conv1/bias", [4*units])
              para[layer].append(first_bias)
              second_weight = tf.get_variable("conv2/kernel", [4*units, units])
              para[layer].append(second_weight)
              second_bias = tf.get_variable("conv2/bias", [units])
              para[layer].append(second_bias)

      l_embedding = tf.get_variable("body/decoder_target_lang_embedding/kernel", [4, units])
      w_embedding = tf.get_variable("symbol_modality_32768_1024/shared/weights_0", [32768, units])
      logit = tf.get_variable("symbol_modality_32768_1024/softmax/weights_0", [32768, units])
      scale = tf.get_variable("body/decoder/layer_prepostprocess/layer_norm/layer_norm_scale", [units])
      bias = tf.get_variable("body/decoder/layer_prepostprocess/layer_norm/layer_norm_bias", [units])
    
      my_tf = tf.load_op_library('./src/decoding.so')
       
      for i in range(6):
        for j in range(21):
          para[i][j] = tf.cast(para[i][j], dtype=tf.float16)
      encoder_output = tf.cast(encoder_output, dtype=tf.float16)
      l_embedding = tf.cast(l_embedding, dtype=tf.float16)
      w_embedding = tf.cast(w_embedding, dtype=tf.float16)
      logit = tf.cast(logit, dtype=tf.float16)
      scale = tf.cast(scale, dtype=tf.float16)
      bias = tf.cast(bias, dtype=tf.float16)
      
      output = my_tf.decoding(encoder_output, language_id, max_length_decode,
                              para[0][0],  para[0][1],  para[0][2],  para[0][3],  para[0][4],  para[0][5],  para[0][6],  para[0][7],  para[0][8],  para[0][9],
                              para[0][10], para[0][11], para[0][12], para[0][13], para[0][14], para[0][15], para[0][16], para[0][17], para[0][18], para[0][19], para[0][20],
                              para[1][0],  para[1][1],  para[1][2],  para[1][3],  para[1][4],  para[1][5],  para[1][6],  para[1][7],  para[1][8],  para[1][9],
                              para[1][10], para[1][11], para[1][12], para[1][13], para[1][14], para[1][15], para[1][16], para[1][17], para[1][18], para[1][19], para[1][20],
                              para[2][0],  para[2][1],  para[2][2],  para[2][3],  para[2][4],  para[2][5],  para[2][6],  para[2][7],  para[2][8],  para[2][9],
                              para[2][10], para[2][11], para[2][12], para[2][13], para[2][14], para[2][15], para[2][16], para[2][17], para[2][18], para[2][19], para[2][20],
                              para[3][0],  para[3][1],  para[3][2],  para[3][3],  para[3][4],  para[3][5],  para[3][6],  para[3][7],  para[3][8],  para[3][9],
                              para[3][10], para[3][11], para[3][12], para[3][13], para[3][14], para[3][15], para[3][16], para[3][17], para[3][18], para[3][19], para[3][20],
                              para[4][0],  para[4][1],  para[4][2],  para[4][3],  para[4][4],  para[4][5],  para[4][6],  para[4][7],  para[4][8],  para[4][9],
                              para[4][10], para[4][11], para[4][12], para[4][13], para[4][14], para[4][15], para[4][16], para[4][17], para[4][18], para[4][19], para[4][20],
                              para[5][0],  para[5][1],  para[5][2],  para[5][3],  para[5][4],  para[5][5],  para[5][6],  para[5][7],  para[5][8],  para[5][9],
                              para[5][10], para[5][11], para[5][12], para[5][13], para[5][14], para[5][15], para[5][16], para[5][17], para[5][18], para[5][19], para[5][20],
                              l_embedding, w_embedding, logit, scale, bias, 
                              head_num=16, size_per_head=64, num_layer=6, memory_hidden_dim=1024, vocab_size=32768, end_id=2)
      return output 

    def _get_vocab_var(name):
      return tf.get_variable(name + "/weights_0",
                             [self.model_config.vocabe_size, self.model_config.num_units],
                             trainable=True, dtype=tf.float32,custom_getter=None)

    def infer(features, translate_decode_length):
      with tf.variable_scope("transformer", dtype=self.model_config.dtype):
        wrapper = (lambda fn: fn)
        @wrapper
        def _build(features):
          inputs = features["inputs"]
          batch_size = tf.shape(inputs)[0]
          bottom(features)
          encoder_output, encoder_decoder_attention_bias = encode(features)
          output = decode(features["target_lang_id"],encoder_output, encoder_decoder_attention_bias)
          return output
        out = _build(features)
      return out
          



    def _bottom_encoder(features):
      with tf.device("cpu"):
        emb = _get_vocab_var("shared")
        y = tf.gather(emb, features["inputs"])
      y *= self.model_config.num_units ** 0.5
      return y

    def _bottom_decoder(features):
      with tf.device("cpu"):
        emb = _get_vocab_var("shared")
        y = tf.gather(emb, features["targets"])
      y *= self.model_config.num_units ** 0.5
      return y

    def bottom(features):
      with tf.variable_scope("symbol_modality_32768_1024", reuse=tf.AUTO_REUSE):
        features["inputs_emb"] = _bottom_encoder(features)
        features["targets_emb"] = _bottom_decoder(features)

    def top(body_output):
      with tf.variable_scope("symbol_modality_32768_1024", reuse=tf.AUTO_REUSE):
        weight = _get_vocab_var("softmax")
        logits = tf.tensordot(body_output, weight, [[2], [1]], name="logits")
        logits = tf.squeeze(logits, axis=1)
      return logits

    def dense(inputs, units, activation=None, use_bias=True, trainable=True, name=None):
      with tf.variable_scope(name):
        weight = tf.get_variable("kernel", [inputs.shape[-1], units], trainable=trainable)
        output = tf.tensordot(inputs, weight, [[-1], [0]])
        if use_bias:
          bias = tf.get_variable("bias", [units], trainable=trainable)
          output = tf.nn.bias_add(output, bias)
        if activation is not None:
          output = activation(output)
        return output

    def shape_list(x):
      x = tf.convert_to_tensor(x)

      # If unknown rank, return dynamic shape
      if x.get_shape().dims is None:
        return tf.shape(x)

      static = x.get_shape().as_list()
      shape = tf.shape(x)

      ret = []
      for i, dim in enumerate(static):
        if dim is None:
          dim = shape[i]
        ret.append(dim)
      return ret

    def embedding(x, vocab_size, dense_size, name=None, reuse=None, dtype=tf.float32):
      with tf.variable_scope(name, default_name="embedding", values=[x], reuse=reuse, dtype=dtype):
        embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        emb_x = tf.gather(embedding_var, x, dtype)
        return emb_x

    def attention_bias_lower_triangle(length):
      out_shape = [1, 1, length, length]
      band = tf.matrix_band_part(
        tf.ones([length, length]), tf.cast(-1, tf.int64),
        tf.cast(0, tf.int64))
      band = tf.reshape(band, out_shape)
      return -1e9 * (1.0 - band)

    def layer_preprocess(layer_input):
      with tf.variable_scope("layer_prepostprocess"):
        with tf.variable_scope("layer_norm"):
          num_units = layer_input.shape[-1]
          scale = tf.get_variable(
            "layer_norm_scale", [num_units], initializer=tf.ones_initializer(), trainable=True)
          bias = tf.get_variable(
            "layer_norm_bias", [num_units], initializer=tf.zeros_initializer(), trainable=True)
          x = layer_input
          epsilon, scale, bias = [tf.cast(t, x.dtype) for t in [tf.constant(1e-06), scale, bias]]
          mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
          variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
          norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
          return norm_x * scale + bias

    def layer_postprocess(layer_input, layer_output):
      with tf.variable_scope("layer_postprocess"):
        x = layer_input + layer_output
        return x

    def split_heads(x, num_heads):
      x_shape = shape_list(x)
      m = x_shape[-1]
      y = tf.reshape(x, tf.concat([x_shape[:-1], [num_heads, m // num_heads]], 0))
      return tf.transpose(y, [0, 2, 1, 3])

    def dot_product_attention(q, k, v, bias, name=None):
      with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
          logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        return tf.matmul(weights, v)

    def compute_qkv(q_a, m_a, kd, vd):
      if m_a is None: m_a = q_a
      q = dense(q_a, kd, use_bias=False,name="q")
      k = dense(m_a, kd, use_bias=False,name="k")
      v = dense(m_a, vd, use_bias=False,name="v")
      return q, k, v

    def dot_product_attention_relative(q, k, v, bias, max_relative_position, name=None, cache=False):
      with tf.variable_scope(name, default_name="dot_product_attention_relative", values=[q, k, v]):
        if not cache:
          q.get_shape().assert_is_compatible_with(k.get_shape())
          q.get_shape().assert_is_compatible_with(v.get_shape())

        depth = k.get_shape().as_list()[3]
        length = shape_list(k)[2]
        relations_keys = _generate_relative_positions_embeddings(
          length, depth, max_relative_position, "relative_positions_keys",
          cache=cache)
        relations_values = _generate_relative_positions_embeddings(
          length, depth, max_relative_position, "relative_positions_values",
          cache=cache)
     
        logits = _relative_attention_inner(q, k, relations_keys, True)
        if bias is not None:
          logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        return _relative_attention_inner(weights, v, relations_values, False)

    def _relative_attention_inner(x, y, z, transpose):
      # yz_add = tf.add(y,z)
      # x_yz_add_matmul = tf.matmul(x,yz_add, transpose_b=True)
      # return x_yz_add_matmul 
      batch_size = tf.shape(x)[0]
      heads = x.get_shape().as_list()[1]
      length = tf.shape(x)[2]
      # xy_matmul is [batch_size, heads, length or 1, length or depth]
      xy_matmul = tf.matmul(x, y, transpose_b=transpose)
      # x_t is [length or 1, batch_size, heads, length or depth]
      x_t = tf.transpose(x, [2, 0, 1, 3])
      # x_t_r is [length or 1, batch_size * heads, length or depth]
      x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
      # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
      x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
      # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
      x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
      # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
      x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
      return xy_matmul + x_tz_matmul_r_t

    def _generate_relative_positions_embeddings(length, depth,
                                                max_relative_position, name,
                                                cache=False):
      with tf.variable_scope(name):
        relative_positions_matrix = _generate_relative_positions_matrix(
          length, max_relative_position, cache=cache)
        vocab_size = max_relative_position * 2 + 1
        # Generates embedding for each relative position of dimension depth.
        embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings

    def _generate_relative_positions_matrix(length, max_relative_position,
                                            cache=False):
      if not cache:
        range_vec = tf.range(length)
        range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
        distance_mat = range_mat - tf.transpose(range_mat)
      else:
        distance_mat = tf.expand_dims(tf.range(-length + 1, 1, 1), 0)
      distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                              max_relative_position)
      final_mat = distance_mat_clipped + max_relative_position
      return final_mat

    def multihead_attention(q_a, m_a, bias, kd, vd, od, num_heads, attention_type="dot_product",
                            max_relative_position=None, cache=None):
      with tf.variable_scope("multihead_attention"):
        if cache is None:
          q, k, v = compute_qkv(q_a, m_a, kd, vd)
          q = split_heads(q, num_heads)
          k = split_heads(k, num_heads)
          v = split_heads(v, num_heads)
        else:
          if m_a is None:
            q, k, v = compute_qkv(q_a, m_a, kd, vd)
            q = split_heads(q, num_heads)
            k = split_heads(k, num_heads)
            v = split_heads(v, num_heads)
            k = cache["k"] = tf.concat([cache["k"], k], axis=2)
            v = cache["v"] = tf.concat([cache["v"], v], axis=2)
          else:
            q = dense(q_a, kd, use_bias=False, name="q")
            q = split_heads(q, num_heads)
            k = cache["k_encdec"]
            v = cache["v_encdec"]
        key_depth_per_head = kd // num_heads
        q *= key_depth_per_head ** -0.5
        if attention_type == "dot_product":
          x = dot_product_attention(q, k, v, bias)
        elif attention_type == "dot_product_relative":
          x = dot_product_attention_relative(q, k, v, bias, max_relative_position, cache=cache is not None)

        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        a, b = x_shape[-2:]
        x = tf.reshape(x, x_shape[:-2] + [a * b])
        x.set_shape(x.shape.as_list()[:-1] + [vd])
        x = dense(x, od, use_bias=False, name="output_transform")
        return x

    def padremover(pad_mask):
      with tf.name_scope("pad_reduce/get_ids"):
        pad_mask = tf.reshape(pad_mask, [-1])
        nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
        dim_origin = tf.shape(pad_mask)[:1]
      return nonpad_ids,dim_origin

    def remove(x,nonpad_ids):
      with tf.name_scope("pad_reduce/remove"):
        x_shape = x.get_shape().as_list()
        x = tf.gather_nd(x, indices=nonpad_ids,)
      if not tf.executing_eagerly():
        x.set_shape([None] + x_shape[1:])
      return x

    def restore(x,nonpad_ids,dim_origin):
      with tf.name_scope("pad_reduce/restore"):
        x = tf.scatter_nd(indices=nonpad_ids,updates=x,shape=tf.concat([dim_origin, tf.shape(x)[1:]], axis=0))
      return x

    return infer(features, translate_decode_length)

if __name__ == "__main__":
  #t = Ctriptranslator({'tpath': '/opt/app/translation_model', 'ldpath': '/opt/app/language_detection_model', 'is_deploy': False, 'is_gpu': False},ModelSpecEnJaKoZh())
  t = Ctriptranslator({'tpath': '/opt/app/translation_model', 'ldpath': '/opt/app/language_detection_model', 'is_deploy': False, 'is_gpu': True},ModelSpecEnJaKoZh())
  lang1=["zh"]
  lang2=["en"]
  input0= ['call out', 'hello world, i am from china']#,'警察直升机在各个区域上方飞行了大约一个小时-没有成功。']
  input1= ['我', '警察直升机在各个区域上方飞行了大约一个小时-没有成功。']
  input2 = ['打人','东南西北','自以为是的美国总统其实是一个傻子','警察直升机在各个区域上方飞行了大约一个小时-没有成功','CUDA中使用多个流并行执行数据复制和核函数运算可以进一步提高计算性能。以下程序使用2个流执行运算', '我们中国共产党是在一个几万万人的大民族中领导伟大革命斗争的党，没有多数才德兼备的领导干部，是不能完成其历史任务的。十七年来，我们党已经培养了不少的领导人材，这是党的光荣所在，也是全民族的光荣所在','前者是正派的路线，后者是不正派的路线。共产党的干部政策，应是以能否坚决地执行党的路线，不谋私利为标准，这就是“任人唯贤”的路线。过去张国焘的干部政策与此相反，结果叛党而去，这是一个大教训。鉴于张国焘的和类似张国焘的历史教训，在干部政策问题上坚持正派的公道的作风，反对不正派的不公道的作风，借以巩固党的统一团结，这是中央和各级领导者的重要的责任。','马克思、恩格斯的理论，是“放之四海而皆准”的理论。不应当把他们的理论当作教条看待，而应当看作行动的指南。不但应当了解马克思,研究广泛的真实的生活和广泛的经验所得出的关于一般规律的结论，而且应当学习他们观察问题和解决问题的立场和方法。我们党的马克思列宁主义的修养，现在已较过去有了一些进步，但是还很不普遍，很不深入。我们的任务，是领导一个几万万人口的大民族，进行空前的伟大的斗争。所以，普遍地深入地研究马克思列宁主义的理论的任务，对于我们，是一个亟待解决并须着重地致力才能解决的大问题。我希望从我们这次中央全会之后，来一个全党的学习竞赛，看谁真正地学到了一点东西，看谁学的更多一点，更好一点。在担负主要领导责任的观点上说，如果我们党有一百个至二百个系统地而不是零碎地、实际地而不是空洞地学会了马克思列宁主义的同志，就会大大地提高我们党的战斗力量，并加速我们战胜日本帝国主义的工作']
  st=time.time()
  #for i in range(1000):
    #result = t.translate(input1, lang1, lang2, [[0]], max_decode_length=256,
    #                   decode_scale=2)
  result = t.translate(input0, ['zh'], ['en','en'], [[0]], max_decode_length=256,decode_scale=2)
  print(result)
  print("total cost time",time.time()-st)
  st=time.time()
  batch = 512 
  for i in range(1):
      result = t.translate(input2*batch, lang1*batch, lang2*batch, [[0]]*batch, max_decode_length=256, decode_scale=2)
  print("total cost time",time.time()-st)
  print(result)
  #print(result)

