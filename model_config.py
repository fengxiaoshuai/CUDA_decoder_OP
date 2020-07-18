#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Li Yan

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from aitranslation.backend.utils.tensorflow import Mode

class ModelBase(object):
  def __init__(self):
    self.checkpoint_dir="translate_enzh_ctrip/transformer_relative_big_ctrip"
    self.vocab_path = "Vocab/spm_model_en_zh_32768.model"
    self.langs = ["en","zh"]
    self.decode_length_extra_scale = 3
    self.max_decode_length = 1000
    self.PAD = 0
    self.EOS = 2
    self.mode = Mode.Ckpt
    self.num_layers = 6
    self.num_units = 1024
    self.filter_size = 4096
    self.num_heads = 16
    self.vocabe_size = 32768
    self.beam_size = None
    self.decode_length = 17
    self.batchsize = 1
    self.dtype = tf.float32
    self.max_relative_position = 20

class ModelSpecEnZh(ModelBase):
  def __init__(self):
    super().__init__()
    self.checkpoint_dir = "translate_enzh_ctrip/transformer_relative_big_ctrip"
    self.vocab_path = "Vocab/spm_model_en_zh_32768.model"
    self.langs = ["en", "zh"]

class ModelSpecEnJaKoZh(ModelBase):
  def __init__(self):
    super().__init__()
    self.checkpoint_dir = "translate_enjakozh_ctrip/transformer_relative_big_ctrip"
    self.vocab_path = "Vocab/spm_model_en_ja_ko_zh_32768.model"
    self.langs = ["en","ja","ko", "zh"]
    self.qkv_ver = 1
    self.cache_embedding_on_cpu = True
    self.mix_precision_train = False
    self.branch_attention = False
    self.attention_type = "dot_product_relative"
    self.max_relative_position = 20
    self.profile = False
    self.batching = False
    self.batching_config = {
      "num_batch_threads": 1,
      "max_batch_size": 1,
      "batch_timeout_micros": 1,
      "allowed_batch_sizes": None,
      "max_enqueued_batches": 10,
    }

class ModelSpecArDeEnEsFrItPtThTrZh(ModelBase):
  def __init__(self):
    super().__init__()
    self.checkpoint_dir = "translate_enzhdefresptittrthar_ctrip/transformer_relative_big_ctrip"
    self.vocab_path = "Vocab/spm_model_ar_de_en_es_fr_it_pt_th_tr_zh_32768.model"
    self.langs = ["ar","de","en","es","fr","it","pt","th","tr","zh"]

class ModelSpecArEnRuZh(ModelBase):
  def __init__(self):
    super().__init__()
    self.checkpoint_dir = "translate_arenruzh_ctrip/transformer_relative_big_ctrip"
    self.vocab_path = "Vocab/spm_model_ar_en_ru_zh_32768.model"
    self.langs = ["ar", "en", "ru", "zh"]

class ModelSpecZhZhyue(ModelBase):
  def __init__(self):
    super().__init__()
    self.checkpoint_dir = "translate_zhzhyue_ctrip/transformer_base_ctrip"
    self.vocab_path = "Vocab/spm_model_zh_zhyue_65536.model"
    self.langs = ['zh', 'zhyue']
    self.vocabe_size=65536

