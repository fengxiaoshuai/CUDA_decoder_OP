/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "decoder.h"
#include "decoding_op.h"
#include "common_op.h"
#include <mutex>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include <cuda_fp16.h>
#include "cuda_runtime.h"
namespace tensorflow
{
namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Decoding")
    .Input("encode_out: T")
    .Input("language_id: int32")
    .Input("max_decode_length: int32")

    //layer_0
    .Input("in_0_self_gamma: T")
    .Input("in_0_self_beta: T")
    .Input("in_0_self_q_kernel: T")
    .Input("in_0_self_k_kernel: T")
    .Input("in_0_self_v_kernel: T")
    .Input("in_0_self_output_kernel: T")
    .Input("in_0_self_position_key: T")
    .Input("in_0_self_position_value: T")

    .Input("in_0_cross_gamma: T")
    .Input("in_0_cross_beta: T")
    .Input("in_0_cross_q_kernel: T")
    .Input("in_0_cross_k_kernel: T")
    .Input("in_0_cross_v_kernel: T")
    .Input("in_0_cross_output_kernel: T")
    .Input("in_0_cross_bias: T")

    .Input("in_0_ffn_gamma: T")
    .Input("in_0_ffn_beta: T")
    .Input("in_0_ffn_kernel1: T")
    .Input("in_0_ffn_bias1: T")
    .Input("in_0_ffn_kernel2: T")
    .Input("in_0_ffn_bias2: T")
    //layer_in_1
    .Input("in_1_self_gamma: T")
    .Input("in_1_self_beta: T")
    .Input("in_1_self_q_kernel: T")
    .Input("in_1_self_k_kernel: T")
    .Input("in_1_self_v_kernel: T")
    .Input("in_1_self_output_kernel: T")
    .Input("in_1_self_position_key: T")
    .Input("in_1_self_position_value: T")

    .Input("in_1_cross_gamma: T")
    .Input("in_1_cross_beta: T")
    .Input("in_1_cross_q_kernel: T")
    .Input("in_1_cross_k_kernel: T")
    .Input("in_1_cross_v_kernel: T")
    .Input("in_1_cross_output_kernel: T")
    .Input("in_1_cross_bias: T")

    .Input("in_1_ffn_gamma: T")
    .Input("in_1_ffn_beta: T")
    .Input("in_1_ffn_kernel1: T")
    .Input("in_1_ffn_bias1: T")
    .Input("in_1_ffn_kernel2: T")
    .Input("in_1_ffn_bias2: T")
    //layer_in_2
    .Input("in_2_self_gamma: T")
    .Input("in_2_self_beta: T")
    .Input("in_2_self_q_kernel: T")
    .Input("in_2_self_k_kernel: T")
    .Input("in_2_self_v_kernel: T")
    .Input("in_2_self_output_kernel: T")
    .Input("in_2_self_position_key: T")
    .Input("in_2_self_position_value: T")

    .Input("in_2_cross_gamma: T")
    .Input("in_2_cross_beta: T")
    .Input("in_2_cross_q_kernel: T")
    .Input("in_2_cross_k_kernel: T")
    .Input("in_2_cross_v_kernel: T")
    .Input("in_2_cross_output_kernel: T")
    .Input("in_2_cross_bias: T")

    .Input("in_2_ffn_gamma: T")
    .Input("in_2_ffn_beta: T")
    .Input("in_2_ffn_kernel1: T")
    .Input("in_2_ffn_bias1: T")
    .Input("in_2_ffn_kernel2: T")
    .Input("in_2_ffn_bias2: T")
    //layer_in_3
    .Input("in_3_self_gamma: T")
    .Input("in_3_self_beta: T")
    .Input("in_3_self_q_kernel: T")
    .Input("in_3_self_k_kernel: T")
    .Input("in_3_self_v_kernel: T")
    .Input("in_3_self_output_kernel: T")
    .Input("in_3_self_position_key: T")
    .Input("in_3_self_position_value: T")

    .Input("in_3_cross_gamma: T")
    .Input("in_3_cross_beta: T")
    .Input("in_3_cross_q_kernel: T")
    .Input("in_3_cross_k_kernel: T")
    .Input("in_3_cross_v_kernel: T")
    .Input("in_3_cross_output_kernel: T")
    .Input("in_3_cross_bias: T")

    .Input("in_3_ffn_gamma: T")
    .Input("in_3_ffn_beta: T")
    .Input("in_3_ffn_kernel1: T")
    .Input("in_3_ffn_bias1: T")
    .Input("in_3_ffn_kernel2: T")
    .Input("in_3_ffn_bias2: T")
    //layer_in_4
    .Input("in_4_self_gamma: T")
    .Input("in_4_self_beta: T")
    .Input("in_4_self_q_kernel: T")
    .Input("in_4_self_k_kernel: T")
    .Input("in_4_self_v_kernel: T")
    .Input("in_4_self_output_kernel: T")
    .Input("in_4_self_position_key: T")
    .Input("in_4_self_position_value: T")

    .Input("in_4_cross_gamma: T")
    .Input("in_4_cross_beta: T")
    .Input("in_4_cross_q_kernel: T")
    .Input("in_4_cross_k_kernel: T")
    .Input("in_4_cross_v_kernel: T")
    .Input("in_4_cross_output_kernel: T")
    .Input("in_4_cross_bias: T")

    .Input("in_4_ffn_gamma: T")
    .Input("in_4_ffn_beta: T")
    .Input("in_4_ffn_kernel1: T")
    .Input("in_4_ffn_bias1: T")
    .Input("in_4_ffn_kernel2: T")
    .Input("in_4_ffn_bias2: T")
    //layer_in_5_
    .Input("in_5_self_gamma: T")
    .Input("in_5_self_beta: T")
    .Input("in_5_self_q_kernel: T")
    .Input("in_5_self_k_kernel: T")
    .Input("in_5_self_v_kernel: T")
    .Input("in_5_self_output_kernel: T")
    .Input("in_5_self_position_key: T")
    .Input("in_5_self_position_value: T")

    .Input("in_5_cross_gamma: T")
    .Input("in_5_cross_beta: T")
    .Input("in_5_cross_q_kernel: T")
    .Input("in_5_cross_k_kernel: T")
    .Input("in_5_cross_v_kernel: T")
    .Input("in_5_cross_output_kernel: T")
    .Input("in_5_cross_bias: T")

    .Input("in_5_ffn_gamma: T")
    .Input("in_5_ffn_beta: T")
    .Input("in_5_ffn_kernel1: T")
    .Input("in_5_ffn_bias1: T")
    .Input("in_5_ffn_kernel2: T")
    .Input("in_5_ffn_bias2: T")

     //decoding
    .Input("embedding_table_init: T")
    .Input("embedding_table_run: T")
    .Input("embedding_kernel: T")
    .Input("decoding_gamma: T")
    .Input("decoding_beta: T")

    .Output("output_ids: int32")

    .Attr("T: {float,half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("memory_hidden_dim: int >= 1")
    .Attr("vocab_size: int >= 1")
    .Attr("end_id: int >= 0");

template <typename Device, typename T>
class DecodingOp : public CommonOp<T>
{
public:
  explicit DecodingOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
    OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
    OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
    OP_REQUIRES_OK(context, context->GetAttr("vocab_size", &vocab_size_));
    OP_REQUIRES_OK(context, context->GetAttr("end_id", &end_id_));
  }

  void Compute(OpKernelContext *context) override
  {
    // input(0): memory_tensor: [batch_size * beam_width, memory_max_seq_len, memory_hidden_dim]
    assert((int)(context->input(0).dims()) == 3);
    const int batch_size_ = (int)context->input(0).dim_size(0);
    const int seq_len_ = (int)context->input(0).dim_size(1);
    const int hidden_num_ = (int)context->input(0).dim_size(2);
    const int* language_id_ = reinterpret_cast<const int *>(context->input(1).flat<int>().data());

    const int* d_decode_length_ = reinterpret_cast<const int *>(context->input(2).flat<int>().data());
    int h_decode_length_ = 0;
    cudaMemcpy(&h_decode_length_, d_decode_length_, sizeof(int), cudaMemcpyDeviceToHost);
    const int max_decode_length_ = h_decode_length_;
    //int* language_id_ = nullptr;

    typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
    DecodingOpenNMT<DecodingTraits_::OpType> *decoding_opennmt_;
    fastertransformer::Allocator<AllocatorType::TF> allocator_(context);
    try
    {
      decoding_opennmt_ = new DecodingOpenNMT<DecodingTraits_::OpType>(allocator_, batch_size_, max_decode_length_,
                                                                       head_num_, size_per_head_,
                                                                       vocab_size_, num_layer_,
                                                                       hidden_num_, seq_len_,
                                                                       language_id_, end_id_);
    }
    catch (std::runtime_error &error)
    {
      OP_REQUIRES(context, false, errors::Internal(error.what()));
    }
    // assert input num is right
    //OP_REQUIRES(context, context->num_inputs() == 28 , errors::InvalidArgument("[ERROR] Less or more input arguments"));

    DecoderInitParam<DataType_> *params = new DecoderInitParam<DataType_>[num_layer_];
    const int hidden_unit = size_per_head_ * head_num_;
    int idx = 3;
    for (int i = 0; i < num_layer_; i++)
    {
      params[i].cublas_handle = this->get_cublas_handler();

      this->get_tensor(context, idx++, &params[i].self_layernorm.gamma);
      this->get_tensor(context, idx++, &params[i].self_layernorm.beta);
      this->get_tensor(context, idx++, &params[i].self_attention.query_weight);
      this->get_tensor(context, idx++, &params[i].self_attention.key_weight);
      this->get_tensor(context, idx++, &params[i].self_attention.value_weight);
      this->get_tensor(context, idx++, &params[i].self_attention.attention_output_weight);
      this->get_tensor(context, idx++, &params[i].self_attention.position_key);
      this->get_tensor(context, idx++, &params[i].self_attention.position_value);
    
      this->get_tensor(context, idx++, &params[i].cross_layernorm.gamma);
      this->get_tensor(context, idx++, &params[i].cross_layernorm.beta);
      this->get_tensor(context, idx++, &params[i].cross_attention.query_weight);
      this->get_tensor(context, idx++, &params[i].cross_attention.key_weight);
      this->get_tensor(context, idx++, &params[i].cross_attention.value_weight);
      this->get_tensor(context, idx++, &params[i].cross_attention.attention_output_weight);
      this->get_tensor(context, idx++, &params[i].cross_bias);

      this->get_tensor(context, idx++, &params[i].ffn_layernorm.gamma);
      this->get_tensor(context, idx++, &params[i].ffn_layernorm.beta);
      this->get_tensor(context, idx++, &params[i].ffn.intermediate_weight.kernel);
      this->get_tensor(context, idx++, &params[i].ffn.intermediate_weight.bias);
      this->get_tensor(context, idx++, &params[i].ffn.output_weight.kernel);
      this->get_tensor(context, idx++, &params[i].ffn.output_weight.bias);
    }

    DecodingInitParam<DataType_> decoding_params;
    decoding_params.cublas_handle = this->get_cublas_handler();

    Tensor *output_ids = nullptr;
    OP_REQUIRES_OK(context,context->allocate_output(0, {max_decode_length_, batch_size_}, &output_ids));
    decoding_params.output_ids = reinterpret_cast<int *>(output_ids->flat<int>().data());
    check_cuda_error(cudaMemset(decoding_params.output_ids, 0, sizeof(int) * max_decode_length_ * batch_size_));

    this->get_tensor(context, 0, &decoding_params.memory_tensor);
    this->get_tensor(context, idx++, &decoding_params.embedding_table_init);
    this->get_tensor(context, idx++, &decoding_params.embedding_table_run);
    this->get_tensor(context, idx++, &decoding_params.embedding_kernel);
    this->get_tensor(context, idx++, &decoding_params.layernorm.gamma);
    this->get_tensor(context, idx++, &decoding_params.layernorm.beta);
    
    mtx.lock();

    OP_REQUIRES_OK(context,functor::DecodingOpFunctor<Device, T>::DynamicDecode(context, num_layer_, params, decoding_opennmt_, max_decode_length_,decoding_params));

    mtx.unlock();
    delete decoding_opennmt_;
    delete params;
  }

private:
  std::mutex mtx;
  int head_num_;
  int size_per_head_;
  int num_layer_;
  int memory_hidden_dim_;
  int vocab_size_;
  int end_id_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)  REGISTER_KERNEL_BUILDER(Name("Decoding").Device(DEVICE_GPU).TypeConstraint<T>("T"),DecodingOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
} //namespace
} //namespace tensorflow
