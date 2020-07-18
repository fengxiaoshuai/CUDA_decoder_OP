#ifndef _DECODER_H_
#define _DECODER_H_

#include "allocator.h"
#include "common.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fastertransformer
{

template <typename T>
class DecoderInitParam
{
public:
  LayerNormWeight<T> self_layernorm;
  AttentionWeight<T> self_attention;

  LayerNormWeight<T> cross_layernorm;
  AttentionWeight<T> cross_attention;
  const T* cross_bias;

  LayerNormWeight<T> ffn_layernorm;
  FFNWeight<T> ffn;

  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <OperationType OpType_>
class DecoderTransformerTraits;

template <>
class DecoderTransformerTraits<OperationType::FP32> : public TransformerTraits<OperationType::FP32>{};

template <>
class DecoderTransformerTraits<OperationType::FP16> : public TransformerTraits<OperationType::FP16>{};

template <OperationType OpType_>
class OpenDecoder
{
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  const IAllocator &allocator_;
  typedef typename Traits_::DataType DataType_;
  DecoderInitParam<DataType_> param_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[4];

  int batch_size_;
  int sentence_len_;
  int max_decode_length_;
  int head_num_;
  int size_per_head_;
  int hidden_units_;
  int memory_hidden_units_;

  DataType_ *norm_from_tensor_buf_; 
  DataType_ *query_buf_;
  DataType_ *context_buf_;
  DataType_ *masked_output_buf_;
  DataType_ *norm_masked_output_buf_;
  DataType_ *cross_output_buf_;
  DataType_ *norm_cross_output_buf_;
  DataType_ *ffn_inner_buf_;
  DataType_ *position_value_buf_;
  DataType_ *position_key_buf_;

public:
  OpenDecoder(const IAllocator &allocator, int batch_size, int seq_len, int head_num, int size_per_head,
              int memory_hidden_units, int max_decode_length); 

  int getWorkspaceSize();

  void initialize(DecoderInitParam<DataType_> param, DataType_ *buf);

  void add_input(DataType_ *output, const DataType_ *input, const int m, const int n);

  void ffn(const DataType_ *input, DataType_ *ffn_inner, DataType_ *output,
           const int m, const int inner_size, const int n);

  void forward(const DataType_ *from_tensor, const DataType_ *memory_tensor,
               DataType_ *key_cache_, DataType_ *value_cache_,
               DataType_ *key_mem_cache_, DataType_ *value_mem_cache_,
               DataType_ *decoder_output, const int step);

  void decoder_norm1(const DataType_ *from_tensor, const DataType_ *gamma,
                     const DataType_ *beta, DataType_ *norm_from_tensor_buf_,
                     const int m, const int n);

  void decoder_norm2(const DataType_ *input, const DataType_ *gamma, 
                     const DataType_ *beta, DataType_ *output, 
                     DataType_ *norm_output, const int m, const int n);

  void embedding_relative_position(DataType_ *position_key_out, 
		                   DataType_ *position_value_out,
                                   const int step);

  void masked_multi_head_attention(const DataType_ *from_tensor, DataType_ *key_cache_,
                                   DataType_ *value_cache_, DataType_ *position_key_, 
                                   DataType_ *position_value_,
				   DataType_ *decoder_output,
                                   const int step);

  void cross_multi_head_attention (const DataType_ *from_tensor, const DataType_ *memory_tensor,
                                   DataType_ *key_mem_cache_, DataType_ *value_mem_cache_, 
                                   DataType_ *decoder_output,  
                                   const int seq_len, 
                                   const int step);

  ~OpenDecoder(){};
};

} 
#endif
