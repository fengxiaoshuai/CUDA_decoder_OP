#ifndef _DECODING_H_
#define _DECODING_H_

#include "common.h"
#include "decoder.h"
#include "allocator.h"
#include <cuda_runtime.h>

namespace fastertransformer
{
template <class T>
class DecodingInitParam
{
public:
  const T *embedding_table_init;
  const T *embedding_table_run;
  const T *embedding_kernel;
  const T *memory_tensor;
  LayerNormWeight<T> layernorm;
  int *output_ids;
  const int *start_ids;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};


template <OperationType OpType_>
class DecodingOpenNMT
{
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const IAllocator &allocator_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[1] = {20};

  int batch_size_;
  int beam_width_;
  int max_decode_length_;//decode_length
  int seq_len_;
  int head_num_;
  int size_per_head_;
  int hidden_units_;
  int decoder_layers_;
  int vocab_size_;
  OpenDecoder<OpType_> *decoder_;
  DataType_ **K_cache_;
  DataType_ **V_cache_;
  DataType_ **K_mem_cache_;
  DataType_ **V_mem_cache_;
  DataType_ *from_tensor_[2];
  DataType_ *decoder_buf_;
  DataType_ *decoder_result_buf_;
  float *logits_buf_;
  int *word_ids_buf_;
  bool *finished_buf_;
  int *output_ids_buf_;
  void *buf_;
  const int *start_id_;
  int end_id_;
  bool *h_finished_buf_;

public:
  DecodingOpenNMT(const IAllocator &allocator, const int batch_size, const int max_decode_length,
                  const int head_num, const int size_per_head, const int vocab_size,
                  const int decoder_layers, const int hidden_units, const int seq_len,
                  const int* start_id, const int end_id);

  void forward(const DecoderInitParam<DataType_> *param, DecodingInitParam<DataType_> decoding_params);
  
  virtual ~DecodingOpenNMT() 
  {
	delete []K_cache_;
	delete []V_cache_;
	delete []K_mem_cache_;
	delete []V_mem_cache_;
	delete []h_finished_buf_;
        delete decoder_;
  };
};

} //namespace fastertransformer
#endif

