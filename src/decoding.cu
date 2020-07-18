#include "decoding.h"
#include <vector>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <iostream>


namespace fastertransformer
{

template <typename T>
__global__ void embedding_lookup_kernel(const T* embedding_table, const int* word_ids,
    const int hidden_units, T* from_tensor)
{
  int write_pos = threadIdx.x + blockIdx.x * hidden_units;
  from_tensor[write_pos] = embedding_table[word_ids[blockIdx.x] * hidden_units + threadIdx.x] * (T)32.0f;
}


template <typename T>
__global__ void embedding_init_lookup_kernel(const T* embedding_table, const int* word_ids,
    const int hidden_units, T* from_tensor)
{
  int write_pos = threadIdx.x + blockIdx.x * hidden_units;
  from_tensor[write_pos] = embedding_table[word_ids[blockIdx.x] * hidden_units + threadIdx.x] ;
}

template <typename T>
__inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  return val;
}


template <typename T>
__inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32]; 
//  __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
  val = warpReduceMax(val);

  return val;
}

template <typename T>
__global__ void greedy_search_kernel(int* out_put, T* logit, int batch, int step)
{
  int id = (threadIdx.x + blockIdx.x * 32) * 1024;//32 is the num of threads in a block
  float max = -1000.0f;
  int pos = 0;
  for(int i=0; i<1024; i++)
  {
     if(logit[id + i]>max)
     {
	max = logit[id + i];
        pos = id + i;
     }	
  };
  float max_val = blockReduceMax<float>(max);
  if(max==max_val){out_put[blockIdx.x]= pos % 32768;};
}
/*i*************************************************************************/

template <typename T>
void greedy_search(int* out_puts, T* logits, const int batch_size, int step, cudaStream_t stream)
{
   dim3 grid(batch_size);
   dim3 block(32);
   greedy_search_kernel<<<grid, block, 0, stream>>>(out_puts, logits, batch_size, step);
}

template <typename T>
void embedding_init_lookup(const T* embedding_table, const int* word_ids, T* from_tensor,
                           const int batch_size, const int hidden_units, cudaStream_t stream)
{
   dim3 grid(batch_size);
   dim3 block(hidden_units);
   assert(hidden_units <= 1024);
   embedding_init_lookup_kernel<<<grid, block, 0, stream>>>(embedding_table, word_ids,
                                                            hidden_units, from_tensor);
}

template <typename T>
void embedding_lookup(const T* embedding_table, const int* word_ids, T* from_tensor, const int batch_size, const int hidden_units, cudaStream_t stream)
{
   dim3 grid(batch_size);
   dim3 block(hidden_units);
   assert(hidden_units <= 1024);
   embedding_lookup_kernel<<<grid, block, 0, stream>>>(embedding_table, word_ids, hidden_units, from_tensor);
}

template <OperationType OpType_>
DecodingOpenNMT<OpType_>::DecodingOpenNMT(const IAllocator &allocator,
                const int batch_size,
                const int max_decode_length,
                const int head_num,
                const int size_per_head,
                const int vocab_size,
                const int decoder_layers,
                const int hidden_units,
                const int seq_len,
                const int* start_id,
                const int end_id) :
                allocator_(allocator),
                batch_size_(batch_size),
                max_decode_length_(max_decode_length), 
                head_num_(head_num),
                size_per_head_(size_per_head),
                vocab_size_(vocab_size),
                decoder_layers_(decoder_layers), 
                hidden_units_(hidden_units),
                start_id_(start_id), 
                end_id_(end_id), 
                seq_len_(seq_len)
{
    K_cache_ = new DataType_ *[decoder_layers_];
    V_cache_ = new DataType_ *[decoder_layers_];

    K_mem_cache_ = new DataType_ *[decoder_layers_];
    V_mem_cache_ = new DataType_ *[decoder_layers_];

    decoder_ = new OpenDecoder<OpType_>(allocator, batch_size, seq_len, head_num, size_per_head, hidden_units, max_decode_length_);
    int tensor_size = batch_size_ * hidden_units_;      //decoding input
    long long int decoder_workspace_size = decoder_->getWorkspaceSize();             // decoder_buf
    long long int cache_size = batch_size_ * max_decode_length_ * hidden_units_; // cache size
    long long int mem_size = batch_size_ * seq_len_ * hidden_units_;
    int decoder_result_size = batch_size * hidden_units_;
    long long int logits_size = batch_size_ *  vocab_size_;         // type float
    int word_ids_size = batch_size_ ; //type int
    int finished_size = batch_size_ ; //type bool
    int output_size = batch_size_ * max_decode_length_;

    long long int datatype_size = tensor_size * 2 + decoder_workspace_size + (cache_size + mem_size) * 2 * decoder_layers_ + decoder_result_size;
    //std::cout<<"tensor_size: "<<tensor_size<<std::endl; 
    //std::cout<<"decoder_space: "<<decoder_workspace_size<<std::endl; 
    //std::cout<<"cache: "<<(cache_size + mem_size)*2*decoder_layers_<<std::endl; 
    //std::cout<<"result_size: "<<decoder_result_size<<std::endl; 

    buf_ = reinterpret_cast<void *>(allocator_.malloc(sizeof(DataType_) * datatype_size + sizeof(float) * (logits_size ) + sizeof(int) * word_ids_size + sizeof(bool) * finished_size + sizeof(int) * output_size));

    from_tensor_[0] = (DataType_ *)buf_;
    from_tensor_[1] = (DataType_ *)(from_tensor_[0] + tensor_size);

    for (int i = 0; i < decoder_layers_; ++i)
    {
      K_mem_cache_[i] = from_tensor_[1] + tensor_size + i * mem_size * 2;
      V_mem_cache_[i] = K_mem_cache_[i] + mem_size;
    }

    for (int i = 0; i < decoder_layers_; ++i)
    {
      K_cache_[i] = V_mem_cache_[decoder_layers - 1] + mem_size + i * cache_size * 2;
      V_cache_[i] = K_cache_[i] + cache_size;
    }

    decoder_buf_ = V_cache_[decoder_layers - 1] + cache_size;
    decoder_result_buf_ = (decoder_buf_ + decoder_workspace_size);
    logits_buf_ = (float* )(decoder_result_buf_ + decoder_result_size);
    word_ids_buf_ = (int* )(logits_buf_ + logits_size);
    finished_buf_ = (bool*)(word_ids_buf_ + word_ids_size);
    output_ids_buf_ = (int* )(finished_buf_ + word_ids_size);
    h_finished_buf_ = new bool[finished_size];

    int err = 0;
    if (err != 1)
    {
      if (Traits_::OpType == OperationType::FP32)
      {
        cublasAlgo_[0] = CUBLAS_GEMM_DEFAULT;
      }
      else
      {
        cublasAlgo_[0] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      }
    }
}

template <OperationType OpType_>
void DecodingOpenNMT<OpType_>::forward(const DecoderInitParam<DataType_> *param,
                              DecodingInitParam<DataType_> decoding_params)
{

    int m = batch_size_;
    int k = hidden_units_;
    int n = vocab_size_;

    int cache_size = batch_size_ * max_decode_length_ * hidden_units_;

    for (int step = 1; step <= max_decode_length_; ++step)
    {
      int kv_cache_id = step & 0x1;
      if(step == 1)
      {
          embedding_init_lookup(decoding_params.embedding_table_init, start_id_, from_tensor_[0],
                       batch_size_,  hidden_units_, decoding_params.stream);
      }
      else
      {
          embedding_lookup(decoding_params.embedding_table_run, word_ids_buf_, from_tensor_[0],
                       batch_size_,  hidden_units_, decoding_params.stream);
      }
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());

      int from_id, out_id;
      for (int layer = 0; layer < decoder_layers_; ++layer)
      {
        from_id = layer & 0x1;
        out_id = 1 - from_id;

        decoder_->initialize(param[layer], decoder_buf_);

        decoder_->forward(from_tensor_[from_id], decoding_params.memory_tensor,
                          K_cache_[layer], V_cache_[layer],
                          K_mem_cache_[layer], V_mem_cache_[layer],
                          from_tensor_[out_id], step);
      }
      decoder_->decoder_norm1(from_tensor_[out_id], decoding_params.layernorm.gamma,
                  decoding_params.layernorm.beta, decoder_result_buf_, m, k);

      float alpha = (float)1.0f;
      float beta = (float)0.0f;
      
      check_cuda_error(cublasGemmEx(decoding_params.cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    decoding_params.embedding_kernel, AType_, k,
                                    decoder_result_buf_, BType_, k,
                                    &beta,
                                    logits_buf_, CUDA_R_32F, n,
                                    CUDA_R_32F,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
      /*
      int* pos = new int[batch_size_];
      for(int i=0; i<batch_size_; i++)
      {
          thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(logits_buf_ + i * vocab_size_);
          thrust::device_ptr<float> iter = thrust::max_element(d_ptr, d_ptr + vocab_size_);

          pos[i] = iter - d_ptr;
          if(pos[i]==2) h_finished_buf_[i] = true;
          // printf("%d_max_value: %d \n", i, pos[i]);
      };
      int size = batch_size_ * sizeof(int);
      cudaMemcpy(word_ids_buf_, pos, size, cudaMemcpyHostToDevice);
      cudaMemcpy(decoding_params.output_ids + (step-1) * batch_size_, word_ids_buf_, size, cudaMemcpyDeviceToDevice);
      delete []pos;
      */
      int pos[2];
      greedy_search(word_ids_buf_, logits_buf_, batch_size_, step, decoding_params.stream);
      int size = batch_size_ * sizeof(int);
      cudaMemcpy(decoding_params.output_ids + (step-1) * batch_size_, word_ids_buf_, size, cudaMemcpyDeviceToDevice);
      cudaMemcpy(pos, word_ids_buf_, size, cudaMemcpyDeviceToHost);
      
      for(int i=0; i<batch_size_; i++)
      {
          if(pos[i]==2) h_finished_buf_[i] = true;
      };

      int sum = 0;
      for(int i = 0; i < batch_size_; i++){
        sum += (int)h_finished_buf_[i];
      }
      if(sum == batch_size_) break;
    };
}

template void DecodingOpenNMT<OperationType::FP16>::forward(const DecoderInitParam<DataType_> *param,
                                                            DecodingInitParam<DataType_> decoding_params);

template DecodingOpenNMT<OperationType::FP16>::DecodingOpenNMT(const IAllocator &allocator,
                const int batch_size,
                const int max_decode_length,
                const int head_num,
                const int size_per_head,
                const int vocab_size,
                const int decoder_layers,
                const int hidden_units,
                const int seq_len,
                const int* start_id,
                const int end_id); 

template void DecodingOpenNMT<OperationType::FP32>::forward(const DecoderInitParam<DataType_> *param,
                                                            DecodingInitParam<DataType_> decoding_params);

template DecodingOpenNMT<OperationType::FP32>::DecodingOpenNMT(const IAllocator &allocator,
                const int batch_size,
                const int max_decode_length,
                const int head_num,
                const int size_per_head,
                const int vocab_size,
                const int decoder_layers,
                const int hidden_units,
                const int seq_len,
                const int* start_id,
                const int end_id); 

}
