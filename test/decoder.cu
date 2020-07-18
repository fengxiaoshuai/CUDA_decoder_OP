#include "decoder.h"

#define FINAL_MASK 0xffffffff


namespace fastertransformer
{

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}


template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  //__shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
                              
  return val;
}


template <typename T>
__inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
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
__global__ 
void add_bias_relu(T* out, const T* bias, int m, int n)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m)
    {
      val = out[tid + i * blockDim.x + row_id * n] + reg_bias;
      out[tid + i * blockDim.x + row_id * n] = (T)(val > 0.0f ? val : 0.0f);
      row_id += gridDim.x;
     }
  }
}


template <typename T>
__global__ 
void add_bias_kernel(T* output,  const T* bias, const int m, const int n)
{
  int id = blockIdx.x * n + threadIdx.x;
  output[id] = output[id]  + __ldg(&bias[threadIdx.x]);
}


template <typename T>
__global__ 
void masked_attention_kernel(T* query_buf, T* key_cache,  T* value_cache, 
                             T* position_key, T* position_value, T* context_buf,
                             int batch_size, int head_num, int size_per_head, 
                             const int step, const T scalar)
{
  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  // k+position_key
  if(tid < size_per_head)
  {
    sq[tid] = query_buf[qkv_id];
  }
  __syncthreads();

  //q*(k+position_key)
  int offset = batch_size * head_num * size_per_head;
  for(int ite = 0; ite < step; ++ite)
  {
    int position_key_id = ite * size_per_head + tid;
    T key = tid < size_per_head ? key_cache[ite * offset + qkv_id] : (T)0.0f;
    //for the last step, we should update K + bias_K to the cache
    if(1 && tid < size_per_head)
    {
      key += position_key[position_key_id];
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads(); //try to remove
  //softmax
  __shared__ float s_max_val, s_sum;
  float local_i = tid < step ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < step ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  if(tid < step)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  // softmax*v 
  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < step; ++ite)
    {
      int position_value_id = ite * size_per_head + tid;
      T value = value_cache[ite * offset + qkv_id];
      //for the last step, we should update K + bias_K to the cache
      if(1)
      {
        value += position_value[position_value_id];
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}


template <typename T>
__global__ 
void embedding_kernel(const T* k_table, const T* v_table, T* k_out, T* v_out,
                      int* position_ids, const int num, int step)
{
  if(threadIdx.x == 0)
  {
   int val = 21 - step + blockIdx.x;
   position_ids[blockIdx.x] = val >= 0? val:0;
  }
  __syncthreads();
  int write_pos = threadIdx.x + blockIdx.x * num;
  k_out[write_pos] = k_table[position_ids[blockIdx.x] * num + threadIdx.x];
  v_out[write_pos] = v_table[position_ids[blockIdx.x] * num + threadIdx.x];
}


template<typename T>
__global__
void cross_attention_kernel(T* query_buf, T* key_cache, T* value_cache, const T* bias,
                            T* context_buf, int batch_size, int head_num, int size_per_head, 
                            int step, const int seq_len, const T scalar)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int length = seq_len ;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id]; 
  __syncthreads();

  //q*k
  for(int ite = 0; ite < length; ++ite)
  {
    int key_id = bid * (seq_len * head_num * size_per_head) + ite * (head_num * size_per_head)
     + head_id * size_per_head + tid;

    T key = tid < size_per_head ? key_cache[key_id] : (T)(0.0f);
    T val = (tid < size_per_head) ? key * sq[tid] * scalar  : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
    {
      logits[ite] = qk + bias[ bid * length + ite];
      //printf("addbias_%d_%d_%d: %f \n", bid, head_id,  ite , logits[ite]);
    }
    __syncthreads(); //try to remove
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = tid < length ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < length ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();
  if(tid < length)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < length; ++ite)
    {
      int value_id = bid * seq_len * head_num * size_per_head + ite * head_num * size_per_head 
        + head_id * size_per_head + tid;
      T value = value_cache[value_id];
      sum += value * logits[ite];
    }
    context_buf[bid * head_num * size_per_head + head_id * size_per_head + tid] = sum;
  }
}


template <typename T>
__global__
void decoder_norm1_kernel(const T* input, const T* gamma, const T* beta, T* output, int m, int n)
{
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float local_out = tid < n ? (float)(__ldg(&input[blockIdx.x * n + tid])) : 0.0f;
  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
  {
    s_mean = mean / n;
  }
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);

  if(threadIdx.x == 0)
  {
    s_variance = rsqrtf(variance / n + 1e-6);
  }
  __syncthreads();

  if(tid < n)
  {
    output[blockIdx.x * n + tid] =  (T)(((local_out - s_mean) * s_variance) * \
    (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
  }

  int index = blockIdx.x * n + tid;
  __syncthreads();
}


template <typename T>
__global__
void decoder_norm2_kernel(const T* input, const T* gamma, const T* beta,
                          T* output, T* norm_output, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  if(tid < n)
  {
    local_out = (float)(__ldg(&input[blockIdx.x * n + tid]));
    local_out += (float)(output[blockIdx.x * n + tid]);
    output[blockIdx.x * n + tid] = (T)local_out;
  }

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  if(tid < n)
    norm_output[blockIdx.x * n + tid] = 
      (T)((local_out - s_mean) * s_variance * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
  int qkv = blockIdx.x * n + tid ;
}


template <typename T>
__global__ 
void add_input_kernel(T* output, const T* input, const int m, const int n)
{
  int id = blockIdx.x * n + threadIdx.x;
  output[id] = output[id] + input[id];
}


/*****************************************************************************************/

template <OperationType OpType_>
OpenDecoder<OpType_>::OpenDecoder(const IAllocator &allocator,
                         int batch_size,
                         int seq_len,
                         int head_num,
                         int size_per_head,
                         int memory_hidden_units,
                         int max_decode_length) :
                         allocator_(allocator), 
                         batch_size_(batch_size),
                         sentence_len_(seq_len),
                         head_num_(head_num),
                         size_per_head_(size_per_head),
                         memory_hidden_units_(memory_hidden_units),
                         max_decode_length_(max_decode_length)
{
    hidden_units_ = head_num_ * size_per_head_;
    if (Traits_::OpType == OperationType::FP32)
      {
        cublasAlgo_[0] = CUBLAS_GEMM_DEFAULT;
        cublasAlgo_[1] = CUBLAS_GEMM_DEFAULT;
        cublasAlgo_[2] = CUBLAS_GEMM_DEFAULT;
        cublasAlgo_[3] = CUBLAS_GEMM_DEFAULT;
      }
     else
      {
        cublasAlgo_[0] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        cublasAlgo_[1] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        cublasAlgo_[2] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        cublasAlgo_[3] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      }
}


template <OperationType OpType_>
int OpenDecoder<OpType_>::getWorkspaceSize()
{
      int buf_size = batch_size_ * hidden_units_;
      return 12 * buf_size + 2 * max_decode_length_ * size_per_head_ ;
}


template <OperationType OpType_>
void OpenDecoder<OpType_>::initialize(DecoderInitParam<DataType_> param, DataType_ *buf)
{
    param_ = param;
    int buf_size = batch_size_ * hidden_units_;
    norm_from_tensor_buf_ = buf;
    query_buf_ = buf + 1 * buf_size;       //store the query values (from_tensor * Q)
    context_buf_ = buf + 2 * buf_size; //store the context result (softmax(qk)v)

    masked_output_buf_ = buf + 3 * buf_size;      //masked_attention_output
    norm_masked_output_buf_ = buf + 4 * buf_size; //norm(masked_attention_output)

    cross_output_buf_ = buf + 5 * buf_size;      //mutli-head attention_output
    norm_cross_output_buf_ = buf + 6 * buf_size; //norm(multi-head attention_output)
    ffn_inner_buf_ = buf + 7 * buf_size;         //4 buf size to store inner product
    position_key_buf_ = ffn_inner_buf_ + 4 * buf_size;
    position_value_buf_ = position_key_buf_ + max_decode_length_ * size_per_head_;
}


template <OperationType OpType_>
void OpenDecoder<OpType_>::forward(const DataType_ *from_tensor, const DataType_ *memory_tensor,
                                   DataType_ *key_cache_, DataType_ *value_cache_,
                                   DataType_ *key_mem_cache_, DataType_ *value_mem_cache_,
                                   DataType_ *decoder_output, const int step)
{
  int m = batch_size_;
  int n = hidden_units_;

  try
  {
    decoder_norm1(from_tensor, param_.self_layernorm.gamma, param_.self_layernorm.beta,
                  norm_from_tensor_buf_, m, n);

    embedding_relative_position(position_key_buf_, position_value_buf_, step);

    masked_multi_head_attention(norm_from_tensor_buf_, key_cache_, value_cache_, 
                                position_key_buf_, position_value_buf_, masked_output_buf_,step);

    decoder_norm2(from_tensor, param_.cross_layernorm.gamma, param_.cross_layernorm.beta,
                  masked_output_buf_,norm_masked_output_buf_, m, n);

    cross_multi_head_attention(norm_masked_output_buf_, memory_tensor, key_mem_cache_, 
                               value_mem_cache_, cross_output_buf_, sentence_len_, step);

    decoder_norm2(masked_output_buf_, param_.ffn_layernorm.gamma, param_.ffn_layernorm.beta,
                  cross_output_buf_, norm_cross_output_buf_, m, n);

    ffn(norm_cross_output_buf_, ffn_inner_buf_, decoder_output, m, 4 * n, n);

    add_input(decoder_output, cross_output_buf_, m, n);
  }
  catch (std::runtime_error &error)
  {
    throw error;
  }
}


template<OperationType OpType_>
void OpenDecoder<OpType_>::decoder_norm1(const DataType_* input, const DataType_* gamma,
                                         const DataType_* beta, DataType_* output,int m, int n)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));
  if(n % 32 != 0)
    block.x = 1024;
  assert(n <= 1024);
  decoder_norm1_kernel<DataType_><<<grid, block, 0, param_.stream>>>(input, gamma, beta, output, m, n);
}


template<OperationType OpType_>
void OpenDecoder<OpType_>::decoder_norm2(const DataType_* input, const DataType_* gamma,
                                         const DataType_* beta, DataType_* output,
                                         DataType_* norm_output, int m, int n)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));
  assert(n <= 1024);
  if(n % 32 != 0)
    block.x = 1024;
  decoder_norm2_kernel<DataType_><<<grid, block, 0, param_.stream>>>(input, gamma, beta, output,
                                                                     norm_output, m, n);
}



template<OperationType OpType_>
void OpenDecoder<OpType_>::ffn(const DataType_* input, DataType_* ffn_inner, DataType_* output,
                               const int m, const int inner_size, const int n)
{
  int m1 = m, k1 = n, n1 = inner_size;
  DataType_ alpha = (DataType_)1.0f;
  DataType_ beta = (DataType_)0.0f;

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n1, m1, k1, 
    &alpha, 
    param_.ffn.intermediate_weight.kernel, AType_, n1, 
    input, BType_, k1, 
    &beta, 
    ffn_inner, CType_, n1, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));

  dim3 grid(m1);
  dim3 block(n1 / 4);

  assert(block.x <= 1024);

  add_bias_relu<DataType_><<<grid, block, 0, param_.stream>>>(ffn_inner, param_.ffn.intermediate_weight.bias, m1, n1);

  int m2 = m, n2 = n, k2 = inner_size;
  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n2, m2, k2, 
    &alpha, 
    param_.ffn.output_weight.kernel, AType_, n2, 
    ffn_inner, BType_, k2, 
    &beta, 
    output, CType_, n2, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[3])));
  grid = m;
  block = n;
  add_bias_kernel<DataType_><<<grid, block, 0, param_.stream>>>(output, param_.ffn.output_weight.bias, m, n);
}



template<OperationType OpType_>
void OpenDecoder<OpType_>::add_input(DataType_* output, const DataType_* input, const int m, const int n)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_input_kernel<<<grid, block, 0, param_.stream>>>(output, input,  m, n);
}


template<OperationType OpType_>
void OpenDecoder<OpType_>::cross_multi_head_attention(const DataType_* from_tensor,
                                                      const DataType_* memory_tensor,
                                                      DataType_* key_mem_cache,
                                                      DataType_* value_mem_cache,
                                                      DataType_* decoder_output,
                                                      const int seq_len,
                                                      const int step)
{
  int m = batch_size_;
  int n = hidden_units_;
  int k = hidden_units_;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  //reuse the query_buf 
  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.cross_attention.query_weight, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    query_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
  if(step == 1)
  {
    m *= seq_len;
    k = memory_hidden_units_;
    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.cross_attention.key_weight, AType_, n, 
      memory_tensor, BType_, k, 
      &beta, 
      key_mem_cache, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.cross_attention.value_weight, AType_, n, 
      memory_tensor, BType_, k, 
      &beta, 
      value_mem_cache, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));
    k = hidden_units_;
  }

  dim3 grid(batch_size_ * head_num_);
  dim3 block(128);

  if(seq_len <= 64)
    block.x = 64;
  else if(seq_len <= 128 && seq_len > size_per_head_)
    block.x = 128;
  else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head_)
    block.x = size_per_head_;

  assert(block.x <= 1024);
  
  DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);

  int shared_size = sizeof(DataType_) * (size_per_head_ + seq_len);
  cross_attention_kernel<DataType_><<<grid, block, shared_size, param_.stream>>>(
    query_buf_, 
    key_mem_cache, 
    value_mem_cache, 
    param_.cross_bias,
    context_buf_,  
    batch_size_,
    head_num_, size_per_head_, step, seq_len, scalar);

  m = batch_size_;
  n = head_num_ * size_per_head_;
  k = n;

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.cross_attention.attention_output_weight, AType_, n, 
    context_buf_, BType_, k, 
    &beta, 
    decoder_output, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
}


template<OperationType OpType_>
void OpenDecoder<OpType_>::masked_multi_head_attention(const DataType_* from_tensor,
                                                       DataType_* key_cache_,
                                                       DataType_* value_cache_,
                                                       DataType_* position_key_,
                                                       DataType_* position_value_,
                                                       DataType_* decoder_output,
                                                       const int step)
{
  int m = batch_size_;
  int n = hidden_units_;
  int k = hidden_units_;

  DataType_* key_buf_ = key_cache_ + (step - 1) * m * n;
  DataType_* value_buf_ = value_cache_ + (step - 1) * m * n;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

   check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.query_weight , AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    query_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.key_weight, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    key_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.value_weight, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    value_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  dim3 grid(batch_size_ * head_num_);
  dim3 block(128);

  //suppose size_per_head <= 128
  if(step <= 64)
    block.x = 64;
  else if(step <= 128 && step > size_per_head_)
    block.x = 128;
  else if(step > 128 && step <= 256)
    block.x = 256;
  else if(step > 256 && step <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head_)  block.x = size_per_head_;
  
  assert(block.x <= 1024);

  DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);

  int shared_size = sizeof(DataType_) * (size_per_head_ + step);

  masked_attention_kernel<DataType_><<<grid, block, shared_size, param_.stream>>>(
    query_buf_,  
    key_cache_, 
    value_cache_, 
    position_key_,
    position_value_,
    context_buf_, batch_size_,
    head_num_, size_per_head_, step, scalar);

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.attention_output_weight, AType_, n, 
    context_buf_, BType_, k, 
    &beta, 
    decoder_output, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
} 


template<OperationType OpType_>
void OpenDecoder<OpType_>::embedding_relative_position(DataType_* position_key_out,
                                                       DataType_* position_value_out,
                                                       const int step)
{
        dim3 grid(step);
	dim3 block(size_per_head_);
        int* ptr;
        cudaMalloc((void**)&ptr, sizeof(int)*step);
        embedding_kernel<DataType_><<<grid, block, 0, param_.stream>>>(
        param_.self_attention.position_key, param_.self_attention.position_value, 
        position_key_out,
        position_value_out,
        ptr, 
	size_per_head_,
        step);
        cudaFree(ptr);
}

/*
template int OpenDecoder<OperationType::FP16>::getWorkspaceSize();

template void OpenDecoder<OperationType::FP16>::initialize(DecoderInitParam<DataType_> param, DataType_ *buf);

template void OpenDecoder<OperationType::FP16>::add_input(DataType_* output, const DataType_* input, const int m, const int n);

template void OpenDecoder<OperationType::FP16>::decoder_norm1(const DataType_* input, const DataType_* gamma,
                                        const DataType_* beta, DataType_* output,int m, int n);

template void OpenDecoder<OperationType::FP16>::decoder_norm2(const DataType_* input, const DataType_* gamma,
                                        const DataType_* beta, DataType_* output,
                                        DataType_* norm_output, int m, int n);

template void OpenDecoder<OperationType::FP16>::ffn(const DataType_* input, DataType_* ffn_inner, DataType_* output,
                                        const int m, const int inner_size, const int n);

template OpenDecoder<OperationType::FP16>::OpenDecoder(const IAllocator &allocator,
                                                        int batch_size,
                                                        int seq_len,
                                                        int head_num,
                                                        int size_per_head,
                                                        int memory_hidden_units,
                                                        int max_decode_length); 

template void OpenDecoder<OperationType::FP16>::cross_multi_head_attention(const DataType_* from_tensor,
                                                        const DataType_* memory_tensor,
                                                        DataType_* key_mem_cache,
                                                        DataType_* value_mem_cache,
                                                        DataType_* decoder_output,
                                                        const int seq_len,
                                                        const int step);

template void OpenDecoder<OperationType::FP16>::masked_multi_head_attention(const DataType_* from_tensor,
                                                       DataType_* key_cache_,
                                                       DataType_* value_cache_,
                                                       DataType_* position_key_,
                                                       DataType_* position_value_,
                                                       DataType_* decoder_output,
                                                       const int step);

template void OpenDecoder<OperationType::FP16>::embedding_relative_position(DataType_* position_key_out,
                                                       DataType_* position_value_out,
                                                       const int step);

template void OpenDecoder<OperationType::FP16>::forward(const DataType_ *from_tensor, const DataType_ *memory_tensor,
                                                      DataType_ *key_cache_, DataType_ *value_cache_,
                                                      DataType_ *key_mem_cache_, DataType_ *value_mem_cache_,
                                                      DataType_ *decoder_output, const int step);

*/
template int OpenDecoder<OperationType::FP32>::getWorkspaceSize();

template void OpenDecoder<OperationType::FP32>::initialize(DecoderInitParam<DataType_> param, DataType_ *buf);

template void OpenDecoder<OperationType::FP32>::add_input(DataType_* output, const DataType_* input, const int m, const int n);

template void OpenDecoder<OperationType::FP32>::decoder_norm1(const DataType_* input, const DataType_* gamma,
                                        const DataType_* beta, DataType_* output,int m, int n);

template void OpenDecoder<OperationType::FP32>::decoder_norm2(const DataType_* input, const DataType_* gamma,
                                        const DataType_* beta, DataType_* output,
                                        DataType_* norm_output, int m, int n);

template void OpenDecoder<OperationType::FP32>::ffn(const DataType_* input, DataType_* ffn_inner, DataType_* output,
                                        const int m, const int inner_size, const int n);

template OpenDecoder<OperationType::FP32>::OpenDecoder(const IAllocator &allocator,
                                                        int batch_size,
                                                        int seq_len,
                                                        int head_num,
                                                        int size_per_head,
                                                        int memory_hidden_units,
                                                        int max_decode_length); 

template void OpenDecoder<OperationType::FP32>::cross_multi_head_attention(const DataType_* from_tensor,
                                                        const DataType_* memory_tensor,
                                                        DataType_* key_mem_cache,
                                                        DataType_* value_mem_cache,
                                                        DataType_* decoder_output,
                                                        const int seq_len,
                                                        const int step);

template void OpenDecoder<OperationType::FP32>::masked_multi_head_attention(const DataType_* from_tensor,
                                                       DataType_* key_cache_,
                                                       DataType_* value_cache_,
                                                       DataType_* position_key_,
                                                       DataType_* position_value_,
                                                       DataType_* decoder_output,
                                                       const int step);

template void OpenDecoder<OperationType::FP32>::embedding_relative_position(DataType_* position_key_out,
                                                       DataType_* position_value_out,
                                                       const int step);

template void OpenDecoder<OperationType::FP32>::forward(const DataType_ *from_tensor, const DataType_ *memory_tensor,
                                                      DataType_ *key_cache_, DataType_ *value_cache_,
                                                      DataType_ *key_mem_cache_, DataType_ *value_mem_cache_,
                                                      DataType_ *decoder_output, const int step);

}
