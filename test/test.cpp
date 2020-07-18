#include "decoder.h"
#include "decoding.h"
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <gtest/gtest.h>
#include "load_data.hpp"

using namespace fastertransformer;
using namespace std;


class TestDecoding: public testing::Test
{
public:
	TestDecoding() 
	{
            struct cudaDeviceProp prop;
            check_cuda_error(cudaGetDeviceProperties(&prop, 0));
            printf("Device %s\n", prop.name);
            //agr
            //para
            int language_id[2] = {1,1};
            int mask_tmp[16] = {1,1,1,1,1,0,0,0, 1,1,1,1,1,1,0,0};
	    start_id = language_id;
            mask = mask_tmp;

            load(encode_out, "./weight/encode_out.txt");
            load(weight_embedding, "./weight/embedding.txt");
            load(language_embedding, "./weight/language_embedding.txt");
            load(weight_scale, "./weight/scale.txt");
            load(weight_bias, "./weight/bias.txt");
            load(logit_weight, "./weight/logit.txt");
            cout << "end load logits" << endl;

	    weight.resize(6);
            for(int i = 0; i<decoder_layers; i++)
            {
                  load_layer_weight(weight[i], i);
            }
  

	};
        void device_malloc(float** ptr, int size, float* h_ptr)
        {
            check_cuda_error(cudaMalloc((void**)ptr, size));
            check_cuda_error(cudaMemcpy(*ptr, h_ptr, size, cudaMemcpyHostToDevice));
        }
        void device_malloc(int** ptr, int size, int* h_ptr)
        {
            check_cuda_error(cudaMalloc((void**)ptr, size));
            check_cuda_error(cudaMemcpy(*ptr, h_ptr, size, cudaMemcpyHostToDevice));
        }

	void SetUp() override
	{
	       cublasHandle_t cublasHandle;
               check_cuda_error(cublasCreate(&cublasHandle));
               cudaStream_t stream;
               check_cuda_error(cudaStreamCreate(&stream));
               check_cuda_error(cublasSetStream(cublasHandle, stream));
	       fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
               param = new DecoderInitParam<float>[6];

               float h_bias[16] = {-1e9};
               for(int i=0; i<batch_size*seq_len; i++) {h_bias[i] = -1e9;};
               for(int i = 0; i < batch_size*length; i++){h_bias[i] *= (1-mask[i]);};
 

               cout << "start malloc for GPU" << endl;
               for(int i = 0; i < decoder_layers; i++)
               {
                 param[i].stream = stream;
                 param[i].cublas_handle = cublasHandle;

                 float *d_self_Q_kernel, *d_self_K_kernel, *d_self_V_kernel, *d_self_output_kernel, *d_self_gamma, *d_self_beta;
                 float *d_self_position_key, *d_self_position_value;
                 float *d_cross_Q_kernel, *d_cross_K_kernel, *d_cross_V_kernel, *d_cross_output_kernel,*d_cross_bias_kernel, *d_cross_gamma, *d_cross_beta;
                 float *d_ffn_kernel1, *d_ffn_bias1, *d_ffn_kernel2, *d_ffn_bias2, *d_ffn_gamma, *d_ffn_beta;

                 device_malloc(&d_self_gamma, sizeof(float) * hidden_units, weight[i][0].data());
                 device_malloc(&d_self_beta, sizeof(float) * hidden_units, weight[i][1].data());
                 device_malloc(&d_self_Q_kernel, sizeof(float) * hidden_units * hidden_units, weight[i][2].data());
                 device_malloc(&d_self_K_kernel, sizeof(float) * hidden_units * hidden_units, weight[i][3].data());
                 device_malloc(&d_self_V_kernel, sizeof(float) * hidden_units * hidden_units, weight[i][4].data());
                 device_malloc(&d_self_output_kernel, sizeof(float) * hidden_units * hidden_units, weight[i][5].data());
                 device_malloc(&d_self_position_key, sizeof(float) * (max_position*2+1) * size_per_head, weight[i][18].data());
                 device_malloc(&d_self_position_value, sizeof(float) * (max_position*2+1) * size_per_head, weight[i][19].data());

                 device_malloc(&d_cross_gamma, sizeof(float) * hidden_units, weight[i][6].data());
                 device_malloc(&d_cross_beta, sizeof(float) * hidden_units, weight[i][7].data());
                 device_malloc(&d_cross_Q_kernel, sizeof(float) * hidden_units * hidden_units, weight[i][8].data());
                 device_malloc(&d_cross_K_kernel, sizeof(float) * hidden_unit * hidden_units, weight[i][9].data());
                 device_malloc(&d_cross_V_kernel, sizeof(float) * hidden_unit * hidden_units, weight[i][10].data());
                 device_malloc(&d_cross_output_kernel, sizeof(float) * hidden_units * hidden_units, weight[i][11].data());
                 device_malloc(&d_cross_bias_kernel, sizeof(float) * batch_size * seq_len, h_bias);

                 device_malloc(&d_ffn_gamma, sizeof(float) * hidden_units, weight[i][12].data());
                 device_malloc(&d_ffn_beta, sizeof(float) * hidden_units, weight[i][13].data());
                 device_malloc(&d_ffn_kernel1, sizeof(float) * inner_size * hidden_units, weight[i][14].data());
                 device_malloc(&d_ffn_bias1, sizeof(float) * inner_size, weight[i][15].data());
                 device_malloc(&d_ffn_kernel2, sizeof(float) * inner_size * hidden_units, weight[i][16].data());
                 device_malloc(&d_ffn_bias2, sizeof(float) * hidden_units, weight[i][17].data());


                 param[i].self_layernorm.gamma = d_self_gamma;
                 param[i].self_layernorm.beta = d_self_beta;
                 param[i].self_attention.query_weight = d_self_Q_kernel;
                 param[i].self_attention.key_weight = d_self_K_kernel;
                 param[i].self_attention.value_weight = d_self_V_kernel;
                 param[i].self_attention.attention_output_weight = d_self_output_kernel;
                 param[i].self_attention.position_key = d_self_position_key;
                 param[i].self_attention.position_value = d_self_position_value;

                 param[i].cross_layernorm.gamma = d_cross_gamma;
                 param[i].cross_layernorm.beta = d_cross_beta;
                 param[i].cross_attention.query_weight = d_cross_Q_kernel;
                 param[i].cross_attention.key_weight = d_cross_K_kernel;
                 param[i].cross_attention.value_weight = d_cross_V_kernel;
                 param[i].cross_attention.attention_output_weight = d_cross_output_kernel;
                 param[i].cross_bias = d_cross_bias_kernel;

                 param[i].ffn_layernorm.gamma = d_ffn_gamma;
                 param[i].ffn_layernorm.beta = d_ffn_beta;
                 param[i].ffn.intermediate_weight.kernel = d_ffn_kernel1;
                 param[i].ffn.intermediate_weight.bias = d_ffn_bias1;
                 param[i].ffn.output_weight.kernel = d_ffn_kernel2;
                 param[i].ffn.output_weight.bias = d_ffn_bias2;
               }


               float *d_encodeout_tensor;
               float *d_embedding_table_init;
               float *d_embedding_table_run;
               float *d_gamma;
               float *d_beta;
               float *d_embedding_kernel;
               int* d_start_ids;

               int* d_output_ids;
               int* d_sequence_lengths;
               device_malloc(&d_encodeout_tensor, sizeof(float) * hidden_units * seq_len * batch_size , encode_out.data());
               device_malloc(&d_embedding_table_init, sizeof(float) * language_num  *hidden_units , language_embedding.data());
               device_malloc(&d_embedding_table_run, sizeof(float) * vocab_size * hidden_units , weight_embedding.data());
               device_malloc(&d_gamma, sizeof(float) * hidden_units, weight_scale.data());
               device_malloc(&d_beta, sizeof(float) * hidden_units, weight_bias.data());
               device_malloc(&d_embedding_kernel, sizeof(float) * hidden_units * vocab_size, logit_weight.data());
               device_malloc(&d_start_ids, sizeof(int) * batch_size, start_id);

               check_cuda_error(cudaMalloc((void**)&d_output_ids, sizeof(int) * (max_decode_length) * batch_size ));

               decoding_params.cublas_handle = cublasHandle;
               decoding_params.stream = stream;
               decoding_params.memory_tensor = d_encodeout_tensor;
               decoding_params.embedding_table_init = d_embedding_table_init;
               decoding_params.embedding_table_run = d_embedding_table_run;
               decoding_params.embedding_kernel = d_embedding_kernel;
               decoding_params.output_ids = d_output_ids;
               decoding_params.layernorm.gamma = d_gamma;
               decoding_params.layernorm.beta = d_beta;


               decoding = new DecodingOpenNMT<OperationType::FP32>(allocator, batch_size,max_decode_length, head_num, size_per_head,
                                         vocab_size, decoder_layers, hidden_unit, seq_len, d_start_ids, end_id);
	};

	void TearDown() override
	{
		delete decoding;
		delete param;
	};
	
	DecoderInitParam<float> *param; 
	DecodingInitParam<float> decoding_params;
	DecodingOpenNMT<OperationType::FP32>* decoding;
            const int batch_size = 2;
            const int head_num = 16;
            const int size_per_head = 64;
            const int vocab_size = 32768;
            const int length = 8;
            const int decoder_layers = 6;
            const int hidden_unit = 1024;
            const int decode_length = 17;
            const int language_num = 2;

	    int* mask;
            int* start_id;
            const int max_decode_length = 17;
            const int seq_len = 8;
            const int end_id = 2;
            const int hidden_units = 1024;
            const int inner_size = 4096;
            const int max_position = 20;

            vector<float> encode_out;

            vector<float> weight_embedding;

            vector<float> language_embedding;

            vector<float> weight_scale;

            vector<float> weight_bias;

            vector<float> logit_weight;

            vector<vector<vector<float>>> weight;
};


TEST_F(TestDecoding, decoding)
{
	decoding->forward(param, decoding_params);
        int* h_out_ids = new int[batch_size * max_decode_length];
        cudaMemcpy(h_out_ids, decoding_params.output_ids, sizeof(int) * batch_size * max_decode_length, cudaMemcpyDeviceToHost);
        EXPECT_EQ(h_out_ids[0],134);
        EXPECT_EQ(h_out_ids[16],7298);
        EXPECT_EQ(h_out_ids[17],127);
        EXPECT_EQ(h_out_ids[33],127);
        delete []h_out_ids;
}

int main(int argc, char** argv)
{
        testing::InitGoogleTest(&argc, argv); 
	return RUN_ALL_TESTS();
}
























