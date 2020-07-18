 #nvcc -o test  main.cpp decode_kernel.cu -arch=sm_75 -lcublas -lcudart 
TF_INCLUDE=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIBRARY=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') 

nvcc -Xcompiler -fPIC -O3  -shared --expt-relaxed-constexpr -o decoding.so  decoding_op.cc decoding_op.cu.cc  decoding.cu  decoder.cu -lcublas -lcudart -arch=sm_70 --std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INCLUDE -I$TF_INCLUDE/external/nsync/public -L$TF_LIBRARY -ltensorflow_framework -DGOOGLE_CUDA=1 -DNDEBUG
