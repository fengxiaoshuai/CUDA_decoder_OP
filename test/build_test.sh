nvcc -o test  test.cpp  decoding.cu  decoder.cu -lcublas -lcudart -lgtest -lpthread -arch=sm_70 --std=c++11
