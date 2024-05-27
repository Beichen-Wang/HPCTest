
#include "cuda_runtime.h"

#define FULL_MASK 0xffffffff
__device__ float WrapReduceSum(float input){
    float sum;
    sum += __shfl_down_sync(FULL_MASK, input, 16);
    sum += __shfl_down_sync(FULL_MASK, input, 8);
    sum += __shfl_down_sync(FULL_MASK, input, 4);
    sum += __shfl_down_sync(FULL_MASK, input, 2);
    sum += __shfl_down_sync(FULL_MASK, input, 1);
    return sum;
}

__global__ void ReduceSum(const __restrict__ float* input, float * ouptut, int M, int N){
    float sum;

    __shared__ float sharedMem[M];
    sharedMem[threadIdx.x] = input[threadIdx.x * M + blockIdx.x];
    sum = WrapReduceSum(sharedMem[threadIdx.x]);
    int wrap_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if(lane_id == 0){
        sharedMem[wrap_id] = sum;
    }
    __syncthreads();
    if(wrap_id == 0){
        sum = WrapReduceSum(sharedMem[lane_id]);
    }
    if(threadIdx.x == 0){
        ouptut[blockIdx.x] = sum;
    }
}