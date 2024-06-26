
// m * n
// 对每一行做softmax
// m * n

#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

inline __device__ WrapSum(float sum){
    sum += __shfl_down_sync(FULL_MASK, sum , 16);
    sum += __shfl_down_sync(FULL_MASK, sum , 8);
    sum += __shfl_down_sync(FULL_MASK, sum , 4);
    sum += __shfl_down_sync(FULL_MASK, sum , 2);
    sum += __shfl_down_sync(FULL_MASK, sum , 1);
}

inline __device__ WrapMax(float maxValue){
    maxValue = std::max(__shfl_down_sync(FULL_MASK, maxValue , 16), maxValue);
    maxValue = std::max(__shfl_down_sync(FULL_MASK, maxValue , 8), maxValue);
    maxValue = std::max(__shfl_down_sync(FULL_MASK, maxValue , 4), maxValue);
    maxValue = std::max(__shfl_down_sync(FULL_MASK, maxValue , 2), maxValue);
    maxValue = std::max(__shfl_down_sync(FULL_MASK, maxValue , 1), maxValue);
}


template <int BLOCK_SIZE>
__global__ void Softmax(float * input, float * output){
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    __shared__ float sme[BLOCK_SIZE / 32];
    __shared__ float smeMaxValue;
    __shared__ float smeSum;
    float maxValue = WrapMax(input[tid]);
    int wrapID = tid / 32;
    int laneID = tid % 32;
    if(laneID == 0){
        sme[wrapID] = maxValue;
    }

    __syncthreads();

    if(wrapID == 0){
        smeMaxValue = WrapMax(sme[laneID]);
    }
    __syncthreads();

    float sum;
    sum = wrapSum(exp(input[tid] - smeMaxValue));
    if(laneID == 0){
        sme[wrapID] = sum;
    }
    __syncthreads();
    if(wrapID == 0){
        smeSum = WrapSum(sme[laneID]);
    }
    __syncthreads();
    output[tid] = exp(input[tid] - smeMaxValue) / smeSum;
}
