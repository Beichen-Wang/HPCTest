
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

__device__ float WrapSum(float sum){
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
}

template <int BLOCK_SIZE>
__global__ void GetDistance(float * input1, float * input2, float * output){
    int tid = threadIdx.x;
    __shared__ float shared[BLOCK_SIZE];
    shared[tid] = input1[tid] * input2[tid];
    __syncthread();
    float sum;
    sum = WrapSum(shared[tid]);
    int warpID = tid / 32;
    int laneID = tid % 32; 
    if(laneID == 0){
        shared[warpID] = sum;
    }
    __syncthreads();
    if(warpID == 0){
        if(laneID > BLOCK_SIZE / 32){
            shared[laneID] = 0;
        }
        sum = WrapSum(shared[laneID]);
    }
    if(tid == 0){
        output[blockIdx.x] = sum;
    }
}