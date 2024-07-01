#include <cuda_runtime.h>
//lock_base

__device__ volatile int g_mutex;
__device__ void GpuSync(int blockNum){
    int tid = threadIdx.x;
    if(tid == 0){
        atomicAdd(&g_mutex, 1);
        while(g_mutex != blockNum){}
    }
    __syncthreads();
}

