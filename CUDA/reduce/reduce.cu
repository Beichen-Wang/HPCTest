#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Dense>

#define FULL_MASK 0xffffffff

#define RunBenchmark(...) \
    do{ \
        cudaEvent_t start, stop; \
        cudaEventCreate(&start); \
        cudaEventCreate(&stop); \
        float msecTotal = 0; \
        int nIter = 1; \
        cudaEventRecord(start); \
        for (int run = 0 ; run < nIter; run ++ ) { \
            __VA_ARGS__; \
        } \
        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        cudaEventElapsedTime(&msecTotal, start, stop); \
        float msecPerMatrixMul = msecTotal / nIter; \
        printf( " Time= %.3f msec\n", msecPerMatrixMul); \
    } while(0)

__device__ __forceinline__ float WrapSum(float sum){
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return sum;
}

template<int BlOCK_SIZE, int NUM_PER_THREAD>
__global__ void ReduceSum(float* __restrict__ input, float* output) {
    int index = blockIdx.x * (BlOCK_SIZE * NUM_PER_THREAD) + threadIdx.x;
    float sum;
#pragma unroll
    for(int i = 0; i < NUM_PER_THREAD;i++){
        sum += input[index + i * BlOCK_SIZE];
    }
    __shared__ float Reduced[BlOCK_SIZE / 32];
    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;
    sum = WrapSum(sum);
    if(laneID == 0){
        Reduced[warpID] = sum;
    }
    __syncthreads();
    if(warpID == 0){
        sum = WrapSum(Reduced[laneID]);
    }
    if(threadIdx.x == 0){
        output[blockIdx.x] = sum;
    }
}

template<int BlOCK_SIZE, int NUM_PER_THREAD>
__global__ void ReduceSum2(float* __restrict__ input, float* output) {
    int index = blockIdx.x * (BlOCK_SIZE * NUM_PER_THREAD) + threadIdx.x * 4;
    float4 sum;
#pragma unroll
    for(int i = 0; i < NUM_PER_THREAD / 4;i++){
        sum.x += input[index + i * BlOCK_SIZE * 4];
        sum.y += input[index + i * BlOCK_SIZE * 4 + 1];
        sum.z += input[index + i * BlOCK_SIZE * 4 + 2];
        sum.w += input[index + i * BlOCK_SIZE * 4 + 3];
    }

    __shared__ float4 Reduced[BlOCK_SIZE / 32];
    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;
    sum.x = WrapSum(sum.x);
    sum.y = WrapSum(sum.y);
    sum.z = WrapSum(sum.z);
    sum.w = WrapSum(sum.w);

    if(laneID == 0){
        Reduced[warpID] = sum;
    }
    __syncthreads();
    if(warpID == 0){
        sum.x = WrapSum(Reduced[laneID].x);
        sum.y = WrapSum(Reduced[laneID].y);
        sum.z = WrapSum(Reduced[laneID].z);
        sum.w = WrapSum(Reduced[laneID].w);
    }
    if(threadIdx.x == 0){
        output[blockIdx.x] = sum.x + sum.y + sum.z + sum.w;
    }
}

int main() {
    const int M = 256;
    const int N = 102400;
    const int num_per_thread = 100;
    Eigen::Matrix<float, -1 , -1, Eigen::RowMajor> h_input(M, N);
    for (int i = 0; i < h_input.rows(); ++i) {
        h_input.row(i).setLinSpaced(h_input.cols(), i + 1, i + h_input.cols());
    }

    h_input.setRandom();

    // 分配GPU内存
    float *d_output, *d_input;
    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * sizeof(float));

    // 从主机复制数据到设备
    cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块大小和网格大小
    const int blockSize = N / num_per_thread;
    int gridSize =  M;

    // 调用核函数
    RunBenchmark({
        ReduceSum<blockSize, num_per_thread><<<gridSize, blockSize>>>(d_input, d_output);
    });
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 从设备复制结果回主机
    Eigen::MatrixXf h_output(M, 1);
    cudaMemcpy(h_output.data(), d_output, M * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果
    Eigen::MatrixXf h_to_check = h_input.rowwise().sum();

    if(h_output.isApprox(h_to_check)){
        std::cout << "结果相同" << std::endl;
    } else {
        std::cout << "结果不同" << std::endl;
        std::cout << "computed : " << h_output << std::endl;
        std::cout << "Origin : " << h_to_check << std::endl;
    }
    
    // 清理资源
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}