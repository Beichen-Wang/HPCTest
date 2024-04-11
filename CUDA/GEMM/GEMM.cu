// #include "./GEMM.hpp"
#include "cuda_runtime.h"
#include "util.cuh"
#include "base.hpp"
#include <stdio.h>
#include <iostream>

// 定义宏：CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(-1); \
    } \
} while (0)

__global__ void gemmKernel1(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K){
    unsigned int m = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int n = threadIdx.y + blockIdx.y * blockDim.y;
    if(m >= M || n >= N){
        return;
    }
    float tc = 0;
#pragma unroll
    for(int k = 0; k < K; k++){
        tc += a[k * M + m] * b[n * K + k];
    }
    c[n * M + m] = alpha * tc + beta * c[n * M + m];
}

__global__ void gemmKernel2(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K){
  unsigned int m = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
  unsigned int n = (threadIdx.y + blockDim.y * blockIdx.y) * 4;
  if(m >= M || n >= N){
      return;
  }

  float4 tc_zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 tc[4] = {tc_zero,tc_zero,tc_zero,tc_zero};

  for (unsigned k = 0; k < K; ++k) {
    float4 fragmentA = *(const float4 *)(a + k * M + m);
    float4 fragmentB = make_float4(*(b + n * K + k), *(b + (n + 1) * K + k),*(b + (n + 2) * K + k), *(b + (n + 3) * K + k));
    
    tc[0].x += fragmentA.x * fragmentB.x;
    tc[0].y += fragmentA.y * fragmentB.x;
    tc[0].z += fragmentA.z * fragmentB.x;
    tc[0].w += fragmentA.w * fragmentB.x;

    tc[1].x += fragmentA.x * fragmentB.y;
    tc[1].y += fragmentA.y * fragmentB.y;
    tc[1].z += fragmentA.z * fragmentB.y;
    tc[1].w += fragmentA.w * fragmentB.y;

    tc[2].x += fragmentA.x * fragmentB.z;
    tc[2].y += fragmentA.y * fragmentB.z;
    tc[2].z += fragmentA.z * fragmentB.z;
    tc[2].w += fragmentA.w * fragmentB.z;

    tc[3].x += fragmentA.x * fragmentB.w;
    tc[3].y += fragmentA.y * fragmentB.w;
    tc[3].z += fragmentA.z * fragmentB.w;
    tc[3].w += fragmentA.w * fragmentB.w;
  }

    for(int i = 0; i < 4; i++){
        tc[i].x = alpha * tc[i].x;
        tc[i].y = alpha * tc[i].y;
        tc[i].z = alpha * tc[i].z;
        tc[i].w = alpha * tc[i].w;
    }

    float4 * f4c = reinterpret_cast<float4 *>(c);
    
  for(int i = 0; i < 4; i++){
    f4c[((n + i) * M + m)/4] = tc[i];
  }
}

__global__ void gemmKernel3(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K){
  unsigned int m = (threadIdx.x + blockDim.x * blockIdx.x) * 8;
  unsigned int n = (threadIdx.y + blockDim.y * blockIdx.y) * 8;
  if(m >= M || n >= N){
      return;
  }
  float4 tc_zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 ftc[4] = {tc_zero,tc_zero,tc_zero,tc_zero};
  float4 tc[4][4] = {ftc[4], ftc[4], ftc[4], ftc[4]};
  float4 * f4c = reinterpret_cast<float4 *>(c);

#pragma unroll
    for(int i = 0; i < 8; i += 4){
#pragma unroll
        for(int j = 0; j < 8; j += 4){
            int index = (i * 2 + j) / 4;
#pragma unroll
            for (unsigned k = 0; k < K; ++k) {
                float4 fragmentA = *(const float4 *)(a + k * M + m + j);
                float4 fragmentB = make_float4(*(b + (n + i) * K + k), *(b + (n + i + 1) * K + k),*(b + (n + i + 2) * K + k), *(b + (n + i + 3) * K + k));
                tc[index][0].x += fragmentA.x * fragmentB.x;
                tc[index][0].y += fragmentA.y * fragmentB.x;
                tc[index][0].z += fragmentA.z * fragmentB.x;
                tc[index][0].w += fragmentA.w * fragmentB.x;

                tc[index][1].x += fragmentA.x * fragmentB.y;
                tc[index][1].y += fragmentA.y * fragmentB.y;
                tc[index][1].z += fragmentA.z * fragmentB.y;
                tc[index][1].w += fragmentA.w * fragmentB.y;

                tc[index][2].x += fragmentA.x * fragmentB.z;
                tc[index][2].y += fragmentA.y * fragmentB.z;
                tc[index][2].z += fragmentA.z * fragmentB.z;
                tc[index][2].w += fragmentA.w * fragmentB.z;

                tc[index][3].x += fragmentA.x * fragmentB.w;
                tc[index][3].y += fragmentA.y * fragmentB.w;
                tc[index][3].z += fragmentA.z * fragmentB.w;
                tc[index][3].w += fragmentA.w * fragmentB.w;
            }
            for(int k = 0; k < 4; k++){
                tc[index][k].x = alpha * tc[index][k].x;
                tc[index][k].y = alpha * tc[index][k].y;
                tc[index][k].z = alpha * tc[index][k].z;
                tc[index][k].w = alpha * tc[index][k].w;
                f4c[((n + k + i) * M + m + j)/4] = tc[index][k];
            }
        }
    }
}

__global__ void gemmKernel4(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K){
  float4 tc[4] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
  float4 * f4c = reinterpret_cast<float4 *>(c);

  __shared__ float4 shemA[32][32];
  __shared__ float4 shemB[32][32];
  float4 fragmentA;
  float4 fragmentB;

    for (unsigned int k = 0; k < K; k += 32) {
        shemA[threadIdx.y][threadIdx.x] = *reinterpret_cast<const float4 *>(a + (k + threadIdx.y) * M + blockIdx.x * 128 + threadIdx.x * 4);
        shemB[threadIdx.y][threadIdx.x] = make_float4(*(b + (blockIdx.y * 128 + threadIdx.y * 4) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 1) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 2) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 3) * K + k + threadIdx.x));
        __syncthreads(); 

    for(int i = 0; i < 32; i++){
        fragmentA = shemA[i][threadIdx.x];
        fragmentB = shemB[threadIdx.y][i];
        
        tc[0].x += fragmentA.x * fragmentB.x;
        tc[0].y += fragmentA.y * fragmentB.x;
        tc[0].z += fragmentA.z * fragmentB.x;
        tc[0].w += fragmentA.w * fragmentB.x;

        tc[1].x += fragmentA.x * fragmentB.y;
        tc[1].y += fragmentA.y * fragmentB.y;
        tc[1].z += fragmentA.z * fragmentB.y;
        tc[1].w += fragmentA.w * fragmentB.y;

        tc[2].x += fragmentA.x * fragmentB.z;
        tc[2].y += fragmentA.y * fragmentB.z;
        tc[2].z += fragmentA.z * fragmentB.z;
        tc[2].w += fragmentA.w * fragmentB.z;

        tc[3].x += fragmentA.x * fragmentB.w;
        tc[3].y += fragmentA.y * fragmentB.w;
        tc[3].z += fragmentA.z * fragmentB.w;
        tc[3].w += fragmentA.w * fragmentB.w;

        }
    
        __syncthreads();
    }

    for(int i = 0; i < 4; i++){
        tc[i].x = alpha * tc[i].x;
        tc[i].y = alpha * tc[i].y;
        tc[i].z = alpha * tc[i].z;
        tc[i].w = alpha * tc[i].w;
    }

    if (blockIdx.y * 128 + threadIdx.y * 4 >= M || blockIdx.x * 128 + threadIdx.x * 4 >= N){
        return;
    } 
    for(int i = 0; i < 4; i++){
        f4c[((blockIdx.y * 128 + threadIdx.y * 4 + i) * M + blockIdx.x * 128 + threadIdx.x * 4) / 4] = tc[i];
    }
}

__global__ void gemmKernel5(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K){
    float4 tc[4] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
    float4 * f4c = reinterpret_cast<float4 *>(c);
    extern __shared__ float4 sharedMatrix[];

    float4 * shemA = sharedMatrix;
    float4 * shemB = sharedMatrix + 32 * 2 * 32;

    // __shared__ float4 shemA[32 * 2][32];
    // __shared__ float4 shemB[32][32 * 2];
    float4 fragmentA;
    float4 fragmentB;

    int A_SM_OFFSET = 0;
    int B_SM_OFFSET = 0;

    shemA[(threadIdx.y + A_SM_OFFSET) * 32 + threadIdx.x] = *reinterpret_cast<const float4 *>(a + threadIdx.y * M + blockIdx.x * 128 + threadIdx.x * 4);
    shemB[threadIdx.y * 32 * 2 + threadIdx.x + B_SM_OFFSET] = make_float4(*(b + (blockIdx.y * 128 + threadIdx.y * 4) * K  + threadIdx.x),
        *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 1) * K + threadIdx.x),
        *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 2) * K + threadIdx.x),
        *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 3) * K + threadIdx.x));
    __syncthreads(); 

    A_SM_OFFSET ^= 32;
    B_SM_OFFSET ^= 32;

    for (unsigned int k = 32; k < K; k += 32) {
        shemA[(threadIdx.y + A_SM_OFFSET) * 32 + threadIdx.x] = *reinterpret_cast<const float4 *>(a + (k + threadIdx.y) * M + blockIdx.x * 128 + threadIdx.x * 4);
        shemB[threadIdx.y * 32 * 2 + threadIdx.x + B_SM_OFFSET] = make_float4(*(b + (blockIdx.y * 128 + threadIdx.y * 4) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 1) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 2) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 3) * K + k + threadIdx.x));

        A_SM_OFFSET ^= 32;
        B_SM_OFFSET ^= 32;

    for(int i = 0; i < 32; i++){
        fragmentA = shemA[(i + A_SM_OFFSET) * 32 + threadIdx.x];
        fragmentB = shemB[threadIdx.y * 32 * 2 + i + B_SM_OFFSET];
        
        tc[0].x += fragmentA.x * fragmentB.x;
        tc[0].y += fragmentA.y * fragmentB.x;
        tc[0].z += fragmentA.z * fragmentB.x;
        tc[0].w += fragmentA.w * fragmentB.x;

        tc[1].x += fragmentA.x * fragmentB.y;
        tc[1].y += fragmentA.y * fragmentB.y;
        tc[1].z += fragmentA.z * fragmentB.y;
        tc[1].w += fragmentA.w * fragmentB.y;

        tc[2].x += fragmentA.x * fragmentB.z;
        tc[2].y += fragmentA.y * fragmentB.z;
        tc[2].z += fragmentA.z * fragmentB.z;
        tc[2].w += fragmentA.w * fragmentB.z;

        tc[3].x += fragmentA.x * fragmentB.w;
        tc[3].y += fragmentA.y * fragmentB.w;
        tc[3].z += fragmentA.z * fragmentB.w;
        tc[3].w += fragmentA.w * fragmentB.w;

        }
    
        __syncthreads();
    }

    A_SM_OFFSET ^= 32;
    B_SM_OFFSET ^= 32;

    for(int i = 0; i < 32; i++){
        fragmentA = shemA[(i + A_SM_OFFSET) * 32 + threadIdx.x];
        fragmentB = shemB[threadIdx.y * 32 * 2 + i + B_SM_OFFSET];
        
        tc[0].x += fragmentA.x * fragmentB.x;
        tc[0].y += fragmentA.y * fragmentB.x;
        tc[0].z += fragmentA.z * fragmentB.x;
        tc[0].w += fragmentA.w * fragmentB.x;

        tc[1].x += fragmentA.x * fragmentB.y;
        tc[1].y += fragmentA.y * fragmentB.y;
        tc[1].z += fragmentA.z * fragmentB.y;
        tc[1].w += fragmentA.w * fragmentB.y;

        tc[2].x += fragmentA.x * fragmentB.z;
        tc[2].y += fragmentA.y * fragmentB.z;
        tc[2].z += fragmentA.z * fragmentB.z;
        tc[2].w += fragmentA.w * fragmentB.z;

        tc[3].x += fragmentA.x * fragmentB.w;
        tc[3].y += fragmentA.y * fragmentB.w;
        tc[3].z += fragmentA.z * fragmentB.w;
        tc[3].w += fragmentA.w * fragmentB.w;

        }

    for(int i = 0; i < 4; i++){
        tc[i].x = alpha * tc[i].x;
        tc[i].y = alpha * tc[i].y;
        tc[i].z = alpha * tc[i].z;
        tc[i].w = alpha * tc[i].w;
    }

    if (blockIdx.y * 128 + threadIdx.y * 4 >= M || blockIdx.x * 128 + threadIdx.x * 4 >= N){
        return;
    } 
    for(int i = 0; i < 4; i++){
        f4c[((blockIdx.y * 128 + threadIdx.y * 4 + i) * M + blockIdx.x * 128 + threadIdx.x * 4) / 4] = tc[i];
    }
}

void CallKernel1(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    dim3 block(32 , 32);
    dim3 grid((M  - 1) / block.x + 1, (N - 1)/ block.y + 1);
    gemmKernel1<<< grid, block >>>(deva, devb, devc, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}

void CallKernel2(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    dim3 block(32 , 32);
    dim3 grid((M /4 - 1) / block.x + 1, (N / 4 - 1)/ block.y + 1);
    gemmKernel2<<< grid, block >>>(deva, devb, devc, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}

void CallKernel3(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    dim3 block(16 , 16);
    dim3 grid((M /8 - 1) / block.x + 1, (N / 8 - 1)/ block.y + 1);
    gemmKernel3<<< grid, block >>>(deva, devb, devc, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}

void CallKernel4(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    dim3 block(32 , 32);
    dim3 grid((M - 1) / block.x + 1, (N - 1)/ block.y + 1);
    gemmKernel4<<< grid, block >>>(deva, devb, devc, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}

void CallKernel5(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    // cudaFuncSetCacheConfig(gemmKernel5, cudaFuncCachePreferShared);
    constexpr int sharedMemorySize = 32 * 32 * 4 * 4 * 4;
    cudaFuncSetAttribute(
        gemmKernel5,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemorySize);
    dim3 block(32 , 32);
    dim3 grid((M - 1) / block.x + 1, (N - 1)/ block.y + 1);
    gemmKernel5<<< grid, block, sharedMemorySize >>>(deva, devb, devc, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}