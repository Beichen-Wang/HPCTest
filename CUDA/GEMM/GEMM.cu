// #include "./GEMM.hpp"
#include "cuda_runtime.h"
#include "util.cuh"
#include "base.hpp"
#include <stdio.h>
#include <iostream>
#include <stdarg.h>

inline __device__ void mma4_4(const float4 fragmentA, const float4 fragmentB, float4 * tc){
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
    // c[n * M + m] = alpha * tc + beta * c[n * M + m];
    c[n * M + m] = alpha * tc;
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
  float4 f4_zero = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

  __shared__ float4 shemA[32][32];
  __shared__ float4 shemB[32][32];
  float4 fragmentA;
  float4 fragmentB;

    for (unsigned int k = 0; k < K; k += 32) {
        bool validA = ((k + threadIdx.y) * M + blockIdx.x * 128 + threadIdx.x * 4) < M * K ? true : false;
        bool validB = ((blockIdx.y * 128 + threadIdx.y * 4) * K  + k + threadIdx.x) < N * K ? true : false;

        shemA[threadIdx.y][threadIdx.x] = validA ? *reinterpret_cast<const float4 *>(a + (k + threadIdx.y) * M + blockIdx.x * 128 + threadIdx.x * 4) : f4_zero;
        shemB[threadIdx.y][threadIdx.x] = validB ? make_float4(*(b + (blockIdx.y * 128 + threadIdx.y * 4) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 1) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 2) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 3) * K + k + threadIdx.x)) : f4_zero;
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

    if (blockIdx.y * 128 + threadIdx.y * 4 >= N || blockIdx.x * 128 + threadIdx.x * 4 >= M){
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
    float4 f4_zero = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

    float4 * shemA = sharedMatrix;
    float4 * shemB = sharedMatrix + 32 * 2 * 32;

    // __shared__ float4 shemA[32 * 2][32];
    // __shared__ float4 shemB[32][32 * 2];
    float4 fragmentA = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};
    float4 fragmentB = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

    int A_SM_OFFSET = 0;
    int B_SM_OFFSET = 0;

    bool validA = (threadIdx.y * M + blockIdx.x * 128 + threadIdx.x * 4) < M * K ? true : false;
    bool validB = ((blockIdx.y * 128 + threadIdx.y * 4) * K  + threadIdx.x) < N * K ? true : false;
    shemA[(threadIdx.y + A_SM_OFFSET) * 32 + threadIdx.x] = validA ? *reinterpret_cast<const float4 *>(a + threadIdx.y * M + blockIdx.x * 128 + threadIdx.x * 4) : f4_zero;
    shemB[threadIdx.y * 32 * 2 + threadIdx.x + B_SM_OFFSET] = validB ? make_float4(*(b + (blockIdx.y * 128 + threadIdx.y * 4) * K  + threadIdx.x),
        *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 1) * K + threadIdx.x),
        *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 2) * K + threadIdx.x),
        *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 3) * K + threadIdx.x)) : f4_zero;
    __syncthreads(); 

    A_SM_OFFSET ^= 32;
    B_SM_OFFSET ^= 32;

    for (unsigned int k = 32; k < K; k += 32) {        
        bool validA = ((k + threadIdx.y) * M + blockIdx.x * 128 + threadIdx.x * 4) < M * K ? true : false;
        bool validB = ((blockIdx.y * 128 + threadIdx.y * 4) * K  + k + threadIdx.x) < N * K ? true : false;

        shemA[(threadIdx.y + A_SM_OFFSET) * 32 + threadIdx.x] = validA ? *reinterpret_cast<const float4 *>(a + (k + threadIdx.y) * M + blockIdx.x * 128 + threadIdx.x * 4) : f4_zero;
        shemB[threadIdx.y * 32 * 2 + threadIdx.x + B_SM_OFFSET] = validB ? make_float4(*(b + (blockIdx.y * 128 + threadIdx.y * 4) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 1) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 2) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 3) * K + k + threadIdx.x)) : f4_zero;

        A_SM_OFFSET ^= 32;
        B_SM_OFFSET ^= 32;
        
        for(int i = 0; i < 32; i++){
            fragmentA = shemA[(i + A_SM_OFFSET) * 32 + threadIdx.x];
            fragmentB = shemB[threadIdx.y * 32 * 2 + i + B_SM_OFFSET];
            
            mma4_4(fragmentA, fragmentB, tc);
        }
        __syncthreads();
    }

    A_SM_OFFSET ^= 32;
    B_SM_OFFSET ^= 32;

    for(int i = 0; i < 32; i++){
        fragmentA = shemA[(i + A_SM_OFFSET) * 32 + threadIdx.x];
        fragmentB = shemB[threadIdx.y * 32 * 2 + i + B_SM_OFFSET];
        
        mma4_4(fragmentA, fragmentB, tc);
    }

    for(int i = 0; i < 4; i++){
        tc[i].x = alpha * tc[i].x;
        tc[i].y = alpha * tc[i].y;
        tc[i].z = alpha * tc[i].z;
        tc[i].w = alpha * tc[i].w;
    }
    if (blockIdx.y * 128 + threadIdx.y * 4 >= N || blockIdx.x * 128 + threadIdx.x * 4 >= M){
        return;
    } 

    #pragma unroll
    for(int i = 0; i < 4; i++){
        f4c[((blockIdx.y * 128 + threadIdx.y * 4 + i) * M + blockIdx.x * 128 + threadIdx.x * 4) / 4] = tc[i];
    }
}

__global__ void gemmKernel6(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K){
    float4 tc[4] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
    float4 * f4c = reinterpret_cast<float4 *>(c);
    extern __shared__ float4 sharedMatrix[];
    float4 f4_zero = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

    float4 * shemA = sharedMatrix;
    float4 * shemB = sharedMatrix + 32 * 2 * 32;

    // __shared__ float4 shemA[32 * 2][32];
    // __shared__ float4 shemB[32][32 * 2];
    float4 fragmentA[2] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
    float4 fragmentB[2] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};

    int A_SM_OFFSET = 0;
    int B_SM_OFFSET = 0;
    int SHARED_OFFSET = 0;

    bool validA = (threadIdx.y * M + blockIdx.x * 128 + threadIdx.x * 4) < M * K ? true : false;
    bool validB = ((blockIdx.y * 128 + threadIdx.y * 4) * K  + threadIdx.x) < N * K ? true : false;
    shemA[(threadIdx.y + A_SM_OFFSET) * 32 + threadIdx.x] = validA ? *reinterpret_cast<const float4 *>(a + threadIdx.y * M + blockIdx.x * 128 + threadIdx.x * 4) : f4_zero;
    shemB[threadIdx.y * 32 * 2 + threadIdx.x + B_SM_OFFSET] = validB ? make_float4(*(b + (blockIdx.y * 128 + threadIdx.y * 4) * K  + threadIdx.x),
        *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 1) * K + threadIdx.x),
        *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 2) * K + threadIdx.x),
        *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 3) * K + threadIdx.x)) : f4_zero;
    __syncthreads(); 

    A_SM_OFFSET ^= 32;
    B_SM_OFFSET ^= 32;

    for (unsigned int k = 32; k < K; k += 32) {        
        bool validA = ((k + threadIdx.y) * M + blockIdx.x * 128 + threadIdx.x * 4) < M * K ? true : false;
        bool validB = ((blockIdx.y * 128 + threadIdx.y * 4) * K  + k + threadIdx.x) < N * K ? true : false;

        shemA[(threadIdx.y + A_SM_OFFSET) * 32 + threadIdx.x] = validA ? *reinterpret_cast<const float4 *>(a + (k + threadIdx.y) * M + blockIdx.x * 128 + threadIdx.x * 4) : f4_zero;
        shemB[threadIdx.y * 32 * 2 + threadIdx.x + B_SM_OFFSET] = validB ? make_float4(*(b + (blockIdx.y * 128 + threadIdx.y * 4) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 1) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 2) * K + k + threadIdx.x),
            *(b + (blockIdx.y * 128 + threadIdx.y * 4 + 3) * K + k + threadIdx.x)) : f4_zero;

        A_SM_OFFSET ^= 32;
        B_SM_OFFSET ^= 32;
        
        fragmentA[SHARED_OFFSET] = shemA[(A_SM_OFFSET) * 32 + threadIdx.x];
        fragmentB[SHARED_OFFSET] = shemB[threadIdx.y * 32 * 2 + B_SM_OFFSET];

        SHARED_OFFSET ^= 1;
        for(int i = 1; i < 32; i++){
            fragmentA[SHARED_OFFSET] = shemA[(i + A_SM_OFFSET) * 32 + threadIdx.x];
            fragmentB[SHARED_OFFSET] = shemB[threadIdx.y * 32 * 2 + i + B_SM_OFFSET];
            SHARED_OFFSET ^= 1;
            
            tc[0].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].x;
            tc[0].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].x;
            tc[0].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].x;
            tc[0].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].x;

            tc[1].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].y;
            tc[1].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].y;
            tc[1].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].y;
            tc[1].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].y;

            tc[2].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].z;
            tc[2].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].z;
            tc[2].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].z;
            tc[2].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].z;

            tc[3].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].w;
            tc[3].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].w;
            tc[3].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].w;
            tc[3].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].w;

        }
    
        SHARED_OFFSET ^= 1;
        tc[0].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].x;
        tc[0].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].x;
        tc[0].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].x;
        tc[0].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].x;

        tc[1].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].y;
        tc[1].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].y;
        tc[1].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].y;
        tc[1].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].y;

        tc[2].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].z;
        tc[2].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].z;
        tc[2].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].z;
        tc[2].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].z;

        tc[3].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].w;
        tc[3].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].w;
        tc[3].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].w;
        tc[3].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].w;
        __syncthreads();
    }

    A_SM_OFFSET ^= 32;
    B_SM_OFFSET ^= 32;

    fragmentA[SHARED_OFFSET] = shemA[(A_SM_OFFSET) * 32 + threadIdx.x];
    fragmentB[SHARED_OFFSET] = shemB[threadIdx.y * 32 * 2 + B_SM_OFFSET];

    SHARED_OFFSET ^= 1;

    for(int i = 1; i < 32; i++){
        fragmentA[SHARED_OFFSET] = shemA[(i + A_SM_OFFSET) * 32 + threadIdx.x];
        fragmentB[SHARED_OFFSET] = shemB[threadIdx.y * 32 * 2 + i + B_SM_OFFSET];
        SHARED_OFFSET ^= 1;
        
        tc[0].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].x;
        tc[0].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].x;
        tc[0].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].x;
        tc[0].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].x;

        tc[1].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].y;
        tc[1].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].y;
        tc[1].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].y;
        tc[1].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].y;

        tc[2].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].z;
        tc[2].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].z;
        tc[2].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].z;
        tc[2].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].z;

        tc[3].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].w;
        tc[3].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].w;
        tc[3].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].w;
        tc[3].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].w;
    }

    SHARED_OFFSET ^= 1;
    tc[0].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].x;
    tc[0].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].x;
    tc[0].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].x;
    tc[0].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].x;

    tc[1].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].y;
    tc[1].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].y;
    tc[1].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].y;
    tc[1].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].y;

    tc[2].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].z;
    tc[2].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].z;
    tc[2].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].z;
    tc[2].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].z;

    tc[3].x += fragmentA[SHARED_OFFSET].x * fragmentB[SHARED_OFFSET].w;
    tc[3].y += fragmentA[SHARED_OFFSET].y * fragmentB[SHARED_OFFSET].w;
    tc[3].z += fragmentA[SHARED_OFFSET].z * fragmentB[SHARED_OFFSET].w;
    tc[3].w += fragmentA[SHARED_OFFSET].w * fragmentB[SHARED_OFFSET].w;

    for(int i = 0; i < 4; i++){
        tc[i].x = alpha * tc[i].x;
        tc[i].y = alpha * tc[i].y;
        tc[i].z = alpha * tc[i].z;
        tc[i].w = alpha * tc[i].w;
    }
    if (blockIdx.y * 128 + threadIdx.y * 4 >= N || blockIdx.x * 128 + threadIdx.x * 4 >= M){
        return;
    } 

    #pragma unroll
    for(int i = 0; i < 4; i++){
        f4c[((blockIdx.y * 128 + threadIdx.y * 4 + i) * M + blockIdx.x * 128 + threadIdx.x * 4) / 4] = tc[i];
    }
}

__global__ void gemmKernel7(float * a, float * b, float *c, float alpha, float beta, int M, int N, int K){
    float4 tc[4] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
    float4 * f4c = reinterpret_cast<float4 *>(c);
    extern __shared__ float sharedMatrix2[];
    float4 f4_zero = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

    float * shemA = sharedMatrix2;
    float * shemB = sharedMatrix2 + 32 * 2 * 32 * 4;

    int tid = threadIdx.y * 32 + threadIdx.x;

    const int A_TILE_PER_ROW = 128 / 4;
    const int B_TILE_PER_ROW = 32 / 4;

    const int A_ROW_START = tid / A_TILE_PER_ROW;
    const int A_COL = tid % A_TILE_PER_ROW * 4;

    const int B_ROW_START = tid / B_TILE_PER_ROW;
    const int B_COL = tid % B_TILE_PER_ROW * 4;

    float4 tempB[2];

    float4 fragmentA = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};
    float4 fragmentB = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

    int SM_OFFSET = 0;

    FETCH_FLOAT4(shemA[
        OFFSET(A_ROW_START, A_COL, 128)]) = FETCH_FLOAT4(a[OFFSET(
        A_ROW_START, A_COL + blockIdx.x * 128, M
    )]);
    tempB[0] = FETCH_FLOAT4(b[OFFSET(
        B_ROW_START + blockIdx.y * 128, B_COL , K
    )]);
    shemB[OFFSET(B_COL, B_ROW_START, 128)] = tempB[0].x;
    shemB[OFFSET(B_COL + 1, B_ROW_START, 128)] = tempB[0].y;
    shemB[OFFSET(B_COL + 2, B_ROW_START, 128)] = tempB[0].z;
    shemB[OFFSET(B_COL + 3, B_ROW_START, 128)] = tempB[0].w;

    __syncthreads(); 

    SM_OFFSET ^= 1;

    for (unsigned int k = 32; k < K; k += 32) {        
        FETCH_FLOAT4(shemA[
            OFFSET(A_ROW_START + SM_OFFSET * 32, A_COL, 128)]) = FETCH_FLOAT4(a[OFFSET(
            A_ROW_START + k, A_COL + blockIdx.x * 128, M
        )]);
        tempB[SM_OFFSET] = FETCH_FLOAT4(b[OFFSET(
            B_ROW_START + blockIdx.y * 128, B_COL + k , K
        )]);
        shemB[OFFSET(B_COL + SM_OFFSET * 32, B_ROW_START, 128)] = tempB[SM_OFFSET].x;
        shemB[OFFSET(B_COL + 1 + SM_OFFSET * 32, B_ROW_START, 128)] = tempB[SM_OFFSET].y;
        shemB[OFFSET(B_COL + 2 + SM_OFFSET * 32, B_ROW_START, 128)] = tempB[SM_OFFSET].z;
        shemB[OFFSET(B_COL + 3 + SM_OFFSET * 32, B_ROW_START, 128)] = tempB[SM_OFFSET].w;

        SM_OFFSET ^= 1;
        
        for(int i = 0; i < 32; i++){
            fragmentA = FETCH_FLOAT4(shemA[OFFSET((i + SM_OFFSET * 32), threadIdx.x * 4, 128)]);
            fragmentB = FETCH_FLOAT4(shemB[OFFSET((i + SM_OFFSET * 32), threadIdx.y * 4, 128)]);
            
            mma4_4(fragmentA, fragmentB, tc);
        }
        __syncthreads();
    }

    SM_OFFSET ^= 1;

    for(int i = 0; i < 32; i++){
        fragmentA = FETCH_FLOAT4(shemA[OFFSET((i + SM_OFFSET * 32), threadIdx.x * 4, 128)]);
        fragmentB = FETCH_FLOAT4(shemB[OFFSET((i + SM_OFFSET * 32), threadIdx.y * 4, 128)]);
        
        mma4_4(fragmentA, fragmentB, tc);
    }

    for(int i = 0; i < 4; i++){
        tc[i].x = alpha * tc[i].x;
        tc[i].y = alpha * tc[i].y;
        tc[i].z = alpha * tc[i].z;
        tc[i].w = alpha * tc[i].w;
    }

    // if (blockIdx.y * 128 + threadIdx.y * 4 >= M || blockIdx.x * 128 + threadIdx.x * 4 >= N){
    //     return;
    // } 

    #pragma unroll
    for(int i = 0; i < 4; i++){
        f4c[((blockIdx.y * 128 + threadIdx.y * 4 + i) * M + blockIdx.x * 128 + threadIdx.x * 4) / 4] = tc[i];
    }
}

__global__ void gemmKernel8(float * a, float * b, float *c, float alpha, float beta, int M, int N, int K){
    float4 tc[4] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
    extern __shared__ float sharedMatrix2[];
    float4 f4_zero = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

    float * shemA = sharedMatrix2;
    float * shemB = sharedMatrix2 + 32 * 2 * 32 * 4;

    int tid = threadIdx.y * 32 + threadIdx.x;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const int b_shared_index = warp_id % 4 * 32 + lane_id % 8 * 4;
    const int a_shared_index = warp_id / 4 * 16 + lane_id / 8 * 4;

    const int A_TILE_PER_ROW = 128 / 4;
    const int B_TILE_PER_ROW = 32 / 4;

    const int A_ROW_START = tid / A_TILE_PER_ROW;
    const int A_COL = tid % A_TILE_PER_ROW * 4;

    const int B_ROW_START = tid / B_TILE_PER_ROW;
    const int B_COL = tid % B_TILE_PER_ROW * 4;

    float4 tempB[2];

    float4 fragmentA = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};
    float4 fragmentB = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

    int SM_OFFSET = 0;

    FETCH_FLOAT4(shemA[
        OFFSET(A_ROW_START, A_COL, 128)]) = FETCH_FLOAT4(a[OFFSET(
        A_ROW_START, A_COL + blockIdx.x * 128, M
    )]);
    tempB[0] = FETCH_FLOAT4(b[OFFSET(
        B_ROW_START + blockIdx.y * 128, B_COL , K
    )]);
    shemB[OFFSET(B_COL, B_ROW_START, 128)] = tempB[0].x;
    shemB[OFFSET(B_COL + 1, B_ROW_START, 128)] = tempB[0].y;
    shemB[OFFSET(B_COL + 2, B_ROW_START, 128)] = tempB[0].z;
    shemB[OFFSET(B_COL + 3, B_ROW_START, 128)] = tempB[0].w;

    __syncthreads(); 

    SM_OFFSET ^= 1;

    for (unsigned int k = 32; k < K; k += 32) {        
        FETCH_FLOAT4(shemA[
            OFFSET(A_ROW_START + SM_OFFSET * 32, A_COL, 128)]) = FETCH_FLOAT4(a[OFFSET(
            A_ROW_START + k, A_COL + blockIdx.x * 128, M
        )]);
        tempB[SM_OFFSET] = FETCH_FLOAT4(b[OFFSET(
            B_ROW_START + blockIdx.y * 128, B_COL + k , K
        )]);
        shemB[OFFSET(B_COL + SM_OFFSET * 32, B_ROW_START, 128)] = tempB[SM_OFFSET].x;
        shemB[OFFSET(B_COL + 1 + SM_OFFSET * 32, B_ROW_START, 128)] = tempB[SM_OFFSET].y;
        shemB[OFFSET(B_COL + 2 + SM_OFFSET * 32, B_ROW_START, 128)] = tempB[SM_OFFSET].z;
        shemB[OFFSET(B_COL + 3 + SM_OFFSET * 32, B_ROW_START, 128)] = tempB[SM_OFFSET].w;

        SM_OFFSET ^= 1;
        
        #pragma unroll
        for(int i = 0; i < 32; i++){
            fragmentA = FETCH_FLOAT4(shemA[OFFSET((i + SM_OFFSET * 32), a_shared_index, 128)]);
            fragmentB = FETCH_FLOAT4(shemB[OFFSET((i + SM_OFFSET * 32), b_shared_index, 128)]);
            
            mma4_4(fragmentA, fragmentB, tc);
        }
        __syncthreads();
    }

    SM_OFFSET ^= 1;

    for(int i = 0; i < 32; i++){
        fragmentA = FETCH_FLOAT4(shemA[OFFSET((i + SM_OFFSET * 32), a_shared_index, 128)]);
        fragmentB = FETCH_FLOAT4(shemB[OFFSET((i + SM_OFFSET * 32), b_shared_index, 128)]);
        
        mma4_4(fragmentA, fragmentB, tc);
    }

#pragma unroll
    for(int i = 0; i < 4; i++){
        tc[i].x = alpha * tc[i].x;
        tc[i].y = alpha * tc[i].y;
        tc[i].z = alpha * tc[i].z;
        tc[i].w = alpha * tc[i].w;
    }

#pragma unroll
    for(int i = 0; i < 4; i++){
        FETCH_FLOAT4(c[OFFSET(
            blockIdx.y * 128 + b_shared_index + i,
            blockIdx.x * 128 + a_shared_index,
            M
        )]) = tc[i];
    }
}

__global__ void gemmKernel9(float * a, float * b, float *c, float alpha, float beta, int M, int N, int K){
    float4 tc[4][4] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
    float4 f4_zero = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

    const int TILE_K = 8;

    __shared__ float shemA[2 * TILE_K * 128];
    __shared__ float shemB[2 * TILE_K * 128];

    int tid = threadIdx.y * 16 + threadIdx.x;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const int b_shared_index = warp_id % 2 * 32 + lane_id % 8 * 4;
    const int a_shared_index = warp_id / 2 * 16 + lane_id / 8 * 4;

    const int A_TILE_PER_ROW = 128 / 4;
    const int B_TILE_PER_ROW = TILE_K / 4;

    const int A_ROW_START = tid / A_TILE_PER_ROW;
    const int A_COL = tid % A_TILE_PER_ROW * 4;

    const int B_ROW_START = tid / B_TILE_PER_ROW;
    const int B_COL = tid % B_TILE_PER_ROW * 4;

    float4 tempB[2];

    float4 fragmentA[2] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
    float4 fragmentB[2] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};

    int SM_OFFSET = 0;

    FETCH_FLOAT4(shemA[
        OFFSET(A_ROW_START, A_COL, 128)]) = FETCH_FLOAT4(a[OFFSET(
        A_ROW_START, A_COL + blockIdx.x * 128, M
    )]);
    tempB[0] = FETCH_FLOAT4(b[OFFSET(
        B_ROW_START + blockIdx.y * 128, B_COL , K
    )]);
    shemB[OFFSET(B_COL, B_ROW_START, 128)] = tempB[0].x;
    shemB[OFFSET(B_COL + 1, B_ROW_START, 128)] = tempB[0].y;
    shemB[OFFSET(B_COL + 2, B_ROW_START, 128)] = tempB[0].z;
    shemB[OFFSET(B_COL + 3, B_ROW_START, 128)] = tempB[0].w;

    __syncthreads(); 

    SM_OFFSET ^= 1;

    for (unsigned int k = TILE_K; k < K; k += TILE_K) {        
        FETCH_FLOAT4(shemA[
            OFFSET(A_ROW_START + SM_OFFSET * TILE_K, A_COL, 128)]) = FETCH_FLOAT4(a[OFFSET(
            A_ROW_START + k, A_COL + blockIdx.x * 128, M
        )]);
        tempB[SM_OFFSET] = FETCH_FLOAT4(b[OFFSET(
            B_ROW_START + blockIdx.y * 128, B_COL + k , K
        )]);
        shemB[OFFSET(B_COL + SM_OFFSET * TILE_K, B_ROW_START, 128)] = tempB[SM_OFFSET].x;
        shemB[OFFSET(B_COL + 1 + SM_OFFSET * TILE_K, B_ROW_START, 128)] = tempB[SM_OFFSET].y;
        shemB[OFFSET(B_COL + 2 + SM_OFFSET * TILE_K, B_ROW_START, 128)] = tempB[SM_OFFSET].z;
        shemB[OFFSET(B_COL + 3 + SM_OFFSET * TILE_K, B_ROW_START, 128)] = tempB[SM_OFFSET].w;

        SM_OFFSET ^= 1;
        
        for(int i = 0; i < TILE_K; i++){
            fragmentA[0] = FETCH_FLOAT4(shemA[OFFSET((i + SM_OFFSET * TILE_K), a_shared_index, 128)]);
            fragmentA[1] = FETCH_FLOAT4(shemA[OFFSET((i + SM_OFFSET * TILE_K), a_shared_index + 64, 128)]);
            fragmentB[0] = FETCH_FLOAT4(shemB[OFFSET((i + SM_OFFSET * TILE_K), b_shared_index, 128)]);
            fragmentB[1] = FETCH_FLOAT4(shemB[OFFSET((i + SM_OFFSET * TILE_K), b_shared_index + 64, 128)]);
            
            mma4_4(fragmentA[0], fragmentB[0], tc[0]);
            mma4_4(fragmentA[0], fragmentB[1], tc[1]);
            mma4_4(fragmentA[1], fragmentB[0], tc[2]);
            mma4_4(fragmentA[1], fragmentB[1], tc[3]);
        }
        __syncthreads();
    }

    SM_OFFSET ^= 1;

    for(int i = 0; i < TILE_K; i++){
        fragmentA[0] = FETCH_FLOAT4(shemA[OFFSET((i + SM_OFFSET * TILE_K), a_shared_index, 128)]);
        fragmentA[1] = FETCH_FLOAT4(shemA[OFFSET((i + SM_OFFSET * TILE_K), a_shared_index + 64, 128)]);
        fragmentB[0] = FETCH_FLOAT4(shemB[OFFSET((i + SM_OFFSET * TILE_K), b_shared_index, 128)]);
        fragmentB[1] = FETCH_FLOAT4(shemB[OFFSET((i + SM_OFFSET * TILE_K), b_shared_index + 64, 128)]);
        
        mma4_4(fragmentA[0], fragmentB[0], tc[0]);
        mma4_4(fragmentA[0], fragmentB[1], tc[1]);
        mma4_4(fragmentA[1], fragmentB[0], tc[2]);
        mma4_4(fragmentA[1], fragmentB[1], tc[3]);
    }

    #pragma unroll
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            tc[i][j].x = alpha * tc[i][j].x;
            tc[i][j].y = alpha * tc[i][j].y;
            tc[i][j].z = alpha * tc[i][j].z;
            tc[i][j].w = alpha * tc[i][j].w;
        }
    }

    for(int i = 0; i < 4; i++){
        FETCH_FLOAT4(c[OFFSET(
            blockIdx.y * 128 + b_shared_index + i,
            blockIdx.x * 128 + a_shared_index,
            M
        )]) = tc[0][i];
    }

    for(int i = 0; i < 4; i++){
        FETCH_FLOAT4(c[OFFSET(
            blockIdx.y * 128 + b_shared_index + 64 + i,
            blockIdx.x * 128 + a_shared_index,
            M
        )]) =  tc[1][i];
    }
    for(int i = 0; i < 4; i++){
        FETCH_FLOAT4(c[OFFSET(
            blockIdx.y * 128 + b_shared_index + i,
            blockIdx.x * 128 + a_shared_index + 64,
            M
        )]) = tc[2][i];
    }
    for(int i = 0; i < 4; i++){
        FETCH_FLOAT4(c[OFFSET(
            blockIdx.y * 128 + b_shared_index + 64 + i,
            blockIdx.x * 128 + a_shared_index + 64,
            M
        )]) = tc[3][i];
    }
}

__global__ void gemmKernel10(float * a, float * b, float *c, float alpha, float beta, int M, int N, int K){
    float4 tc[4][4] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
    float4 * f4c = reinterpret_cast<float4 *>(c);
    float4 f4_zero = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};

    const int TILE_K = 8;

    __shared__ float4 shemA[TILE_K * 2][32];
    __shared__ float4 shemB[32][TILE_K * 2];
    float4 fragmentA[2] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
    float4 fragmentB[2] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};

    int A_SM_OFFSET = 0;
    int B_SM_OFFSET = 0;

    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int A_COL = tid % 32;
    const int A_ROW = tid / 32;
    const int B_COL = tid % TILE_K;
    const int B_ROW = tid / TILE_K;
    const int a_shared_index = tid % 16;
    const int b_shared_index = tid / 16;

    bool validA = blockIdx.x * 128 + A_COL * 4 < M;

    shemA[A_ROW + A_SM_OFFSET][A_COL] = validA ? FETCH_FLOAT4(a[OFFSET(
            A_ROW,
            blockIdx.x * 128 + A_COL * 4,
            M
        )]) : f4_zero;
    shemB[B_ROW][B_COL + B_SM_OFFSET] = make_float4(*(b + (blockIdx.y * 128 + B_ROW * 4) * K  + B_COL),
        *(b + (blockIdx.y * 128 + B_ROW * 4 + 1) * K + B_COL),
        *(b + (blockIdx.y * 128 + B_ROW * 4 + 2) * K + B_COL),
        *(b + (blockIdx.y * 128 + B_ROW * 4 + 3) * K + B_COL));
    __syncthreads(); 

    A_SM_OFFSET ^= TILE_K;
    B_SM_OFFSET ^= TILE_K;

    for (unsigned int k = TILE_K; k < K; k += TILE_K) {
        shemA[A_ROW + A_SM_OFFSET][A_COL] = FETCH_FLOAT4(a[OFFSET(
            k + A_ROW,
            blockIdx.x * 128 + A_COL * 4,
            M
        )]);
        shemB[B_ROW][B_COL + B_SM_OFFSET] = make_float4(*(b + (blockIdx.y * 128 + B_ROW * 4) * K + k + B_COL),
            *(b + (blockIdx.y * 128 + B_ROW * 4 + 1) * K + k + B_COL),
            *(b + (blockIdx.y * 128 + B_ROW * 4 + 2) * K + k + B_COL),
            *(b + (blockIdx.y * 128 + B_ROW * 4 + 3) * K + k + B_COL));   

        A_SM_OFFSET ^= TILE_K;
        B_SM_OFFSET ^= TILE_K;
        
        for(int i = 0; i < TILE_K; i++){
            fragmentA[0] = shemA[(i + A_SM_OFFSET)][a_shared_index];
            fragmentA[1] = shemA[(i + A_SM_OFFSET)][a_shared_index + 16];
            fragmentB[0] = shemB[b_shared_index][i + B_SM_OFFSET];
            fragmentB[1] = shemB[b_shared_index + 16][i + B_SM_OFFSET];
            
            mma4_4(fragmentA[0], fragmentB[0], tc[0]);
            mma4_4(fragmentA[0], fragmentB[1], tc[1]);
            mma4_4(fragmentA[1], fragmentB[0], tc[2]);
            mma4_4(fragmentA[1], fragmentB[1], tc[3]);
        }
        __syncthreads();
    }

    A_SM_OFFSET ^= TILE_K;
    B_SM_OFFSET ^= TILE_K;

    for(int i = 0; i < TILE_K; i++){
        fragmentA[0] = shemA[(i + A_SM_OFFSET)][a_shared_index];
        fragmentA[1] = shemA[(i + A_SM_OFFSET)][a_shared_index + 16];
        fragmentB[0] = shemB[b_shared_index][i + B_SM_OFFSET];
        fragmentB[1] = shemB[b_shared_index + 16][i + B_SM_OFFSET];
        
        mma4_4(fragmentA[0], fragmentB[0], tc[0]);
        mma4_4(fragmentA[0], fragmentB[1], tc[1]);
        mma4_4(fragmentA[1], fragmentB[0], tc[2]);
        mma4_4(fragmentA[1], fragmentB[1], tc[3]);
    }

    #pragma unroll
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            tc[i][j].x = alpha * tc[i][j].x;
            tc[i][j].y = alpha * tc[i][j].y;
            tc[i][j].z = alpha * tc[i][j].z;
            tc[i][j].w = alpha * tc[i][j].w;
        }
    }
    // if (blockIdx.y * 128 + b_shared_index * 4 >= N || blockIdx.x * 128 + a_shared_index * 4 >= M){
    //     return;
    // } 

    for(int i = 0; i < 4; i++){
        f4c[OFFSET(
            blockIdx.y * 128 + b_shared_index * 4 + i,
            blockIdx.x * 32 + a_shared_index,
            M / 4)] = tc[0][i];
    }

    // if (blockIdx.y * 128 + b_shared_index * 4 + 64 >= N || blockIdx.x * 128 + a_shared_index * 4 >= M){
    //     return;
    // }

    for(int i = 0; i < 4; i++){
        f4c[OFFSET(
            blockIdx.y * 128 + b_shared_index * 4 + 64 + i,
            blockIdx.x * 32 + a_shared_index,
            M / 4)] = tc[1][i];
    }

    // if (blockIdx.y * 128 + b_shared_index * 4 >= N || blockIdx.x * 128 + a_shared_index * 4 + 64 >= M){
    //     return;
    // }

    for(int i = 0; i < 4; i++){
        f4c[OFFSET(
            blockIdx.y * 128 + b_shared_index * 4 + i,
            blockIdx.x * 32 + a_shared_index + 16,
            M / 4)] = tc[2][i];
    }

    // if (blockIdx.y * 128 + b_shared_index * 4 + 64 >= N || blockIdx.x * 128 + a_shared_index * 4 + 64 >= M){
    //     return;
    // }

    for(int i = 0; i < 4; i++){
        f4c[OFFSET(
            blockIdx.y * 128 + b_shared_index * 4 + 64 + i,
            blockIdx.x * 32 + a_shared_index + 16,
            M / 4)] = tc[3][i];
    }
}

template <
    const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void gemmKernelS(float * A, float * B, float *C, float alpha, float beta, int M, int N, int K){
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X];
    #pragma unroll
    for(int i=0; i<THREAD_SIZE_Y; i++){
        #pragma unroll
        for(int j=0; j<THREAD_SIZE_X; j++){
            accum[i][j]=0.0;
        }
    }
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];

    //load index of the tile
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int a_tile_index =  warp_id/2*16 + lane_id/8*4; //warp_id * 8 + (lane_id / 16)*4; // (warp_id/4)*32 + ((lane_id%16)/2)*4;
    const int b_tile_index =  warp_id%2*32 + lane_id%8*4; //(lane_id % 16) * 4; // (warp_id%4)*16 + (lane_id/16)*8 + (lane_id%2)*4;
    
    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }
    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N )]);
    }
    __syncthreads();
    
    // load A from shared memory to register
    FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[0][0][a_tile_index]);
    FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[0][0][a_tile_index + 64]);
    
    // load B from shared memory to register
    FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[0][0][b_tile_index]);
    FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[0][0][b_tile_index + 64]);
    
    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        // next tile index
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if(tile_idx< K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL, // col
                    N )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K - 1; ++j){
            // load next tile from shared mem to register 
            // load A from shared memory to register
            FETCH_FLOAT4(frag_a[(j+1)%2][0]) = FETCH_FLOAT4(As[load_stage_idx][(j+1)][a_tile_index]);
            FETCH_FLOAT4(frag_a[(j+1)%2][4]) = FETCH_FLOAT4(As[load_stage_idx][(j+1)][a_tile_index + 64]);
            // load B from shared memory to register
            FETCH_FLOAT4(frag_b[(j+1)%2][0]) = FETCH_FLOAT4(Bs[load_stage_idx][(j+1)][b_tile_index]);
            FETCH_FLOAT4(frag_b[(j+1)%2][4]) = FETCH_FLOAT4(Bs[load_stage_idx][(j+1)][b_tile_index + 64]);
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){
            // load A from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[load_stage_idx^1][0][a_tile_index]);
        FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[load_stage_idx^1][0][a_tile_index + 64]);
        // load B from shared memory to register
        FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][b_tile_index]);
        FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][b_tile_index + 64]);
        // compute C THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx< K);
    
    const int c_block_row = a_tile_index;
    const int c_block_col = b_tile_index;

    //store C00 block
    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + c_block_row + i,
        BLOCK_SIZE_N * bx + c_block_col,
        N)]) = FETCH_FLOAT4(accum[i][0]);
    }
    //store C01 block
    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + c_block_row + i,
        BLOCK_SIZE_N * bx + c_block_col + 64,
        N)]) = FETCH_FLOAT4(accum[i][4]);
    }
    //store C10 block
    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + c_block_row + 64 + i,
        BLOCK_SIZE_N * bx + c_block_col,
        N)]) = FETCH_FLOAT4(accum[i+4][0]);
    }
    //store C11 block
    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + c_block_row + 64 + i,
        BLOCK_SIZE_N * bx + c_block_col + 64,
        N)]) = FETCH_FLOAT4(accum[i+4][4]);
    }
}

void CallKernel1(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    dim3 block(32 , 32);
    dim3 grid((M  - 1) / block.x + 1, (N - 1)/ block.y + 1);
    printf(" Kernel1");
    RunBenchmark({
        gemmKernel1<<< grid, block>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernel2(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    dim3 block(32 , 32);
    dim3 grid((M /4 - 1) / block.x + 1, (N / 4 - 1)/ block.y + 1);
    printf(" Kernel2");
    RunBenchmark({
        gemmKernel2<<< grid, block>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernel3(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    // dim3 block(16 , 16);
    dim3 block(32 , 32);
    dim3 grid((M /8 - 1) / block.x + 1, (N / 8 - 1)/ block.y + 1);
    printf(" Kernel3");
    RunBenchmark({
        gemmKernel3<<< grid, block>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernel4(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    dim3 block(32 , 32);
    dim3 grid((M/4 - 1) / block.x + 1, (N/4 - 1)/ block.y + 1);
    printf(" Kernel4");
    RunBenchmark({
        gemmKernel4<<< grid, block>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernel5(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    // cudaFuncSetCacheConfig(gemmKernel5, cudaFuncCachePreferShared);
    constexpr int sharedMemorySize = 32 * 32 * 4 * 4 * 4;
    cudaFuncSetAttribute(
        gemmKernel5,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemorySize);
    dim3 block(32 , 32);
    dim3 grid((M/4 - 1) / block.x + 1, (N/4 - 1)/ block.y + 1);
    printf(" Kernel5");
    RunBenchmark({
        gemmKernel5<<< grid, block, sharedMemorySize>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernel6(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    constexpr int sharedMemorySize = 32 * 32 * 4 * 4 * 4;
    cudaFuncSetAttribute(
        gemmKernel6,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemorySize);
    dim3 block(32 , 32);
    dim3 grid((M/4 - 1) / block.x + 1, (N/4 - 1)/ block.y + 1);
    printf(" Kernel6");
    RunBenchmark({
        gemmKernel6<<< grid, block, sharedMemorySize>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernel7(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    constexpr int sharedMemorySize = 32 * 32 * 4 * 4 * 4;
    cudaFuncSetAttribute(
        gemmKernel7,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemorySize);
    dim3 block(32 , 32);
    dim3 grid((M/4 - 1) / block.x + 1, (N/4 - 1)/ block.y + 1);
    printf(" Kernel7");
    RunBenchmark({
        gemmKernel7<<< grid, block, sharedMemorySize>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernel8(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    constexpr int sharedMemorySize = 32 * 32 * 4 * 4 * 4;
    cudaFuncSetAttribute(
        gemmKernel8,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemorySize);
    dim3 block(32 , 32);
    dim3 grid((M/4 - 1) / block.x + 1, (N/4 - 1)/ block.y + 1);
    printf(" Kernel8");
    RunBenchmark({
        gemmKernel8<<< grid, block, sharedMemorySize>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernel9(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    dim3 block(16 , 16);
    dim3 grid((M/8 - 1) / block.x + 1, (N/8 - 1)/ block.y + 1);
    printf(" Kernel9");
    RunBenchmark({
        gemmKernel9<<< grid, block>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernel10(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
    dim3 block(16, 16);
    dim3 grid((M/8 - 1) / block.x + 1, (N/8 - 1)/ block.y + 1);
    printf(" Kernel10");
    RunBenchmark({
        gemmKernel10<<< grid, block>>>(deva, devb, devc, alpha, beta, M, N, K);
    });
    cudaDeviceSynchronize();
}

void CallKernelS(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K){
        const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    dim3 block(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 grid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    gemmKernelS<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
    <<< grid, block >>>(deva, devb, devc, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}