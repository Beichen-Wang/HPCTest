#pragma once
#include "cuda_runtime.h"

#define RunBenchmark(...) \
    do{ \
        cudaEvent_t start, stop; \
        cudaEventCreate(&start); \
        cudaEventCreate(&stop); \
        float msecTotal = 0; \
        int nIter = 100; \
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

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])