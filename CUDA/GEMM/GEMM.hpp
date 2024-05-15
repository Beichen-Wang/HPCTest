#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include "cuda_runtime.h"
#include "base.hpp"
#include <cublas_v2.h>
#include <vector>
extern "C" {
    __global__ void gemmKernel1(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel2(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel3(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel4(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel5(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel6(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel7(float * a, float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel8(float * a, float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel9(float * a, float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel10(float * a, float * b, float *c, float alpha, float beta, int M, int N, int K);
}
    
void CallKernel1(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel2(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel3(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel3_1(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel3_2(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);

void CallKernel4(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel5(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel6(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel7(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel8(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel9(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel10(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernelS(float * deva, float * devb, float * devc, float alpha, float beta, int M, int N, int K);

#define CallKernelX(num, ...) CallKernel##num(__VA_ARGS__)

#define TIME_IT(func) \
    auto start = std::chrono::high_resolution_clock::now(); \
    (func); \
    auto end = std::chrono::high_resolution_clock::now(); \
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); \
    std::cout << "Function " #func " took " << duration << " milliseconds to execute." << std::endl;

class GEMM {
    private:
    Eigen::MatrixXf a;
    Eigen::MatrixXf b;
    Eigen::MatrixXf c;
    float alpha;
    float beta;
    int M;
    int N;
    int K;
    float * deva, * devb, * devc;
    cublasHandle_t handle;
    void (*kernelPointers[12])(float*, float*, float*, float, float, int, int, int) = {
        CallKernel1, CallKernel2, CallKernel3, CallKernel3_1, CallKernel3_2, CallKernel4, CallKernel5, CallKernel6, CallKernel7, CallKernel8, CallKernel9, CallKernel10};
    public:
    ~GEMM(){
        cudaFree(deva);
        cudaFree(devb);
        cudaFree(devc);
        cublasDestroy(handle);
    }
    GEMM(int _M, int _K, int _N):M(_M), N(_N), K(_K), alpha(1.0f),beta(0.0f){
        
        Eigen::VectorXf veca1(M * K / 2); 
        veca1.setLinSpaced(1, 1); 
        Eigen::VectorXf vecb1(N * K / 2); 
        vecb1.setLinSpaced(1, 1);

        Eigen::VectorXf veca2(M * K / 2); 
        veca2.setLinSpaced(2, 2); 
        Eigen::VectorXf vecb2(N * K / 2); 
        vecb2.setLinSpaced(2, 2);

        Eigen::VectorXf vecaa(M * K);
        Eigen::VectorXf vecbb(N * K);

        vecaa << veca1, veca2;
        vecbb << vecb1, vecb2;
        vecbb = vecaa;

        a = vecaa;
        b = vecbb;
        
        a.resize(M, K);
        b.resize(K, N);

        a.setRandom();
        b.setRandom();
        // a.setOnes();
        // b.setOnes();

        c = Eigen::MatrixXf::Zero(M, N);

        cudaMalloc(&deva, M * K * sizeof(float));
        cudaMemcpy(deva, a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&devb, K * N * sizeof(float));
        cudaMemcpy(devb, b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&devc, M * N * sizeof(float));
        cublasCreate(&handle);
    }

    Eigen::MatrixXf PureExcute(){
        c.setZero();
        for(int m = 0; m < M; m++){
            for(int n = 0; n < N; n++){
                float tc = 0;
                for(int k = 0; k < K; k++){
                    tc += a(m, k) * b(k, n);
                }
                c(m, n) = alpha * tc + beta * c(m, n);
            }
        }
        return c;
    }
    Eigen::MatrixXf EigenExcute(){
        c.setZero();
        c.noalias() = alpha * a * b + beta * c;
        return c;
    }

    Eigen::MatrixXf CUDAExecute(int i){
        c.setZero();
        cudaMemset(devc, 0, M * N * sizeof(float));
        // CallKernel3_2(deva, devb, devc, alpha, beta, M, N, K); 
        kernelPointers[i](deva, devb, devc, alpha, beta, M, N, K);
        cudaMemcpy(c.data(), devc, M * N * sizeof(float), cudaMemcpyDeviceToHost); 
        return c;
    }

    Eigen::MatrixXf CUBLASExecute(){
        c.setZero();
        cudaMemset(devc, 0, M * N * sizeof(float));
        printf(" cublas");
        RunBenchmark(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, deva, M, devb, K, &beta, devc, M));
        // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, deva, M, devb, K, &beta, devc, M); 
        cudaMemcpy(c.data(), devc, M * N * sizeof(float), cudaMemcpyDeviceToHost); 
        return c;
    }

     void checkAndPrintTiming() {
        std::vector<Eigen::MatrixXf> cudaResult(12);
        auto cublasResult = this->CUBLASExecute();
        for(int i = 0; i < 12; i++){
            cudaResult[i] = this->CUDAExecute(i);

            if (cublasResult.isApprox(cudaResult[i], 1e-6)) {
                // std::cout << "executions are the same." << std::endl;
            } else {
                std::cout << "Executions are different:" << std::endl;
                std::cout << "cublas Result: \n" << cublasResult << std::endl;
                std::cout << "cuda Result: \n" << cudaResult[i] << std::endl;
                continue;
            }
        }
    }
};