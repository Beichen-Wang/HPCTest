#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include "cuda_runtime.h"
#include <cublas_v2.h>
extern "C" {
    __global__ void gemmKernel1(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel2(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel3(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel4(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
    __global__ void gemmKernel5(const float * a, const float * b, float *c, float alpha, float beta, int M, int N, int K);
}
void CallKernel1(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel2(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel3(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel4(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K);
void CallKernel5(const float * deva, const float * devb, float * devc, float alpha, float beta, int M, int N, int K);
class GEMM {
    private:
    Eigen::MatrixXf a;
    Eigen::MatrixXf b;
    float alpha;
    float beta;
    int M;
    int N;
    int K;
    float * deva, * devb, * devc;
    cublasHandle_t handle;
    std::pair<double, Eigen::MatrixXf> executeAndTime(const std::function<Eigen::MatrixXf()>& func) {
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXf result = func();
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return {duration, result};
    }

    public:
    ~GEMM(){
        cudaFree(deva);
        cudaFree(devb);
        cudaFree(devc);
        cublasDestroy(handle);
    }
    GEMM(int _M, int _K, int _N):M(_M), N(_N), K(_K), alpha(1.0f),beta(1.0f){
        
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
        // vecbb = vecaa;

        a = vecaa;
        b = vecbb;
        

        a.resize(M, K);
        b.resize(K, N);
        // b = a.transpose();

        a.setRandom();
        b.setRandom();

        // a.setOnes();
        // b.setOnes();
        // std::cout << a.transpose();

        cudaMalloc(&deva, M * K * sizeof(float));
        cudaMemcpy(deva, a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&devb, K * N * sizeof(float));
        cudaMemcpy(devb, b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&devc, M * N * sizeof(float));
        cublasCreate(&handle);
    }

    Eigen::MatrixXf PureExcute(){
        Eigen::MatrixXf c = Eigen::MatrixXf::Zero(M, N);
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
        Eigen::MatrixXf c = Eigen::MatrixXf::Zero(M, N);
        c.noalias() = alpha * a * b + beta * c;
        return c;
    }

    Eigen::MatrixXf CUDAExecute(){
        Eigen::MatrixXf c = Eigen::MatrixXf::Zero(M, N);
        cudaMemset(devc, 0, M * N * sizeof(float));
        CallKernel5(deva, devb, devc, alpha, beta, M, N, K);
        cudaMemcpy(c.data(), devc, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        return c;
    }

    Eigen::MatrixXf CUBLASExecute(){
        Eigen::MatrixXf c = Eigen::MatrixXf::Zero(M, N);
        cudaMemset(devc, 0, M * N * sizeof(float));
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, deva, M, devb, K, &beta, devc, M);
        // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, deva, M, devb, K, &beta, devc, M);
        cudaMemcpy(c.data(), devc, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        return c;
    }

     void checkAndPrintTiming() {
         auto [cudaTime, cudaResult] = executeAndTime([this](){ return this->CUDAExecute(); });
        auto [eigenTime, eigenResult] = executeAndTime([this](){ return this->EigenExcute(); });
        auto [cublasTime, cublasResult] = executeAndTime([this](){ return this->CUBLASExecute(); });
        // auto cublasResult = eigenResult;
        

        std::cout << "Eigen Excute Time: " << eigenTime << " ms" << std::endl;
        std::cout << "cublas Excute Time: " << cublasTime << " ms" << std::endl;
        std::cout << "cuda Excute Time: " << cudaTime << " ms" << std::endl;

        if (cublasResult.isApprox(cudaResult, 1e-6)) {
            std::cout << "executions are the same." << std::endl;
        } else {
            std::cout << "Executions are different:" << std::endl;
            std::cout << "Eigen Result: \n" << eigenResult << std::endl;
            std::cout << "cublas Result: \n" << cublasResult << std::endl;
            std::cout << "cuda Result: \n" << cudaResult << std::endl;
        }

        // if (eigenResult.isApprox(cudaResult, 1e-6) && eigenResult.isApprox(cublasResult, 1e-6)) {
        //     std::cout << "executions are the same." << std::endl;
        // } else {
        //     std::cout << "Executions are different:" << std::endl;
        //     std::cout << "Eigen Result: \n" << eigenResult << std::endl;
        //     std::cout << "cublas Result: \n" << cublasResult << std::endl;
        //     std::cout << "cuda Result: \n" << cudaResult << std::endl;
        // }
    }
};