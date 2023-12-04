#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <vector>
#include <random>
#include <cassert>

#include "Timer.hpp"

void Check(DLTensor a, DLTensor b)
{
    // 检查形状是否相同
    assert(a.ndim == b.ndim);
    for (int i = 0; i < a.ndim; ++i)
    {
        assert(a.shape[i] == b.shape[i]);
    }
    for (int i = 0; i < a.shape[0]; ++i)
    {
        for (int j = 0; j < a.shape[1]; ++j)
        {
            assert(static_cast<float *>(a.data)[i * a.shape[1] + j] == static_cast<float *>(b.data)[i * a.shape[1] + j]);
        }
    }

    // 检查数据类型是否相同
    assert(a.dtype.code == b.dtype.code);
    assert(a.dtype.bits == b.dtype.bits);
    assert(a.dtype.lanes == b.dtype.lanes);

    std::cout << "Check passed" << std::endl;
}

#define ITERTIME 10
#define ITER(func)                     \
    for (int i = 0; i < ITERTIME; i++) \
    {                                  \
        func;                          \
    }

class GEMM
{
private:
    std::shared_ptr<DLTensor> A, B, C;
    util::Timer timer;

public:
    void Init(int M, int K, int N)
    {
        // 创建输入矩阵并填充数据
        tvm::runtime::NDArray A_array = tvm::runtime::NDArray::Empty({M, K}, {kDLFloat, 32, 1}, {kDLCPU, 0});
        tvm::runtime::NDArray B_array = tvm::runtime::NDArray::Empty({K, N}, {kDLFloat, 32, 1}, {kDLCPU, 0});
        A = std::make_shared<DLTensor>(A_array.ToDLPack()->dl_tensor);
        B = std::make_shared<DLTensor>(B_array.ToDLPack()->dl_tensor);

        // 填充输入矩阵
        std::random_device rd;
        std::default_random_engine generator(rd());

        // 定义随机数分布范围
        std::uniform_int_distribution<int> distribution(1, 10);

        for (int i = 0; i < A->shape[0] * A->shape[1]; i++)
        {
            static_cast<float *>(A->data)[i] = distribution(generator);
        }
        for (int i = 0; i < B->shape[0] * B->shape[1]; i++)
        {
            static_cast<float *>(B->data)[i] = distribution(generator);
        }
    };

    DLTensor Process(std::string fname)
    {
        tvm::runtime::NDArray C_array = tvm::runtime::NDArray::Empty({A->shape[0], B->shape[1]}, {kDLFloat, 32, 1}, {kDLCPU, 0});
        tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(fname);
        tvm::runtime::PackedFunc f = mod.GetFunction("blockVectoryParallel");
        DLTensor C = C_array.ToDLPack()->dl_tensor;
        timer.start();
        ITER(f(A.get(), B.get(), &C));
        timer.stop();
        std::cout << "TVM GEMM used : " << timer.duration()/ITERTIME << " ms" << std::endl;
        return C;
    }

    DLTensor NaiveProcess()
    {
        // 矩阵乘法
        tvm::runtime::NDArray C_array = tvm::runtime::NDArray::Empty({A->shape[0], B->shape[1]}, {kDLFloat, 32, 1}, {kDLCPU, 0});
        DLTensor C = C_array.ToDLPack()->dl_tensor;
        timer.start();
        for (int i = 0; i < A->shape[0]; ++i)
        {
            for (int j = 0; j < B->shape[1]; ++j)
            {
                float sum = 0;
                for (int k = 0; k < A->shape[1]; ++k)
                {
                    sum += static_cast<float *>(A->data)[i * A->shape[1] + k] * static_cast<float *>(B->data)[k * B->shape[1] + j];
                }
                static_cast<float *>(C.data)[i * C.shape[1] + j] = sum;
            }
        }
        timer.stop();
        std::cout << "Naive GEMM used : " << timer.duration() << " ms" << std::endl;
        // 返回结果tensor
        return C;
    }
};

int main(int chrc, char **chrv)
{
    if (chrc < 4)
    {
        std::cout << "you should use: ./gemm 512 5120 512" << std::endl;
        exit(1);
    }
    int M = atoi(chrv[1]);
    int K = atoi(chrv[2]);
    int N = atoi(chrv[3]);
    GEMM instance;
    instance.Init(M, K, N);
    DLTensor CP = instance.Process("../../lib/GEMM.so");
    DLTensor CN = instance.NaiveProcess();
    Check(CP, CN);
}