import tvm
import tvm.testing
from tvm import te
import numpy as np
import timeit
import os

class GEMM:
    def __init__(self, M, N, K, bs, targetI = "llvm"):
        self.M = M
        self.N = N
        self.K = K
        self.bs = bs
        self.target = targetI
        self.dtype = "float32"
        self.target = tvm.target.Target(target=targetI)
        self.dev = tvm.device(self.target.kind.name)
        self.a = tvm.nd.array(np.random.rand(M,K).astype(self.dtype), self.dev)
        self.b = tvm.nd.array(np.random.rand(K,N).astype(self.dtype), self.dev)
        self.c = tvm.nd.array(np.zeros((M,N), dtype=self.dtype), self.dev)
        self.log = []
    #utils
    def EvaluateOperation(self, func, baseC):
        self.c = tvm.nd.array(np.zeros((M,N), dtype=self.dtype), self.dev)
        func(self.a, self.b, self.c)
        tvm.testing.assert_allclose(self.c.numpy(), baseC, rtol=1e-5)
        evaluator = func.time_evaluator(func.entry_name, self.dev, number=10)
        mean_time = evaluator(self.a, self.b, self.c).mean
        print("%s: %f" % (func.name, mean_time))
        self.log.append((func.name, mean_time))
    #numpy
    def NumpyGEMM(self):
        npRepeatNum = 1
        npRunningTime = timeit.timeit(
            setup="import numpy\n",
            stmt="answer = numpy.dot(a_np, b_np)",
            number=npRepeatNum,
            globals={"a_np": self.a.numpy(), "b_np": self.b.numpy()}
        )
        print("Numpy running time: %f" % (npRunningTime / npRepeatNum))
        return np.dot(self.a.numpy(), self.b.numpy())
    #default
    def TEDefaultGemm(self):
        k = te.reduce_axis((0, self.K), "k")
        A = te.placeholder((self.M, self.K), name="A")
        B = te.placeholder((self.K, self.N), name="B")
        C = te.compute((self.M, self.N), lambda x, y: te.sum(A[x,k]*B[k,y], axis = k), name="C")
        s = te.create_schedule(C.op)
        func = tvm.build(s, [A,B,C], target = self.target, name = "default")
        print(tvm.lower(s, [A,B,C], simple_mode=True))
        return func
    #optimizer1---final--block,vectory,parallel
    def TEBlockVectoryParallelGemm(self):
        self.M = te.var("M")
        self.K = te.var("K")
        self.N = te.var("N")
        k = te.reduce_axis((0, self.K), "k")
        A = te.placeholder((self.M, self.K), name="A")
        B = te.placeholder((self.K, self.N), name="B")
        C = te.compute((self.M, self.N), lambda x, y: te.sum(A[x,k]*B[k,y], axis = k), name="C")
        s = te.create_schedule(C.op)
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], self.bs, self.bs)
        ko, ki = s[C].split(k, factor=16)
        s[C].reorder(xo, yo, ko, xi, ki, yi)
        # s[C].unroll(ki)
        s[C].vectorize(yi)
        s[C].parallel(xo)
        func = tvm.build(s, [A,B,C], target = self.target, name = "blockVectoryParallel")
        print(tvm.lower(s, [A,B,C], simple_mode=True))
        return func
    #optimizer2.1--+cache
    def TECacheGemm(self):
        k = te.reduce_axis((0, self.K), "k")
        A = te.placeholder((self.M, self.K), name="A")
        B = te.placeholder((self.K, self.N), name="B")
        C = te.compute(
            (self.M, self.N),
            lambda x, y: te.sum(A[x, k] * B[k,y], axis=k),
            name = "C",
        )
        s = te.create_schedule(C.op)
        CC = s.cache_write(C, "global")
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], self.bs, self.bs)
        s[CC].compute_at(s[C],yo)
        # New inner axes
        xc, yc = s[CC].op.axis

        (k,) = s[CC].op.reduce_axis
        ko, ki = s[CC].split(k, factor=16)
        s[CC].reorder(ko, xc, ki, yc)
        s[CC].unroll(ki)
        s[CC].vectorize(yc)

        # parallel
        s[C].parallel(xo)

        func = tvm.build(s, [A,B,C], target = self.target, name = "CacheParallel")
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        return func
    #optimizer2.2--+pack
    def TEPackGemm(self):
        k = te.reduce_axis((0, self.K), "k")
        A = te.placeholder((self.M, self.K), name="A")
        B = te.placeholder((self.K, self.N), name="B")
        packedB = te.compute((self.N / self.bs, self.K, self.bs), lambda x, y, z: B[y, x * self.bs + z], name="packedB")
        C = te.compute(
            (self.M, self.N),
            lambda x, y: te.sum(A[x, k] * packedB[tvm.tir.indexdiv(y, self.bs), k,tvm.tir.indexmod(y, self.bs)], axis=k),
            name = "C",
        )
        s = te.create_schedule(C.op)
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], self.bs, self.bs)
        (k,) = s[C].op.reduce_axis
        ko, ki = s[C].split(k, factor=16)
        s[C].reorder(xo, yo, ko, xi, ki, yi)
        s[C].vectorize(yi)

        x, y, z = s[packedB].op.axis
        s[packedB].vectorize(z)
        s[packedB].parallel(x)

        s[C].parallel(xo)
        func = tvm.build(s, [A,B,C], target = self.target, name = "PackParallel")
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        return func
    #optimizer3.1--+cache+pack
    def TECachePackGemm(self):
        k = te.reduce_axis((0, self.K), "k")
        A = te.placeholder((self.M, self.K), name="A")
        B = te.placeholder((self.K, self.N), name="B")
        packedB = te.compute((self.N / self.bs, self.K, self.bs), lambda x, y, z: B[y, x * self.bs + z], name="packedB")
        C = te.compute(
            (self.M, self.N),
            lambda x, y: te.sum(A[x, k] * packedB[tvm.tir.indexdiv(y, self.bs), k,tvm.tir.indexmod(y, self.bs)], axis=k),
            name = "C",
        )
        s = te.create_schedule(C.op)
        CC = s.cache_write(C, "global")
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], self.bs, self.bs)
        s[CC].compute_at(s[C],yo)
        # New inner axes
        xc, yc = s[CC].op.axis

        (k,) = s[CC].op.reduce_axis
        ko, ki = s[CC].split(k, factor=64)
        s[CC].reorder(ko, xc, ki, yc)
        s[CC].unroll(ki)
        # s[CC].pragma(ki, "unroll_explicit", 2)
        s[CC].vectorize(yc)

        # parallel
        s[C].parallel(xo)
        x, y, z = s[packedB].op.axis
        s[packedB].vectorize(z)
        s[packedB].parallel(x)
        func = tvm.build(s, [A,B,C], target = self.target, name = "CacheParallel")
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        return func

    def GetLibrary(self, func):
        curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        dylib_path = os.path.join(curr_path, "lib/GEMM.so")
        func.export_library(dylib_path)


if __name__ == "__main__":
    M = 512
    K = 5120
    N = 512
    bs = 64
    instance = GEMM(M,N,K,bs)
    baseC = instance.NumpyGEMM()
    funcDefault = instance.TEDefaultGemm()
    instance.EvaluateOperation(funcDefault,baseC)
    funcBlockPermuteVectory = instance.TEBlockVectoryParallelGemm()
    instance.EvaluateOperation(funcBlockPermuteVectory,baseC)
    instance.GetLibrary(funcBlockPermuteVectory)
    # funcPack = instance.TEPackGemm()
    # instance.EvaluateOperation(funcPack,baseC)
    # funcCache = instance.TECachePackGemm()
    # instance.EvaluateOperation(funcCache,baseC)
    # funcCache = instance.TECacheGemm()
    # instance.EvaluateOperation(funcCache,baseC)