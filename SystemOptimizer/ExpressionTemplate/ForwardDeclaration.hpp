#pragma once
template <typename Scalar, int Size>
class Matrix;
template <typename T>
class Trait;
template <typename ScalerBinaryOp, typename TL, typename TR>
class BinaryOps;
template <typename Derived>
class MatrixBase;
template <typename Derived>
class evaluator;
template<typename Derived>
class evaluator<const Derived>
  :public evaluator<Derived>
{
    public:
    explicit evaluator(const Derived& xpr) : evaluator<Derived>(xpr) {}
};