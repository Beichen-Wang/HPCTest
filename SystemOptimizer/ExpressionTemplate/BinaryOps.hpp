#pragma once
#include "MathFunc.hpp"
#include "ForwardDeclaration.hpp"

template <typename ScalerBinaryOp, typename TL, typename TR>
struct Trait<BinaryOps<ScalerBinaryOp, TL, TR>>{
        typedef typename Trait<TL>::Scalar Scalar;
};

template <typename ScalerBinaryOp, typename TL, typename TR>
class BinaryOps : public MatrixBase<BinaryOps<ScalerBinaryOp, TL, TR>>{
    public:
    BinaryOps(TL L, TR R, ScalerBinaryOp BinaryOp = ScalerBinaryOp()):mL(L),mR(R),m_functor(BinaryOp){}
    ScalerBinaryOp func(){
        return m_functor;
    }
    TL left(){
        return mL;
    }
    TR right(){
        return mR;
    }
    private:
    TL mL;
    TR mR;
    ScalerBinaryOp m_functor;
};