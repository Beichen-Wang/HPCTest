#pragma once
#include "ForwardDeclaration.hpp"
#include "BinaryOps.hpp"

template <typename _Scalar, int _Size>
class evaluator<Matrix<_Scalar, _Size>>{
    Matrix<_Scalar, _Size> & m_data;
    public:
    typedef _Scalar Scalar;
    evaluator(Matrix<_Scalar, _Size> & mat):m_data(mat){}
    _Scalar & getR(int Index){
        return m_data.getR(Index);
    }
};

template <typename BinaryOP, typename TL, typename TR>
class evaluator<BinaryOps<BinaryOP, TL, TR>> {
    BinaryOps<BinaryOP, TL, TR> & m_data;
public:
    evaluator(BinaryOps<BinaryOP, TL, TR> & data):m_data(data){}
    evaluator(const BinaryOps<BinaryOP, TL, TR> & data):m_data(const_cast<BinaryOps<BinaryOP, TL, TR> &>(data)){};
    typename Trait<TL>::Scalar get(int Index) {
        return m_data.func()(m_data.left().get(Index), m_data.right().get(Index));
    }
};