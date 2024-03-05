#pragma once
#include "BinaryOps.hpp"
#include "MathFunc.hpp"
#include "ForwardDeclaration.hpp"
#include "Assignment.hpp"
#include "Evaluator.hpp"

template <typename _Scalar, int _Size>
struct Trait<Matrix<_Scalar, _Size>>{
        typedef _Scalar Scalar;
        enum {
            Size = _Size,
        };
};

template <typename Scalar, int Size>
struct Storage {
    Scalar m_data[Size];
    Scalar& operator[](int Index){
        return m_data[Index];
    }
};

template <typename Scalar, int Size>
class Matrix : public MatrixBase<Matrix<Scalar, Size>>{
    Storage<Scalar, Size> m_data;
    public:
    Matrix():m_data(){}
    Scalar& getR(int Index){
        return m_data[Index];
    }
    Scalar get(int Index){
        return m_data[Index];
    }
    template <typename OtherDerived>
    Matrix& operator=(const MatrixBase<OtherDerived> & other){
        return MatrixBase<Matrix<Scalar, Size>>::operator=(other);
    }
    Scalar& operator[](int Index){
        return getR(Index);
    }
};

template <typename Derived>
class MatrixBase {
    public:
    typedef typename Trait<Derived>::Scalar Scalar;
    Derived& derived(){
        return *static_cast<Derived*>(this);
    }
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }
    template <typename OtherDerived>
    inline BinaryOps<Sum<Scalar>,Derived, OtherDerived> operator+(const MatrixBase<OtherDerived>& other){
        return BinaryOps<Sum<Scalar>,Derived, OtherDerived>(derived(), other.derived());
    }
    template <typename OtherDerived>
    inline BinaryOps<Sub<Scalar>,Derived, OtherDerived> operator-(const MatrixBase<OtherDerived>& other){
        return BinaryOps<Sub<Scalar>,Derived, OtherDerived>(derived(), other.derived());
    }
    template <typename OtherDerived>
    Derived& operator=(const MatrixBase<OtherDerived> & other){
        call_assignment(derived(), other.derived());
        return derived();
    }
    Scalar get(int index){
        return evaluator<Derived>(derived()).get(index);
    }
};