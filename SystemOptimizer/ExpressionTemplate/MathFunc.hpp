#pragma once
template <typename Derived>
class ScaleBinaryOprBase {
    public:
    Derived& derived(){
        return *static_cast<Derived *>(this);
    }
};

template <typename T>
class Sum : ScaleBinaryOprBase<Sum<T>> {
    public:
    T operator()(T a, T b){
        return a + b;
    }
};

template <typename T>
class Sub : ScaleBinaryOprBase<Sum<T>> {
    public:
    T operator()(T a, T b){
        return a - b;
    }
};

template <typename T>
class Assign : ScaleBinaryOprBase<Assign<T>> {
    public:
    void operator()(T & a, const T & b){
        a = b;
    }
};

template <typename T>
class Product: ScaleBinaryOprBase<Product<T>> {
    public:
    T operator()(T a, T b){
        return a * b;
    }
};