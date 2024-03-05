#include <iostream>
#include "Matrix.hpp"

int main(){
    Matrix<double, 4> a;
    Matrix<double, 4> b;
    Matrix<double, 4> c;
    Matrix<double, 4> d;
    for(int i = 0; i < 4; i++){
        a[i] = i;
        b[i] = i;
        c[i] = i + 2;
    }
    d = a + b - c;
    for(int i = 0; i < 4; i++){
        std::cout << d[i] << " ";
    }
}