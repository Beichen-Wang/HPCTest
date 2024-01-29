#include "./MyVector.hpp"
#include <iostream>

int main(){
    Vector<double> a;
    a.push_back(1.0f);
    std::cout << "the first num is : " << a[0] << " size : "<< a.size() << std::endl;
}