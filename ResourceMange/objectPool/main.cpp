#include <iostream>
#include <memory>
#include "objectPool.hpp"

class A {
    int i;
    void * ptr;
    public:
    A(){
        std::cout << "construct A" << std::endl;
        ptr = malloc(10);
    }
    ~A(){
        std::cout << "destruct A" << std::endl;
        free(ptr);
    }
};

int main(){
    {
        ObjectPool<A> ap;
        {
            auto a = ap.get_shared_pointer();
            std::cout << "a.size() = " << ap.size() << " capacity() is " << ap.capacity() << std::endl;
        }
        std::cout << "a.size() = " << ap.size() << " capacity() is " << ap.capacity() << std::endl;
    }
    return 0;
}