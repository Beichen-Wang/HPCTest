#include "GEMM.hpp"

int main(){
    // GEMM g(2000, 2000, 2000);
    GEMM g(8, 64, 8);
    // GEMM g(1024, 1024, 1024);
    g.checkAndPrintTiming(); 
}