#include "GEMM.hpp"

int main(){
    GEMM g(2048, 2048, 2048);
    // GEMM g(1024, 2048, 4096);
    // GEMM g(8, 16, 8);
    g.checkAndPrintTiming(); 
}