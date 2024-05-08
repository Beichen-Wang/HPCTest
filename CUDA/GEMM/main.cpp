#include "GEMM.hpp"

int main(){
    GEMM g(2048, 2048, 2048);
    // GEMM g(512, 512, 512);
    // GEMM g(8, 8, 8);
    g.checkAndPrintTiming(); 
}