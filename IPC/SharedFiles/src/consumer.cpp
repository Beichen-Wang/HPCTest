#include "SharedFiles.h"

using namespace IPC;

int main(int argc, char ** argv){
    const char * fileName;
    if(argc < 2){
        fileName = "data.txt";
    } else {
        fileName = argv[1];
    }
    ReadFile rf(fileName);
    std::cout << rf.GetBuffer();
}