#include "SharedFiles.h"

using namespace IPC;

int main(int argc, char ** argv) {
   const char * data, *fileName;
   if(argc < 3){
      fileName = "data.txt";
   } else {
      fileName = argv[2];
   }
   if(argc < 2){
      data = "Welcome to HPC/IPC";
   } else {
      data = argv[1];
   }

   WriteFile w(data, fileName);
}