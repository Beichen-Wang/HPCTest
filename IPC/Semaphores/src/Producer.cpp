#include <iostream>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/stat.h>

int main(){
    sem_t* semptr = sem_open("/sem", O_CREAT, 0644, 0);
    sem_post(semptr);
    std::cout << "semaphre have been publish" << std::endl;
}