#include <iostream>
#include <semaphore.h>
#include <fcntl.h>

int main(){
    sem_t* semptr = sem_open("/sem", O_CREAT, 0644, 0);
    sem_wait(semptr);
    std::cout << "semaphre have been recevie" << std::endl;
    sem_close(semptr);
}