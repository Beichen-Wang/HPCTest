#include <stdio.h>
#include <stdlib.h>
#include <csignal>
#include <unistd.h>
#include <iostream>
#include <atomic>
#include <mutex>

std::atomic<bool> sigusr1_received(false);
class Signal{
    std::atomic<bool> signal;
    public:
        static Signal * Instance(){
            static Signal * instance = nullptr;
            if(!instance){
                static std::once_flag flag;
                std::call_once(flag, [&]{instance = new Signal();});
            } 
            return instance;

        }
        bool GetSignal(){
            return signal.load(std::memory_order_acquire);
        }
        void SetSignal(bool value){
            signal.store(value, std::memory_order_release);
        }   
        void exchangeSignal(){
            SetSignal(!GetSignal());
        }
    private:
        // Signal(){
        //     std::cout << "construct" << std::endl;
        // };
        Signal():signal(false){};
        Signal(const Signal &) = delete;
        Signal& operator=(const Signal &) = delete;
};
// 信号处理函数
void sigusr1_handler(int signo) {
    Signal::Instance()->exchangeSignal();
}

int main() {
    int count = 0;

    // 将 SIGUSR1 信号与信号处理函数关联
    if (signal(SIGUSR1, sigusr1_handler) == SIG_ERR) {
        perror("Unable to set up signal handler");
        sleep(2);
        exit(EXIT_FAILURE);
    }

    printf("Waiting for SIGUSR1 signal...\n");

    // 进入一个无限循环，等待 SIGUSR1 信号
    while (1) {
        if (Signal::Instance()->GetSignal()) {
            printf("Received SIGUSR1 signal\n");
            // Signal::Instance()->SetSignal(false);
            sleep(5);
            // 这里可以添加处理 SIGUSR1 信号的代码
        }
        

        std::cout << " count : " << count++ << std::endl;
        sleep(1);
    }

    return 0;
}
