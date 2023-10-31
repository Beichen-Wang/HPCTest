#include <iostream>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <string.h>

namespace util{
    void dump_stack(int sig) {
        void* buffer[30];
        size_t size;
        char** strings = nullptr;

        size = backtrace(buffer, 30);
        strings = backtrace_symbols(buffer, size);

        if (strings == nullptr) {
            perror("backtrace_symbols");
            exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < size; ++i) {
            std::cout << strings[i] << std::endl;
        }

        free(strings);
    }
}