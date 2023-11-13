#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <memory>
#include <iostream>

namespace IPC { // namespace IPC
#define CHECK(cmd)                                                             \
  if (cmd < 0) {                                                               \
    fprintf(                                                                   \
        stderr,                                                                \
        "the file is %s in %d, the command is %s, and the error info is %s",   \
        __FILE__, __LINE__, #cmd, strerror(cmd));                              \
    exit(1);                                                                   \
  }
class Flock {
    private:
    flock lock;

    public:
    Flock(short l_type, pid_t l_pid, short l_whence = SEEK_SET, short l_start = 0,
            short l_len = 0) {
        lock.l_type = l_type;
        lock.l_whence = l_whence;
        lock.l_start = l_start;
        lock.l_len = l_len;
        lock.l_pid = l_pid;
    }

    void SetLockType(short l_type) { lock.l_type = l_type; };
    flock& GetLock() { return lock;}
    ~Flock() { SetLockType(F_UNLCK); }
};

class ReadFile {
    private:
        std::string stringBuffer;
    public:
    ReadFile(const char *fileName = "data.txt") {
        int fd;
        fd = open(fileName, O_RDONLY);
        Flock lock(F_RDLCK, getpid());
        CHECK(fcntl(fd, F_SETLKW, &lock));
        
        off_t fileSize = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);

        stringBuffer.resize(fileSize);
        read(fd, &stringBuffer[0], fileSize);

        lock.SetLockType(F_UNLCK);
        CHECK(fcntl(fd, F_SETLK, &lock));
        close(fd);
    };
    const std::string & GetBuffer() {return stringBuffer;};
};

class WriteFile {
public:
    WriteFile(const char * data, const char * fileName = "data.txt"){
        int fd;
        fd = open(fileName, O_RDWR | O_CREAT, 0666);
        
        Flock lock(F_WRLCK, getpid());
        CHECK(fcntl(fd, F_SETLK, &lock.GetLock()));

        CHECK(write(fd, data, strlen(data)));
        CHECK(truncate(fileName, strlen(data)));

        lock.SetLockType(F_UNLCK);
        CHECK(fcntl(fd, F_SETLK, &lock.GetLock()));
        close(fd);

        std::cout << "write success and the file is "<< fileName << std::endl;
    }
};
} // namespace IPC