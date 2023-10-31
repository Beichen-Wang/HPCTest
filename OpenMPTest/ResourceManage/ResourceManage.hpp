#include <iostream>
#include <thread>
#include <sched.h>
#include <atomic>

class CPULock
{
private:
    bool _tryLock(int cpuId)
    {
        // 设置当前线程亲缘调度
        CPU_ZERO(&m_mask);
        CPU_SET(cpuId, &m_mask);
        if (sched_setaffinity(0, sizeof(m_mask), &m_mask) == -1)
        {
            return false;
        } else {
            return true;
        }
    }
public:
    void Lock(){
        int numCpus = std::thread::hardware_concurrency();
        for (int cpuId = numCpus; cpuId > 0; cpuId--)
        {
            if (_tryLock(cpuId))
            {
                lock.store(true);
                break;
            }
        }
        std::cerr << "Failed to set CPU affinity" << std::endl;
    }

    void Unlock(){
        if(lock.load()){
            CPU_ZERO(&m_mask);
            CPU_SETALL(&m_mask);
            if (sched_setaffinity(0, sizeof(m_mask), &m_mask) == -1)
            {
                return false;
        }
    }

    ~CPULock()
    {
        // 恢复默认的调度策略
        if (sched_setaffinity(0, sizeof(m_mask), &m_mask) == -1)
        {
            std::cerr << "Failed to restore CPU affinity" << std::endl;
            return;
        }
    }

private:
    cpu_set_t m_mask;
    std::atomic<bool> lock(false);
};

// 程序执行函数
void programExecution()
{
    // 在这里编写需要独占CPU执行的程序逻辑
    // ...

    // 离开作用域时，CPULock对象的析构函数会自动恢复默认的调度策略
}

int main()
{
    // 创建一个新线程执行程序
    std::thread programThread([](){
        CPULock cpuLock(0); // 设置CPU 0亲缘调度
        programExecution();
    });

    // 等待程序执行完成
    programThread.join();

    return 0;
}
