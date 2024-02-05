#include <atomic>
#include <sched.h>
#include <time.h>
class spinlock {
    std::atomic_flag v = ATOMIC_FLAG_INIT;
    bool try_lock(){
        return !v.test_and_set(std::memory_order_relaxed);
    }
    void yield(int k){
        if(k < 32){
            sched_yield();
        } else {
            timespec ts = { 0, 0 };
            ts.tv_nsec = 1000;
            nanosleep(&ts, nullptr);
        }
    }
    void lock(){
        size_t k = 0;
        for(int i = 0; !try_lock(); i++){
            yield(k);
        }
    }
    void unlock(){
        v.clear(std::memory_order_relaxed);
    }
    public:
    class scoped_lock {
        private:
            spinlock & sp_;
        public:
            scoped_lock(spinlock & sp): sp_(sp){
                sp_.lock();
            }
            ~scoped_lock(){
                sp_.unlock();
            }
    };
};