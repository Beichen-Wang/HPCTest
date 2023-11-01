#include <omp.h>
#include <memory>
#include <signal.h>
#include "backtrace.hpp"

struct NodeList{
    size_t val;
    std::unique_ptr<NodeList> next;
    NodeList(size_t input):val(input),next(nullptr){};
};

class SubProcess {
    public:
        template <typename Func>
        void operator() (size_t & mulSum, size_t n){
            if(typeid(Func).name() == typeid(&SubProcess::SubProcess1).name()){
                SubProcess1(mulSum, n);
            } else if (typeid(Func).name() == typeid(&SubProcess::SubProcess2).name()){
                SubProcess2(n);
            } else if (typeid(Func).name() == typeid(&SubProcess::SubProcess3).name()){
                SubProcess3(n, mulSum);
            }else if (typeid(Func).name() == typeid(&SubProcess::SubProcess4).name()){
                SubProcess4(n, n + 1);
            } else {
                signal(SIGTERM, util::dump_stack);
                exit(1);
            }
        }

        void SubProcess1(size_t & mulSum, size_t n){
            mulSum *= (n + 1);
        };
        void SubProcess2(size_t n){
            size_t temp = n * (n + 1);
        };
        void SubProcess3(size_t n, size_t & mulSum){
            size_t count = (n + 3) + 10;
            for(size_t i = 1; i < count - 1; i++){
                mulSum += i * (i + 1);
            }
        }
        void SubProcess4(size_t n, size_t t){
            size_t temp;
            size_t count = (n + t) * 1000;
            for(size_t i = 1; i < count - 1; i++){
                temp += i * (i + 1);
            }
        }
};

class Unboundloop{
    private:
        std::unique_ptr<NodeList> Init(size_t n){
            std::unique_ptr<NodeList> head = nullptr;
            NodeList * current = nullptr;
            for(size_t i = 0; i < n; i++){
                if(head == nullptr){
                    head = std::make_unique<NodeList>(i);
                    current = head.get();
                } else {
                    current->next = std::make_unique<NodeList>(i);
                    current = (current->next).get();
                }
            }
            return head;
        };

    public:
        Unboundloop(size_t n):mulSum(1){
            node = Init(n);
        };
        template <typename Func>
        size_t OMPProcess(){
            mulSum = 1;
            NodeList * current;
            current = node.get();
            #pragma omp parallel num_threads(4)
            {
                #pragma omp single 
                {
                    #pragma omp taskgroup
                    while(current){
                        #pragma omp task default(shared) untied
                            subProcess.template operator()<Func>(mulSum, current->val);
                        current = (current->next).get();
                    }
            }
            }
            return mulSum;
        }
        template <typename Func>
        size_t NorProcess(){
            mulSum = 1;
            NodeList * current;
            current = node.get();
            while(current){
                subProcess.template operator()<Func>(mulSum, current->val);
                current = (current->next).get();
            }
            return mulSum;
        }
    private:
        std::unique_ptr<NodeList> node;
        size_t mulSum;
        SubProcess subProcess;
};
