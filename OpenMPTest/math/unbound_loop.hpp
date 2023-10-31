#include <omp.h>
#include <memory>
#include <signal.h>
#include "backtrace.hpp"

struct NodeList{
    int val;
    std::unique_ptr<NodeList> next;
    NodeList(int input):val(input),next(nullptr){};
};

class SubProcess {
    public:
        template <typename Func>
        void operator() (int & mulSum, int n){
            if(typeid(Func).name() == typeid(&SubProcess::SubProcess1).name()){
                SubProcess1(mulSum, n);
            } else if (typeid(Func).name() == typeid(&SubProcess::SubProcess2).name()){
                SubProcess2(n);
            } else {
                signal(SIGTERM, util::dump_stack);
                exit(1);
            }
        }

        void SubProcess1(int & mulSum, int n){
            mulSum *= (n + 1);
        };
        void SubProcess2(int n){
            int temp = n * (n + 1);
        };
};

class Unboundloop{
    private:
        std::unique_ptr<NodeList> Init(int n){
            std::unique_ptr<NodeList> head = nullptr;
            NodeList * current = nullptr;
            for(int i = 0; i < n; i++){
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
        Unboundloop(int n):mulSum(1){
            node = Init(n);
        };
        int OMPProcess(){
            mulSum = 1;
            NodeList * current;
            current = node.get();
            #pragma omp parallel num_threads(2)
            {
                #pragma omp single 
                {
                    while(current){
                        #pragma omp task
                            subProcess.template operator()<decltype(&SubProcess::SubProcess2)>(mulSum, current->val);
                        current = (current->next).get();
                    }
            }
            }
            return mulSum;
        }
        int NorProcess(){
            mulSum = 1;
            NodeList * current;
            current = node.get();
            while(current){
                subProcess.template operator()<decltype(&SubProcess::SubProcess2)>(mulSum, current->val);
                current = (current->next).get();
            }
            return mulSum;
        }
    private:
        std::unique_ptr<NodeList> node;
        int mulSum;
        SubProcess subProcess;
};
