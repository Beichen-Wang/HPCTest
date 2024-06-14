

#include <iostream>
#include <vector>

struct Node {
    int value;
    Node * next;
    Node(int value_input):value(value_input),next(nullptr){};
};

Node * RevertList(Node * input){
    if(input == nullptr){
        std::cout << "input 为空";
        return nullptr;
    }
    if(input->next == nullptr){
        std::cout << "input size为1";
        return input;
    }
    Node * pre = nullptr;
    Node * cur = input;
    Node * next;
    while(cur != nullptr){
        next = cur ->next;
        cur->next = pre;
        pre = cur;
        cur = next;  
    }
    return pre;
}

int main(){
    const int size = 5;
    Node * head;
    head = new Node(0);
    Node * cur = head;
    for(int i = 1; i < size; i++){
        cur -> next = new Node(i);
        cur = cur ->next;
    }
    Node * end = RevertList(head);
    // Node * end = head;
    while(end != nullptr){
        std::cout << end->value;
        end = end ->next;
    }
    
}

