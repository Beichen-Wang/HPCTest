
#include <iostream>

struct Node {
    int value;
    Node * next;
    Node(int value_):value(value_),next(nullptr){};
};

class Solution {
    private:
    Node * RevertList(Node * input){
        Node * pre = nullptr;
        Node * cur = input;
        Node * next;
        while(cur -> next){
            next = cur -> next;
            cur -> next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
    public:
    Node * TwoNumAdd(Node * input1, Node * input2){
        input1 = RevertList(input1);
        input2 = RevertList(input2);
        Node * result = new Node(0);
        int carry = 0;
        while(input1->next && input2 ->next){
            result -> value = (input1->value + input2->value + carry) % 10;
            carry = (input1->value + input2->value + carry) / 10;
            result ->next = new Node(0);
            result = result ->next;
            input1 = input1 ->next;
            input2 = input2 ->next;
        }
        while(input1->next){
            result -> value = (input1->value + carry) % 10;
            carry = (input1->value + carry) / 10;
            result ->next = new Node(0);
            result = result ->next;
            input1 = input1 ->next;
        }
        while(input2->next){
            result -> value = (input2->value + carry) % 10;
            carry = (input2->value + carry) / 10;
            result ->next = new Node(0);
            result = result ->next;
            input2 = input2 ->next;
        }
        result = RevertList(result);
        return result;
    }
};
