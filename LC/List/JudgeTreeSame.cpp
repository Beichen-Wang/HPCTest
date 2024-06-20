#include <iostream>

struct Node {
    int value;
    Node * left;
    Node * right;
};

bool JudgeTreeSame(Node * left, Node * right){
    if(left == nullptr && right == nullptr){
        return true;
    }
    if(left == nullptr || right == nullptr){
        return false;
    }
    if(left->value != right->value){
        return false;
    }
    if(!JudgeTreeSame(left -> left, left -> right)
    || !JudgeTreeSame(right-> left, right-> right)
    || !JudgeTreeSame(left -> left, right-> right)
    || !JudgeTreeSame(left -> right,right-> left)){
        return false;
    }
    return true;
}
