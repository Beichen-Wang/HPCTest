#include <iostream>

using namespace std;

struct Node{
    int key;
    int value;
    Node * pre;
    Node * next;
    Node():key(0),value(0),pre(nullptr),next(nullptr){};
    Node(int key_, int value_):key(key_),value(value_):pre(nullptr),next(nullptr){};
};

template <int NUM>
class LRU{
    unordered_map<int, Node *> map;
    Node * head;
    Node * end;
    int size;
    inline void _MoveToHead(Node * node){
        _RemoveNode(node);
        _AddToHead(node);
    }
    inline void _RemoveNode(Node * node){
        node->pre->next = node->next;
        node->next->pre = node->pre;
    }
    inline void _AddToHead(Node * node){
        node->next=head->next;
        node->pre=head;
        head->next->pre=node;
        head->next=node;
    }
    inline Node * _RemoveEnd(){
        Node * removed_node = end->pre;
        _RemoveNode(removed_node);
        return removed_node;
    }
public:
    LRU(){
        head = new Node();
        end = new Node();
        head->next = end;
        end->pre = head;
        size = 0;
    }
    int get(int key){
        if(!map.count(key)){
            std::cerr << "没有该key: " << key << std::endl;
            return -1;
        }
        _MoveToHead(map[key]);
        return map.count(key)->value;
    }
    void put(int key, int value){
        if(!map.count(key)){
            Node * cur = new Node(key, value);
            map[key] = cur;
            _AddToHead(cur);
            size++;
        } else {
            _MoveToHead(map[key]);
            map[key]->value = value;
        }
        if(size > NUM){
            Node * removed_node = _RemoveEnd();
            map.erase(key);
            delete removed_node;
            size--;
        }
  }
};
