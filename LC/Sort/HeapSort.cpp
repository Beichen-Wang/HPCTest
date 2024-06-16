#include <iostream>
#include <vector>
#include <queue>

using namespace std;
int FindNElement(vector<int> input, int n){
    if (n <= 0 || n > input.size()) {
        throw std::invalid_argument("Invalid value of n");
    }
    std::priority_queue<int, vector<int>, greater<int>> q;
    for(int i = 0; i < input.size(); i++){
        if(q.size() < n){
            q.push(input[i]);
        } else {
            if(q.top() > input[i]){
                q.pop();
                q.push(input[i]);
            }
        }
    }
    return q.top();
}