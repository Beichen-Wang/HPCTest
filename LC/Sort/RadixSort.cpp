#include <iostream>
#include <vector>
#include <limits>

using namespace std;
int FindMaxBit(vector<int> & input){
    int maxValue = std::numeric_limits<int>::min();
    int result;
    for(int i = 0; i < input.size(); i++){
        if(input[i] > maxValue){
            maxValue = input[i];
        }
    }
    while(maxValue){
        maxValue = maxValue / 10;
        result++;
    }
}

void RadixSort(vector<int> & input){
    int d = FindMaxBit(input);
    int base = 1;
    vector<int> count(10);
    vector<int> start(10);
    vector<int> tmp(input.size());
    while(d--){
        fill(count.begin(), count.end(), 0);
        for(int i = 0; i < input.size(); i++){
            int k = input[i] / base % 10;
            count[k]++;
        }
        fill(start.begin(), start.end(), 0);
        for(int i = 1; i < 10; i++){
            start[i] = start[i - 1] + count[i - 1];
        }
        for(int i = 0; i < input.size(); i++){
            int k = input[i] / base % 10;
            tmp[start[k]++] = input[i];
        }
        input = tmp;
        base *= 10;
    }
}