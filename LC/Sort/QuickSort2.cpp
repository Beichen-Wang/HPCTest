#include <iostream>
#include <vector>
#include <utility>

using namespace std;
void Sort(vector<int> & input, int size, int index){
    int left = 2 * index + 1;
    int right = 2 * index + 2;
    int maxIndex = index;
    if(left < size && input[left] > input[index]){
        maxIndex = left;
    }
    if(right < size && input[right] > input[index]){
        maxIndex = right;
    }
    if(maxIndex != index){
        swap(input[maxIndex], input[index]);
        Sort(input, size, maxIndex);
    }
}

void HeapSort(vector<int> & input){
    int size = input.size();
    for(int i = size/2; i >= 0; i--){
        Sort(input, size, i);
    }
    for(int s = size - 1; s >= 1; s--){
        swap(input[s], input[0]);
        Sort(input, s, 0);
    }
}