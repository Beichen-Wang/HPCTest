
// INT_MAX [5, 7, 9, 8, 11, 12] INT_MAX
// 找到局部最小值

#include <vector>
#include <iostream>

using namespace std;

inline int FindMin(int start, int end){
    return start + (start + end) >> 1;
}

int FindLocalMinValue(vector<int> input){
    int start = 0;
    int end = input.size() - 1;

    int min;
    switch (input.size()){
        case 0:
            std::cerr << "input is none" << std::endl;
            return -1;
        case 1:
            return input[0];
        default:
            if(input[0] < input[1]){
                return input[0];
            } 

            if(input[end] < input[end - 1]){
                return input[end];
            }
            start = 1;
            end = end - 1;
            while(start <= end){
                min = FindMin(start, end);
                if(input[min] > input[min - 1]){
                    end = min - 1;
                    continue;
                } 
                if(input[min] > input[min + 1]){
                    start = min + 1;
                    continue;
                }
                return input[min];
            }
    }
}

