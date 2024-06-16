#include <iostream>
#include <vector>
#include <utility>

// using namespace std;
int QuickSort(std::vector<int> input, int th, int low, int high){
    int choosed = input[low];
    int left = low;
    int right = high - 1;
    while(left < right){
        while(left < right && input[right] >= choosed){
            right--;
        }
        while(left < right && input[left] <= choosed){
            left++;
        }
        std::swap(input[left], input[right]);
    }
    std::swap(choosed, input[left]);
    if(th == left){
        return input[left];
    }else if(th < left){
        QuickSort(input, th, low, left);
    } else {
        QuickSort(input, th, left, high);
    }
};

