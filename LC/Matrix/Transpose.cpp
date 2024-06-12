// 0 1 2
// 3 4 5

// 0 3
// 1 4
// 2 5

// 0 3 1
// 4 2 5

// (i, j) @ (m, n) 转置后的位置 > (j, i) @ (n, m) -视为 (m, n) 行中中的座标是 (i', j')
// // 0,1 -> 0,2 -> 1,1 -> 1,0 -> 0,1

// (j * m + i) / n, (j * m + i) % n

#include <iostream>
#include <vector>
using namespace std;
#define OFFSET(point, a, b, N) point[a * N + b]

void ChangeValue(int* A, int i, int j, int m, int n, int input, int count, int min){
    if(min == i * n + j && count != 0){
        return;
    }
    int temp = OFFSET(A, (j * m + i) / n, (j * m + i) % n, n);
    OFFSET(A, (j * m + i) / n, (j * m + i) % n, n) = input;
    ChangeValue(A, (j * m + i) / n, (j * m + i) % n, m, n, temp, ++count, min);
}

int FindMin(int* A, int i, int j, int m, int n, int count, int min){
    if(min == i * n + j && count != 0){
        return min;
    }
    int cur = i * n + j;
    if (cur < min){
        min = cur;
    }
    return FindMin(A, (j * m + i) / n, (j * m + i) % n, m, n, ++count, min);;
}

void Transpose(int* A, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(i * n + j == FindMin(A, i, j, m, n, 0, i * n + j)){
                ChangeValue(A, i, j, m, n, OFFSET(A, i, j, n), 0, i * n + j);
            }
        }
    }
};


int main() {
    int m = 4;
    int n = 4;
    std::vector<int> input(m * n, 0);
    for(int i = 0; i < m * n; i++){
        input[i] = i;
    }
    Transpose(input.data(), m, n);
    for(int i = 0; i < m * n; i++){
        std::cout << input[i] << std::endl;
    }
}