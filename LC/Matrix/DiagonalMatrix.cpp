// # [[1, 2,  3,  4],
// #  [5, 6,  7,  8],
// #  [9, 10, 11, 12]]

// # 1
// # 2 5
// # 3 6 9
// # 4 7 10
// # 8 11
// # 12

#include <vector>
#include <iostream>

using namespace std;
void DialectMatrix(vector<vector<int>> input){
    int m = input.size();
    int n = input[0].size();
    for(int i = 0; i < m + n - 1; i++){
        for(int k = 0; k < i + 1; k++){
            if(k < m && i - k < n){
                    std::cout << " " << input[k][i - k];
            }
        }
        std::cout << std::endl;
  }
}