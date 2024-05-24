#include <iostream>
#include <bitset>
#include <vector>

using namespace std;

class FindIslandNum{
    int m;
    int n;
    vector<vector<int>> status;
    vector<vector<char>> input_;
    bool CheckValid(int i, int j){
        if(i >= m || j >= n || i < 0 || j < 0){
            return false;
        } else{
            return true;
        }
    }
    void DFS(int i, int j){
        if(CheckValid(i+1,j) && (input_[i+1][j] == '1') && (status[i+1][j] == 0)){
            status[i+1][j] = 1;
            DFS(i+1,j);
        }
        if (CheckValid(i,j+1) && (input_[i][j + 1] == '1') && (status[i][j + 1] == 0)){
            status[i][j + 1] = 1;
            DFS(i,j+1);
        }
        if (CheckValid(i-1,j) && (input_[i-1][j] == '1') && (status[i - 1][j] == 0)){
            status[i-1][j] = 1;
            DFS(i-1,j);
        } 
        if (CheckValid(i,j-1) && (input_[i][j - 1] == '1') && (status[i][j-1] == 0)){
            status[i][j-1] = 1;
            DFS(i,j-1);
        }
        return;
    }
public:
    FindIslandNum(vector<vector<char>> input):input_(input){
        m = input.size();
        n = input[0].size();
        status.resize(m, vector<int>(n, 0));
    }

    int Find(){
        int count = 0;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if((input_[i][j] == '1') && (status[i][j] == 0)){
                    status[i][j] = 1;
                    DFS(i, j);
                    count++;
                }
            }
        }
        return count;
    }
};

int main(){
    vector<vector<char>> input0 = {
        {'0','1', '0'},
        {'1','1', '1'},
        {'0','1', '0'},
    };
    vector<vector<char>> input1 = {
    {'1','1','1','1','0'},
    {'1','1','0','1','0'},
    {'1','1','0','0','0'},
    {'0','0','0','0','0'}
    };

    FindIslandNum findIslandNum(input1);
    int islandNum = findIslandNum.Find();
    std::cout << islandNum << std::endl;
}
