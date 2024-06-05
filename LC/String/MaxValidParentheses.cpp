// 最长有效括号﻿
 
// 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
// 示例 1：

// 输入：s = "(()"
// 输出：2
// 解释：最长有效括号子串是 "()"

// 示例 2：

// 输入：s = ")()())"
// 输出：4
// 解释：最长有效括号子串是 "()()"

// 示例 3：

// 输入：s = ""
// 输出：0
// 提示：


#include <iostream>
#include <stack>

using namespace std;
class MaxValidParentheses {
    private:
        string input_;
        stack<char> temp;
    public:
    MaxValidParentheses(const string & input):input_(input){}
    int execute(){
        int output = 0;
        for(auto c : input_){
            if(c == '('){
                temp.push(c);
            } else if (c == ')'){
                if (!temp.empty()) { 
                    temp.pop(); 
                    output++;
                }
            }
        }
        return output*2;
    }
};

int main(){
    std::string input = "(()";
    MaxValidParentheses maxValidParentheses(input);
    auto result = maxValidParentheses.execute();
    std::cout << result << std::endl;
    return 0;
}








