// 输入是 "()[]"
// 输出是 true

#include <iostream>
#include <unordered_map>
#include <stack>
#include <string>
using namespace std;
class FindValidParenteses {
    public:
    bool Execute(const string & input){
        stack<char> st;
        std::unordered_map<char, char> map = {
            {'{', '}'},
            {'(', ')'},
            {'[', ']'}
        };
        for(auto & c : input){
            if (map.find(c) != map.end()) {
                st.push(map[c]);
            }
            if (c == st.top()){
                st.pop();
            }
        }
        return st.size() == 0;
    }
};

int main(){
    string input = "()[";
    FindValidParenteses findValidParenteses;
    auto result = findValidParenteses.Execute(input);
    std::cout << result << std::endl;
}