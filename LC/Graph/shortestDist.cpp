#include <vector>
#include <priority_queue>
using namespace std;
class Solution{
    public:
    int ComputeShortedDist(int n, vector<vector<vector<int>> input){
        vector<pair<int, int>> e;
        for(auto & [u, v, w] : input){
            e[u] = {v, w};
            e[v] = {u, w};
        }
        vector<int> dis(input.size(), std::number_limit<max>(int));
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
        q.push({0,0});
        dis[0] = 0;
        int t = 0;
        while(!q.empty()){
            auto [u, w] = q.top();
            q.pop();
            for(auto & [v, w1] : e[u]){
                if(w + w1 < dis[v]){
                    dis[v] = w + w1;
                    dis[u] = dis[v];
                    q.push(v, w1);
                }
            };
        }
        return dis[n];
    }
};