#include <bits/stdc++.h>

using namespace std;

/**Day 1*/
//bool validPosition(vector<vector<int>> &grid, int row, int col, int height, int width) {
//    return row >= 0 && col >= 0 && row < height && col < width && grid[row][col];
//}
//
//
//void dfsSubIsland(vector<vector<int>> &grid, int row, int col) {
//    int rows[4] = {1, 0, -1, 0};
//    int cols[4] = {0, 1, 0, -1};
//    int height = grid.size();
//    int width = grid[0].size();
//    grid[row][col] = 0;
//    for (int index = 0; index < 4; index++) {
//        if (validPosition(grid, row + rows[index], col + cols[index], height, width))
//            dfsSubIsland(grid, row + rows[index], col + cols[index]);
//    }
//}
//
//
//bool isBoundary(vector<vector<int>> &grid, int row, int col, int height, int width) {
//    return grid[row][col] && (row == 0 || col == 0 || row == height - 1 || col == width - 1);
//}
//
//
//int numEnclaves(vector<vector<int>>& grid) {
//    int height = grid.size();
//    int width = grid[0].size();
//    int ans{0};
//    for (int row = 0; row < height; row++) {
//        for (int col = 0; col < width; col++) {
//            if (isBoundary(grid, row, col, height, width))
//                dfsSubIsland(grid, row, col);
//        }
//    }
//
//    for (int row = 0; row < height; row++) {
//        for (int col = 0; col < width; col++) {
//            if (grid[row][col])
//                ans++;
//        }
//    }
//
//    return ans;
//}


/**Day 2*/
// Definition for a Node.
//class Node {
//public:
//    int val;
//    vector<Node *> neighbors;
//
//    Node() {
//        val = 0;
//        neighbors = vector<Node *>();
//    }
//
//    Node(int _val) {
//        val = _val;
//        neighbors = vector<Node *>();
//    }
//
//    Node(int _val, vector<Node *> _neighbors) {
//        val = _val;
//        neighbors = _neighbors;
//    }
//};
//
//
//void cloneDFS(unordered_set<int> &visited, Node *answer, Node *neighbour) {
//    visited.insert(answer->val);
//    Node *newNeighbour = new Node(neighbour->val);
//    answer->neighbors.push_back(newNeighbour);
//    if (!visited.count(neighbour->val))
//        for (Node *nextNeighbour : neighbour->neighbors)
//            cloneDFS(visited, newNeighbour, nextNeighbour);
//
//}
//
//
//Node *cloneGraph(Node *node) {
//    if (node) {
//        Node *answer = new Node(node->val);
//        unordered_set<int> visited;
//        visited.insert(answer->val);
//        for (Node *neighbour: node->neighbors) {
//            cloneDFS(visited, answer, neighbour);
//        }
//        return answer;
//    }
//    return nullptr;
//}


/**Day 3*/
//unordered_set<int> getRoots(const string &colors, const vector<vector<int>> &edges) {
//    unordered_set<int> roots;
//    for (int i = 0; i < colors.size(); i++) {
//        roots.insert(i);
//    }
//
//    for (vector<int> edge: edges) {
//        if (roots.find(edge[1]) != roots.end())
//            roots.erase(edge[1]);
//    }
//
//    return roots;
//}
//
//
//vector<vector<int>> constructGraph(int numberOfNodes, const vector<vector<int>> &edges) {
//    vector<vector<int>> graph(numberOfNodes);
//
//    for (vector<int> edge: edges)
//        graph[edge[0]].push_back(edge[1]);
//
//    return graph;
//}
//
//
//void updateMaxMap(unordered_map<char, int> &maxColorCount, const vector<int> &colorsCount) {
//    for (auto it = maxColorCount.begin(); it != maxColorCount.end(); it++) {
//        it->second = max(it->second, colorsCount[it->first - 'a']);
//    }
//}
//
//
//int searchNode(unordered_map<char, int> &maxColorCount, vector<int> &colorsCount, const vector<vector<int>> &graph,
//               const string &colors, vector<bool> visited, int node) {
//    visited[node] = true;
//    colorsCount[colors[node] - 'a']++;
//
//    for (int nextNode: graph[node]) {
//        if (!visited[nextNode])
//            searchNode(maxColorCount, colorsCount, graph, colors, visited, nextNode);
//        else
//            return -1;
//    }
//
//    updateMaxMap(maxColorCount, colorsCount);
//    visited[node] = false;
//    colorsCount[colors[node] - 'a']--;
//
//    return 0;
//}
//
//
//int largestPathValue(string colors, vector<vector<int>> &edges) {
//    vector<vector<int>> graph = constructGraph(colors.size(), edges);
//    unordered_set<int> roots = getRoots(colors, edges);
//    unordered_map<char, int> maxColorCount;
//
//    for (char current = 'a'; current <= 'z'; current++)
//        maxColorCount[current] = -1;
//
//    for (int root: roots) {
//        vector<int> colorCount(26, 0);
//        vector<bool> visited(graph.size(), false);
//        if (searchNode(maxColorCount, colorCount, graph, colors, visited, root) == -1)
//            return -1;
//    }
//
//    int ans = -1;
//    for (pair<char, int> color: maxColorCount) {
//        if (ans < color.second)
//            ans = color.second;
//    }
//    return ans;
//}


/**Day 4 Monday 10/4*/
//bool isValid(string s) {
//    stack<char> myStack;
//    unordered_map<char, char> openCloseMap({{']', '['},
//                                            {'}', '{'},
//                                            {')', '('}});
//    for (char c : s) {
//        if (c == '[' || c == '(' || c == '{')
//            myStack.push(c);
//        else {
//            if (myStack.empty() || myStack.top() != openCloseMap[c])
//                return false;
//            else
//                myStack.pop();
//        }
//    }
//
//    return myStack.empty();
//}


/**Day 5 Tuesday 11/4*/
//string removeStars(string s) {
//    string ans = "";
//
//    for (char c : s) {
//        if (c == '*') {
//            if (!ans.empty())
//                ans.pop_back();
//        } else
//            ans.push_back(c);
//    }
//
//    return ans;
//}


/**Day 6 Wednesday 12/4*/
//void add_to_deque(const string &path, deque<string> &deque, int &idx) {
//    string current_dir = "";
//    while (idx < path.size() && (path[idx] != '/'))
//        current_dir += path[idx++];
//    deque.push_back(current_dir);
//}
//
//void pop_deque(deque<string> &deque) {
//    if (!deque.empty())
//        deque.pop_back();
//}
//string simplifyPath(string path) {
//    deque<string> directory_queue;
//    string absolute_path = "";
//    for (int idx = 0; idx < path.size(); idx++) {
//        if (path[idx] != '/') {
//            if (path[idx] != '.') {
//                add_to_deque(path, directory_queue, idx);
//            } else {
//                if ((idx < path.size() - 2 && path[idx + 1] == '.' && path[idx + 2] == '/') ||
//                    (idx == path.size() - 2 && path[idx + 1] == '.')) {
//                    pop_deque(directory_queue);
//                    idx += 1;
//                } else if (idx < path.size() - 1 && path[idx + 1] != '/')
//                    add_to_deque(path, directory_queue, idx);
//            }
//        }
//    }
//
//    while (!directory_queue.empty()) {
//        absolute_path += '/' + directory_queue.front();
//        directory_queue.pop_front();
//    }
//
//    if (absolute_path.size())
//        return absolute_path;
//    else return "/";
//}

/**Day 7 Thursday 13/4*/
//bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
//    int pushed_idx{1}, popped_idx;
//    int size = popped.size();
//    stack<int> my_stack;
//    my_stack.push(pushed[0]);
//
//    for (popped_idx = 0; popped_idx < size; popped_idx++) {
//        while((my_stack.empty() || my_stack.top() != popped[popped_idx]) && pushed_idx < size) {
//            my_stack.push(pushed[pushed_idx++]);
//        }
//        if (my_stack.top() == popped[popped_idx]) {
//            my_stack.pop();
//        } else
//            return false;
//    }
//    return pushed_idx == size && popped_idx == size && my_stack.empty();
//}


/**Day 8 Friday 14/4*/
//int DP[1000][1000] = {0};
//int counter = 0;
//
//int LPS(string &s, int start, int end) {
//    counter++;
//    if (end < start)
//        return 0;
//    if (start == end)
//        return 1;
//    if (DP[start][end])
//        return DP[start][end];
//    if (s[start] == s[end]) {
//        DP[start][end] = 2 + LPS(s, start + 1, end - 1);
//        return DP[start][end];
//    }
//
//    DP[start][end] = (max(LPS(s, start, end - 1), LPS(s, start + 1, end)));
//    return DP[start][end];
//}
//
//int longestPalindromeSubseq(string s) {
//    return LPS(s, 0, s.size() - 1);
//}


/**Day 9 Saturday 15/4 REVISIT*/
//class Solution {
//public:
//    int maxValueOfCoins(vector<vector<int>>& piles, int k) {
//        int n = piles.size();
//        vector dp(n + 1, vector<int>(k + 1));
//        for (int i = 1; i <= n; i++) {
//            for (int coins = 0; coins <= k; coins++) {
//                int currentSum = 0;
//                for (int currentCoins = 0; currentCoins <= min((int)piles[i - 1].size(), coins); currentCoins++) {
//                    if (currentCoins > 0) {
//                        currentSum += piles[i - 1][currentCoins - 1];
//                    }
//                    dp[i][coins] = max(dp[i][coins], dp[i - 1][coins - currentCoins] + currentSum);
//                }
//            }
//        }
//        return dp[n][k];
//    }
//};


/**Day 9 Sunday 16/4*/
//long long ways(const vector<string> &words, const string &target, int ws, int ts, vector<vector<long long>> &DP,
//               const vector<vector<int>> &countChar) {
//    if (ts == target.size())
//        return 1;
//
//    if (DP[ws][ts] != -1)
//        return DP[ws][ts];
//
//    long long ans = 0;
//
//    char targetChar = target[ts] - 'a';
//    for (int i = ws; i < words[0].size() - (target.size() - ts) + 1; i++) {
//        if (countChar[targetChar][i])
//            ans += (countChar[targetChar][i] *
//                    (ways(words, target, i + 1, ts + 1, DP, countChar))) % (1000000000 + 7);
//    }
//
//    DP[ws][ts] = ans % (1000000000 + 7);
//    return DP[ws][ts];
//}
//
//int numWays(vector<string> &words, string target) {
//    vector<vector<long long>> DP(1001, vector<long long>(1001, -1));
//    vector<vector<int>> countChar(26, vector<int>(words[0].size(), 0));
//    for (int i = 0; i < words.size(); i++) {
//        for (int j = 0; j < words[0].size(); j++) {
//            countChar[words[i][j] - 'a'][j]++;
//        }
//    }
//    return ways(words, target, 0, 0, DP, countChar);
//}


/**Day 11 Monday 17/4*/
//string mergeAlternately(string word1, string word2) {
//    int i{0};
//    string ans = "";
//    while (i < word1.size() && i < word2.size()) {
//        ans += word1[i];
//        ans += word2[i];
//        i++;
//    }
//    while (i < word1.size())
//        ans += word1[i++];
//    while (i < word2.size())
//        ans += word2[i++];
//
//    return ans;
//}


/**Day 12 Tuesday 17/4*/
//struct TreeNode {
//    int val;
//    TreeNode *left;
//    TreeNode *right;
//
//    TreeNode() : val(0), left(nullptr), right(nullptr) {}
//
//    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
//
//    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
//};
//
//class Solution {
//private:
//    enum direction {
//        left, right, isRoot
//    };
//
//    int zigzag(TreeNode *root, direction direction_from_parent, int count) {
//        if (!root)
//            return 0;
//
//        if (direction_from_parent == isRoot)
//            return max(zigzag(root->left, left, 1), zigzag(root->right, right, 1));
//
//        int leftZigzag, rightZigzag;
//
//        leftZigzag = zigzag(root->left, left, count * (direction_from_parent != left) + 1);
//        rightZigzag = zigzag(root->right, right, count * (direction_from_parent != right) + 1);
//
//
//        return max(leftZigzag, max(count, rightZigzag));
//    }
//
//public:
//    int longestZigZag(TreeNode *root) {
//        return zigzag(root, isRoot, 0);
//    }
//};

int main() {
    cout << "?";
}
