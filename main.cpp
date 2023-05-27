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


/**Day 10 Sunday 16/4*/
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


/**Day 12 Tuesday 18/4*/
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


/**Day 13 Wednesday 19/4*/
//struct TreeNode {
//    int val;
//    TreeNode *left;
//    TreeNode *right;
//    TreeNode() : val(0), left(nullptr), right(nullptr) {}
//    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
//    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
//};
//
//class Solution {
//public:
//    int widthOfBinaryTree(TreeNode* root) {
//        int max_width{0}, current_width{0};
//        int size;
//        pair<TreeNode*, int> current, first_in_level;
//        queue<pair<TreeNode*, int>> bfs_queue;
//        bfs_queue.push({root, 1});
//        while (!bfs_queue.empty()) {
//            size = bfs_queue.size();
//            current_width = 0;
//            first_in_level = bfs_queue.front();
//            while (size--) {
//                current = bfs_queue.front();
//                current_width = max(current_width, current.second - first_in_level.second);
//                bfs_queue.pop();
//                if (current.first->left) {
//                    bfs_queue.push({current.first->left, current.second * 2});
//                }
//                if (current.first->right)
//                    bfs_queue.push({current.first->right, current.second * 2 + 1});
//            }
//            max_width = max(max_width, current_width);
//        }
//        return max_width + 1;
//    }
//};


/**Day 14 Friday 21/4 REVISIT, Incorrect*/
//class Solution {
//private:
//    int mem[101][101][101];
//public:
//    int takeOrLeave(int n, int m, int start, vector<int>& group, vector<int>& profit) {
//        m = max(m, 0);
//
//        if (start >= group.size() || n <= 0)
//            return m <= 0;
//
//        if (mem[n][m][start] != -1)
//            return mem[n][m][start];
//
//        mem[n][m][start] = ((m <= 0) + takeOrLeave(n - group[start], m - profit[start], start + 1, group, profit) +
//                takeOrLeave(n, m, start + 1, group, profit)) % (int)(1e9 + 7);
//
//        return mem[n][m][start];
//    }
//
//    int profitableSchemes(int n, int minProfit, vector<int>& group, vector<int>& profit) {
//        memset(mem, -1, sizeof mem);
//        return takeOrLeave(n, minProfit, 0, group, profit) % (int)(1e9 + 7);
//    }
//};


/**Day 15 Saturday 22/4*/
//class Solution {
//private:
//    int mem[501][501];
//    int countInsertions(const string &s, int start, int end) {
//        if (start >= end)
//            return 0;
//
//        if (mem[start][end] != -1)
//            return mem[start][end];
//
//        if (s[start] == s[end])
//            mem[start][end] = countInsertions(s, start + 1, end - 1);
//        else
//            mem[start][end] = 1 + min(countInsertions(s, start + 1, end), countInsertions(s, start, end - 1));
//
//        return mem[start][end];
//    }
//public:
//    int minInsertions(string s) {
//        memset(mem, -1, sizeof mem);
//        return countInsertions(s, 0, s.size() - 1);
//    }
//};


/**Day 16 Sunday 23/4*/
//class Solution {
//private:
//    int dp[100001] = {0};
//    int mod = 1e9 + 7;
//
//    int count(const string &s, int k, int start) {
//
//        if (start == s.size())
//            return 1;
//
//        if (s[start] == '0')
//            return 0;
//
//        if (dp[start])
//            return dp[start];
//
//        int cnt = 0;
//
//        string temp;
//        for (int i = 1; i <= s.size() - start; i++) {
//            temp = s.substr(start, i);
//            if (stoi(temp) > k) {
//                break;
//            }
//            cnt = (cnt % mod + count(s, k, start + i) % mod) % mod;
//        }
//
//        cnt %= mod;
//        dp[start] = cnt;
//        return cnt;
//    }
//
//public:
//    int numberOfArrays(string s, int k) {
//        return count(s, k, 0);
//    }
//};


/**Day 17 Monday 24/4*/
//int lastStoneWeight(vector<int>& stones) {
//    priority_queue<int> stones_queue;
//    for (int stone : stones)
//        stones_queue.push(stone);
//
//    int stone1, stone2;
//    while (stones_queue.size() > 1) {
//        stone1 = stones_queue.top();
//        stones_queue.pop();
//        stone2 = stones_queue.top();
//        stones_queue.pop();
//
//        if (stone1 - stone2) {
//            stones_queue.push(stone1 - stone2);
//        }
//    }
//    if (stones_queue.empty())
//        return 0;
//    return stones_queue.top();
//}


/**Day 18 Tuesday 25/4*/
//class SmallestInfiniteSet {
//private:
//    int infinite_set_top;
//    set<int> added_set;
//public:
//    SmallestInfiniteSet(): infinite_set_top(1) {}
//
//    int popSmallest() {
//        if (added_set.empty()) {
//            return infinite_set_top++;
//        } else {
//            int rv = *added_set.begin();
//            added_set.erase(rv);
//            return rv;
//        }
//    }
//
//    void addBack(int num) {
//        if (num < infinite_set_top)
//            added_set.insert(num);
//    }
//};


/**Day 19 Wednesday 26/4*/
// int addDigits(int num) {
//     if (num == 0) return 0;
//     if (num % 9 == 0) return 9;
//     return num % 9;
// }


/**Day 20 Thursday 27/4*/
//int bulbSwitch(int n) {
//    return (int) Math.sqrt(n);
//}


/**Day 21 Friday 28/4*/
//bool matched(string s1, string s2) {
//    int differences = 0;
//    for (int i = 0; i < s1.size() && differences < 3; i++) {
//        if (s1[i] != s2[i])
//            differences++;
//    }
//    return differences < 3;
//}
//
//int count_groups(unordered_set<string> &current_set) {
//    if (current_set.empty())
//        return 0;
//    vector<string> new_set;
//    new_set.push_back(*current_set.begin());
//    current_set.erase(current_set.begin());
//    for (int i = 0; i < new_set.size(); i++) {
//        for (auto it = current_set.begin(); it != current_set.end();) {
//            if (matched(new_set[i], *it)) {
//                new_set.push_back(*it);
//                current_set.erase(it++);
//            } else
//                it++;
//        }
//    }
//
//    return 1 + count_groups(current_set);
//}
//
//int numSimilarGroups(vector<string> &strs) {
//    unordered_set<string> initial_set;
//    for (const string &str: strs) {
//        initial_set.insert(str);
//    }
//
//    return count_groups(initial_set);
//}


/**Day 22 Saturday 29/4 REVISIT(DISJOINT-SET UNION)*/
//class UnionFind {
//public:
//    vector<int> group;
//    vector<int> rank;
//
//    UnionFind(int size) {
//        group = vector<int>(size);
//        rank = vector<int>(size);
//        for (int i = 0; i < size; ++i) {
//            group[i] = i;
//        }
//    }
//
//    int find(int node) {
//        if (group[node] != node) {
//            group[node] = find(group[node]);
//        }
//        return group[node];
//    }
//
//    void join(int node1, int node2) {
//        int group1 = find(node1);
//        int group2 = find(node2);
//
//        // node1 and node2 already belong to same group.
//        if (group1 == group2) {
//            return;
//        }
//
//        if (rank[group1] > rank[group2]) {
//            group[group2] = group1;
//        } else if (rank[group1] < rank[group2]) {
//            group[group1] = group2;
//        } else {
//            group[group1] = group2;
//            rank[group2] += 1;
//        }
//    }
//
//    bool areConnected(int node1, int node2) {
//        int group1 = find(node1);
//        int group2 = find(node2);
//        return group1 == group2;
//    }
//};
//
//
//class Solution {
//public:
//    // Sort in increasing order based on the 3rd element of the array.
//    bool static compare(vector<int>& a, vector<int>& b) {
//        return a[2] < b[2];
//    }
//
//    vector<bool> distanceLimitedPathsExist(int n, vector<vector<int>>& edgeList, vector<vector<int>>& queries) {
//        UnionFind uf(n);
//        int queriesCount = queries.size();
//        vector<bool> answer(queriesCount);
//
//        // Store original indices with all queries.
//        vector<vector<int>> queriesWithIndex(queriesCount);
//        for (int i = 0; i < queriesCount; ++i) {
//            queriesWithIndex[i] = queries[i];
//            queriesWithIndex[i].push_back(i);
//        }
//
//        int edgesIndex = 0;
//
//        // Sort all edges in increasing order of their edge weights.
//        sort(edgeList.begin(), edgeList.end(), compare);
//        // Sort all queries in increasing order of the limit of edge allowed.
//        sort(queriesWithIndex.begin(), queriesWithIndex.end(), compare);
//
//        // Iterate on each query one by one.
//        for (auto& query : queriesWithIndex) {
//            int p = query[0];
//            int q = query[1];
//            int limit = query[2];
//            int queryOriginalIndex = query[3];
//
//            // We can attach all edges which satisfy the limit given by the query.
//            while (edgesIndex < edgeList.size() && edgeList[edgesIndex][2] < limit) {
//                int node1 = edgeList[edgesIndex][0];
//                int node2 = edgeList[edgesIndex][1];
//                uf.join(node1, node2);
//                edgesIndex += 1;
//            }
//
//            // If both nodes belong to the same component, it means we can reach them.
//            answer[queryOriginalIndex] = uf.areConnected(p, q);
//        }
//
//        return answer;
//    }
//};


/**Day 23 Sunday 30/4*/
//class Solution {
//private:
//    vector<vector<int>> type_1_edges;
//    vector<vector<int>> type_2_edges;
//    vector<vector<int>> type_3_edges;
//
//    int removable_edges = 0;
//
//    vector<int> groups;
//
//    int current_group;
//
//    void classify_edges(vector<vector<int>> &edges) {
//        for (vector<int> edge: edges) {
//            if (edge[0] == 1)
//                type_1_edges.push_back(edge);
//            else if (edge[0] == 2)
//                type_2_edges.push_back(edge);
//            else
//                type_3_edges.push_back(edge);
//        }
//    }
//
//
//    void place_type_3_edges() {
//        int u, v, temp;
//        for (vector<int> edge: type_3_edges) {
//            u = edge[1];
//            v = edge[2];
//            if (groups[u] && groups[u] == groups[v]) {
//                removable_edges++;
//                continue;
//            }
//            if (groups[u]) {
//                if (groups[v]) {
//                    temp = groups[v];
//                    for (int &g: groups)
//                        if (g == temp)
//                            g = groups[u];
//                } else {
//                    groups[v] = groups[u];
//                }
//            } else if (groups[v]) {
//                groups[u] = groups[v];
//            } else {
//                groups[u] = groups[v] = ++current_group;
//            }
//        }
//    }
//
//    bool place_edges(vector<vector<int>> &edges, vector<int> groups) {
//        int u, v, temp;
//        for (vector<int> edge: edges) {
//            u = edge[1];
//            v = edge[2];
//            if (groups[u] && groups[u] == groups[v]) {
//                removable_edges++;
//                continue;
//            }
//            if (groups[u]) {
//                if (groups[v]) {
//                    temp = groups[v];
//                    for (int &g: groups)
//                        if (g == temp)
//                            g = groups[u];
//                } else {
//                    groups[v] = groups[u];
//                }
//            } else if (groups[v]) {
//                groups[u] = groups[v];
//            } else {
//                groups[u] = groups[v] = ++current_group;
//            }
//        }
//        for (int i = 1; i < groups.size() - 1; i++) {
//            if (groups[i] != groups[i + 1] || (!groups[i] || !groups[i + 1]))
//                return false;
//        }
//        return true;
//    }
//
//public:
//    int maxNumEdgesToRemove(int n, vector<vector<int>> &edges) {
//        classify_edges(edges);
//        groups = vector<int>(n + 1);
//        current_group = 0;
//        place_type_3_edges();
//        if (place_edges(type_1_edges, groups) && place_edges(type_2_edges, groups))
//            return removable_edges;
//        return -1;
//    }
//};


/**Day 24 Monday 1/5*/
//double average(vector<int>& salary) {
//    int min = salary[0], max = salary[0];
//    int sum = 0;
//    for (int num : salary) {
//        sum += num;
//        if (num < min)
//            min = num;
//        if (num > max)
//            max = num;
//    }
//    return (sum - min - max)/(salary.size() - 2.0);
//}


/**Day 25 Tuesday 2/5*/
//int arraySign(vector<int>& nums) {
//    int ans = 1;
//    for (int num : nums) {
//        if (num < 0)
//            ans *= -1;
//        else if (num == 0)
//            return 0;
//    }
//    return ans;
//}


/**Day 26 Wednesday 3/5*/
//def findDifference(self, nums1, nums2):
//    """
//    :type nums1: List[int]
//    :type nums2: List[int]
//    :rtype: List[List[int]]
//    """
//    set1 = set(nums1)
//    set2 = set(nums2)
//    list1 = [num for num in set1 if num not in set2]
//    list2 = [num for num in set2 if num not in set1]
//    return [list1, list2]


/**Day 27 Thursday 4/5*/
//string predictPartyVictory(string senate) {
//    vector<bool> can_vote(senate.size(), true);
//    int r_votes{0}, d_votes{0};
//    int r = count(senate.begin(), senate.end(), 'R');
//    int d = senate.size() - r;
//    int s = senate.size();
//    for (int i = 0; i < senate.size(); i++) {
//        if (!can_vote[i % s])
//            continue;
//        if (senate[i % s] == 'R') {
//            if (d_votes) {
//                can_vote[i % s] = false;
//                r--;
//                d_votes--;
//            } else {
//                r_votes++;
//            }
//        } else {
//            if (r_votes) {
//                can_vote[i % s] = false;
//                d--;
//                r_votes--;
//            } else {
//                d_votes++;
//            }
//        }
//    }
//    for (int i = 0; i < INT_MAX; i++) {
//        if (!can_vote[i % s])
//            continue;
//        if (senate[i % s] == 'R') {
//            if (d_votes){
//                can_vote[i % s] = false;
//                r--;
//                d_votes--;
//            } else {
//                if (r > d)
//                    return "Radiant";
//                else
//                    r_votes++;
//            }
//        } else {
//            if (r_votes){
//                can_vote[i % s] = false;
//                d--;
//                r_votes--;
//            } else {
//                if (d > r)
//                    return "Dire";
//                else
//                    d_votes++;
//            }
//        }
//    }
//    return "?";
//}


/**Day 28 Friday 5/5*/
//bool isVowel(char c) {
//    return c == 'a' || c == 'e' || c == 'u' || c == 'i' || c == 'o';
//}
//
//int maxVowels(string s, int k) {
//
//    int max_v{0}, current{0};
//    int i;
//    for (i = 0; i < k; i++) {
//        if (isVowel(s[i]))
//            current++;
//    }
//    max_v = current;
//    for (; i < s.size(); i++) {
//        if (isVowel(s[i]))
//            current++;
//        if (isVowel(s[i - k]))
//            current--;
//        max_v = max(max_v, current);
//    }
//    return max_v;
//}


/**Day 29 Saturday 6/5 REVISIT*/
//class Solution:
//    def numSubseq(self, nums: List[int], target: int) -> int:
//        n = len(nums)
//        mod = 10 ** 9 + 7
//        nums.sort()
//
//        answer = 0
//
//        for left in range(n):
//        # Find the insertion position for `target - nums[left]`
//        # 'right' equals the insertion index minus 1.
//        right = bisect.bisect_right(nums, target - nums[left]) - 1
//        if right >= left:
//        answer += pow(2, right - left, mod)
//        return answer % mod
//


/**Day 30 Sunday 7/5*/
//DP solution -> n^2 :(
//class Solution {
//private:
//    int findSubSeqLength(int idx, vector<int> &obstacles, vector<int> &memo) {
//        if (idx < 0)
//            return 0;
//        if (memo[idx])
//            return memo[idx];
//        int length = 1;
//        int sub_length = 0;
//
//        for (int i = idx - 1; i >= length - 1; i--) {
//                sub_length = findSubSeqLength(i, obstacles, memo);
//                if (obstacles[i] <= obstacles[idx])
//                    length = max(length, 1 + sub_length);
//
//        }
//
//        memo[idx] = length;
//        return length;
//    }
//
//public:
//    vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles) {
//        vector<int> memo(obstacles.size());
//        findSubSeqLength(obstacles.size() - 1, obstacles, memo);
//
//        return memo;
//    }
//};
//
//Binary search -> nlogn :)
//class Solution {
//public:
//    vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles) {
//        int n = obstacles.size();
//
//        // lis[i] records the lowest increasing sequence of length i + 1.
//        vector<int> answer(n, 1), lis;
//
//        for (int i = 0; i < n; ++i) {
//            // Find the rightmost insertion position idx.
//            int idx = upper_bound(lis.begin(), lis.end(), obstacles[i]) - lis.begin();
//            if (idx == lis.size())
//                lis.push_back(obstacles[i]);
//            else
//                lis[idx] = obstacles[i];
//            answer[i] = idx + 1;
//        }
//        return answer;
//    }
//};


/**Day 31 Monday 8/5*/
//int diagonalSum(vector<vector<int>>& mat) {
//    int ans = 0;
//    int s = mat.size();
//    for (int i = 0; i < s; i++) {
//        ans += mat[i][i];
//        if (s - i - 1 != i)
//            ans += mat[s - i - 1][i];
//    }
//    return ans;
//}


/**Day 32 Tuesday 9/5*/
//vector<int> ans;
//
//void traverse_square(vector<vector<int>> &matrix, int start_row, int start_col, int end_row, int end_col) {
//    if (start_row > end_row || start_col > end_col)
//        return;
//
//    for (int i = start_col; i <= end_col; i++) {
//        ans.push_back(matrix[start_row][i]);
//    }
//    for (int i = start_row + 1; i <= end_row; i++) {
//        ans.push_back(matrix[i][end_col]);
//    }
//    if (start_row < end_row)
//        for (int i = end_col - 1; i >= start_col; i--) {
//            ans.push_back(matrix[end_row][i]);
//        }
//    if (start_col < end_col)
//        for (int i = end_row - 1; i > start_row; i--) {
//            ans.push_back(matrix[i][start_col]);
//        }
//
//    traverse_square(matrix, start_row + 1, start_col + 1, end_row - 1, end_col - 1);
//}
//
//
//vector<int> spiralOrder(vector<vector<int>>& matrix) {
//    traverse_square(matrix, 0, 0, matrix.size() - 1, matrix[0].size() - 1);
//    return ans;
//}


/**Day 33 Wednesday 10/5*/
//class Solution {
//public:
//
//    int floorMod(int x, int y) {
//        return ((x % y) + y) % y;
//    }
//
//    vector<vector<int>> generateMatrix(int n) {
//        vector<vector<int>> result (n, vector<int>(n));
//        int cnt = 1;
//        int dir[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
//        int d = 0;
//        int row = 0;
//        int col = 0;
//        while (cnt <= n * n) {
//            result[row][col] = cnt++;
//            int r = floorMod(row + dir[d][0], n);
//            int c = floorMod(col + dir[d][1], n);
//            // change direction if next cell is non zero
//            if (result[r][c] != 0) d = (d + 1) % 4;
//            row += dir[d][0];
//            col += dir[d][1];
//        }
//        return result;
//    }
//};


/**Day 34 Thursday 11/5 REVISIT*/
//class Solution {
//public:
//    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
//        int n1 = nums1.size();
//        int n2 = nums2.size();
//
//        vector<int> dp(n2 + 1), dpPrev(n2 + 1);
//
//        for (int i = 1; i <= n1; i++) {
//            for (int j = 1; j <= n2; j++) {
//                if (nums1[i - 1] == nums2[j - 1]) {
//                    dp[j] = 1 + dpPrev[j - 1];
//                } else {
//                    dp[j] = max(dp[j - 1], dpPrev[j]);
//                }
//            }
//            dpPrev = dp;
//        }
//
//        return dp[n2];
//    }
//};


/**Day 35 Friday 12/5*/
//class Solution {
//private:
//    long long arr[100000];
//
//    long long try_start(vector<vector<int>>& questions, int start) {
//        if (start >= questions.size())
//            return 0;
//
//        if (arr[start])
//            return arr[start];
//
//        long long skipped = try_start(questions, start + 1);
//        long long picked = questions[start][0] + try_start(questions, start + questions[start][1] + 1);
//        arr[start] = max(picked, skipped);
//        return arr[start];
//    }
//
//public:
//    long long mostPoints(vector<vector<int>>& questions) {
//        return try_start(questions, 0);
//    }
//};


/**Day 36 Saturday 13/5*/
//class Solution {
//private:
//    int memo[100001];
//    int mod = 1e9 + 7;
//
//    int count_start(int low, int high, int zero, int one, int current_length) {
//        if (current_length > high)
//            return 0;
//
//        if (memo[current_length] != -1)
//            return memo[current_length];
//
//        int choose_zero, choose_one;
//        choose_zero = count_start(low, high, zero, one, current_length + zero) % mod;
//        choose_one = count_start(low, high, zero, one, current_length + one) % mod;
//        memo[current_length] = ((current_length >= low) + choose_one + choose_zero) % mod;
//        return memo[current_length];
//    }
//
//public:
//    int countGoodStrings(int low, int high, int zero, int one) {
//        memset(memo, -1, sizeof memo);
//        return count_start(low, high, zero, one, 0);
//    }
//};


/**Day 37 Sunday 14/5 REVISIT*/
//class Solution {
//public:
//    int maxScore(vector<int>& nums) {
//        int maxStates = 1 << nums.size(); // 2^(nums array size)
//        int finalMask = maxStates - 1;
//
//        // 'dp[i]' stores max score we can get after picking remaining numbers represented by 'i'.
//        vector<int> dp(maxStates);
//
//        // Iterate on all possible states one-by-one.
//        for (int state = finalMask; state >= 0; state -= 1) {
//            // If we have picked all numbers, we know we can't get more score as no number is remaining.
//            if (state == finalMask) {
//                dp[state] = 0;
//                continue;
//            }
//
//            int numbersTaken = __builtin_popcount(state);
//            int pairsFormed = numbersTaken / 2;
//            // States representing even numbers are taken are only valid.
//            if (numbersTaken % 2) {
//                continue;
//            }
//
//            // We have picked 'pairsFormed' pairs, we try all combinations of one more pair now.
//            // We itearte on two numbers using two nested for loops.
//            for (int firstIndex = 0; firstIndex < nums.size(); firstIndex += 1) {
//                for (int secondIndex = firstIndex + 1; secondIndex < nums.size(); secondIndex += 1) {
//                    // We only choose those numbers which were not already picked.
//                    if (((state >> firstIndex) & 1) == 1 || ((state >> secondIndex) & 1) == 1) {
//                        continue;
//                    }
//                    int currentScore = (pairsFormed + 1) * __gcd(nums[firstIndex], nums[secondIndex]);
//                    int stateAfterPickingCurrPair = state | (1 << firstIndex) | (1 << secondIndex);
//                    int remainingScore = dp[stateAfterPickingCurrPair];
//                    dp[state] = max(dp[state], currentScore + remainingScore);
//                }
//            }
//        }
//
//        // Returning score we get from 'n' remaining numbers of array.
//        return dp[0];
//    }
//};


/**Day 38 Monday 15/5*/
//struct ListNode {
//    int val;
//    ListNode *next;
//
//    ListNode() : val(0), next(nullptr) {}
//
//    ListNode(int x) : val(x), next(nullptr) {}
//
//    ListNode(int x, ListNode *next) : val(x), next(next) {}
//};
//
//class Solution {
//public:
//    ListNode *swapNodes(ListNode *head, int k) {
//        int n{};
//        ListNode *cur = head;
//        while (cur) {
//            n++;
//            cur = cur->next;
//        }
//
//        ListNode *node1, *node2;
//        int val1, val2;
//        cur = head;
//        for (int i = 0; i < n; i++, cur = cur->next) {
//            if (i == k - 1) {
//                val1 = cur->val;
//                node1 = cur;
//            }
//            if (i == n - k) {
//                val2 = cur->val;
//                node2 = cur;
//            }
//        }
//
//        node1->val = val2;
//        node2->val = val1;
//
//        return head;
//    }
//};


/**Day 39 Tuesday 16/5*/
//class Solution {
//public:
//    ListNode* swapPairs(ListNode* head) {
//        ListNode *cur = head, *temp, *prev{nullptr};
//        while (cur && cur->next) {
//            temp = cur->next;
//            cur->next = temp->next;
//            temp->next = cur;
//            if (cur == head)
//                head = temp;
//
//            if (prev)
//                prev->next = temp;
//            prev = cur;
//            cur = cur->next;
//        }
//        return head;
//    }
//};


/**Day 40 Wednesday 17/5*/
//struct ListNode {
//    int val;
//    ListNode *next;
//
//    ListNode() : val(0), next(nullptr) {}
//
//    ListNode(int x) : val(x), next(nullptr) {}
//
//    ListNode(int x, ListNode *next) : val(x), next(next) {}
//};
//
//class Solution {
//private:
//    ListNode *reverse_linked_list(ListNode *head) {
//        if (!head->next)
//            return head;
//
//        ListNode *first, *second, *third;
//        first = head;
//        second = head->next;
//        third = second->next;
//        first->next = nullptr;
//        while (second) {
//            second->next = first;
//            first = second;
//            second = third;
//            if (third)
//                third = third->next;
//        }
//        return first;
//    }
//
//public:
//    int pairSum(ListNode *head) {
//        ListNode *slow{head}, *fast{head};
//        while (fast) {
//            slow = slow->next;
//            fast = fast->next->next;
//        }
//        ListNode *halfHead = reverse_linked_list(slow);
//        int rv = 0;
//        ListNode *leftCur{head}, *rightCur{halfHead};
//        while (leftCur && rightCur) {
//            rv = max(leftCur->val + rightCur->val, rv);
//            leftCur = leftCur->next;
//            rightCur = rightCur->next;
//        }
//
//        return rv;
//
//    }
//};


/**Day 41 Thursday 18/5*/
//class Solution {
//public:
//    vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) {
//        vector<bool> isRoot(n, true);
//        for (vector<int> edge : edges) {
//            isRoot[edge[1]] = false;
//        }
//
//        vector<int> roots;
//        for (int i = 0; i < n; i++) {
//            if (isRoot[i])
//                roots.push_back(i);
//        }
//
//        return roots;
//    }
//};


/**Day 42 Friday 19/5*/
//class Solution {
//private:
//    bool dfs(int node, vector<vector<int>>& graph, vector<int> &groups ,int group) {
//        if (groups[node])
//            return groups[node] == group;
//
//        groups[node] = group;
//
//        for (int neighbour : graph[node]) {
//            if (!dfs(neighbour, graph, groups, 3 - group))
//                return false;
//        }
//        return true;
//    }
//public:
//    bool isBipartite(vector<vector<int>>& graph) {
//        vector<int> groups(graph.size(), 0);
//        bool isBipartite = true;
//
//        for (int node = 0; node < graph.size() && isBipartite; node++) {
//            if (!groups[node]) {
//                isBipartite = dfs(node, graph, groups, 1);
//            }
//        }
//        return isBipartite;
//    }
//};


/**Day 43 Saturday 20/5 NOT MY SOLUTION, REVISIT*/
//class Solution {
//public:
//    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
//        unordered_map<string, unordered_map<string, double>> graph = buildGraph(equations, values);
//        vector<double> results;
//
//        for (const auto& query : queries) {
//            const string& dividend = query[0];
//            const string& divisor = query[1];
//
//            if (graph.find(dividend) == graph.end() || graph.find(divisor) == graph.end()) {
//                results.push_back(-1.0);
//            } else {
//                results.push_back(bfs(dividend, divisor, graph));
//            }
//        }
//
//        return results;
//    }
//
//private:
//    unordered_map<string, unordered_map<string, double>> buildGraph(const vector<vector<string>>& equations, const vector<double>& values) {
//        unordered_map<string, unordered_map<string, double>> graph;
//
//        for (int i = 0; i < equations.size(); i++) {
//            const string& dividend = equations[i][0];
//            const string& divisor = equations[i][1];
//            double value = values[i];
//
//            graph[dividend][divisor] = value;
//            graph[divisor][dividend] = 1.0 / value;
//        }
//
//        return graph;
//    }
//
//    double bfs(const string& start, const string& end, unordered_map<string, unordered_map<string, double>>& graph) {
//        queue<pair<string, double>> q;
//        unordered_set<string> visited;
//        q.push({start, 1.0});
//
//        while (!q.empty()) {
//            string node = q.front().first;
//            double value = q.front().second;
//            q.pop();
//
//            if (node == end) {
//                return value;
//            }
//
//            visited.insert(node);
//
//            for (const auto& neighbor : graph[node]) {
//                const string& neighborNode = neighbor.first;
//                double neighborValue = neighbor.second;
//
//                if (visited.find(neighborNode) == visited.end()) {
//                    q.push({neighborNode, value * neighborValue});
//                }
//            }
//        }
//
//        return -1.0;
//    }
//};


/**Day 44 Sunday 21/5*/
//int rows[4] = {1, 0, -1, 0};
//int cols[4] = {0, 1, 0, -1};
//
//bool isValid(int i, int j, int n) {
//    return i >= 0 && j >= 0 && i < n && j < n;
//}
//
//void dfs(vector<vector<bool>> &visited, int i, int j, vector<vector<int>> &grid, queue<pair<int, int>> &bfs_queue) {
//    visited[i][j] = true;
//    bfs_queue.emplace(i, j);
//    for (int neighbour = 0; neighbour < 4; neighbour++) {
//        int neigh_row = i + rows[neighbour];
//        int neigh_col = j + cols[neighbour];
//        int n = grid.size();
//        if (isValid(neigh_row, neigh_col, n) && !visited[neigh_row][neigh_col] && grid[neigh_row][neigh_col])
//            dfs(visited, neigh_row, neigh_col, grid, bfs_queue);
//    }
//}
//
//void add_neighbours(vector<vector<bool>> &visited, int i, int j, vector<vector<int>> &grid,
//                    queue<pair<int, int>> &bfs_queue) {
//    for (int neighbour = 0; neighbour < 4; neighbour++) {
//        int neigh_row = i + rows[neighbour];
//        int neigh_col = j + cols[neighbour];
//        int n = grid.size();
//        if (isValid(neigh_row, neigh_col, n) && !visited[neigh_row][neigh_col]) {
//            bfs_queue.emplace(neigh_row, neigh_col);
//            visited[neigh_row][neigh_col] = true;
//        }
//    }
//}
//
//int shortestBridge(vector<vector<int>> &grid) {
//    int n = grid.size();
//    queue<pair<int, int>> bfs_queue;
//    vector<vector<bool>> visited(n, vector<bool>(n, false));
//    bool foundFirstIsland = false;
//
//    for (int i = 0; i < n && !foundFirstIsland; i++) {
//        for (int j = 0; j < n && !foundFirstIsland; j++) {
//            if (grid[i][j]) {
//                dfs(visited, i, j, grid, bfs_queue);
//                foundFirstIsland = true;
//            }
//        }
//    }
//
//    int shortest_path = 0;
//    int queue_size;
//    int cur_i, cur_j;
//    bool pathFound = false;
//    while (!bfs_queue.empty() && !pathFound) {
//        queue_size = bfs_queue.size();
//        while (queue_size-- && !pathFound) {
//            cur_i = bfs_queue.front().first;
//            cur_j = bfs_queue.front().second;
//            add_neighbours(visited, cur_i, cur_j, grid, bfs_queue);
//            if (shortest_path && grid[cur_i][cur_j])
//                pathFound = true;
//            bfs_queue.pop();
//        }
//        shortest_path += !pathFound;
//    }
//    return shortest_path - 1;
//}


/**Day 45 Saturday 27/5 WRONG, REVISIT, ZERO SUM*/
//int dp(vector<int> &stoneValue, int current_stone, int moves_left, vector<vector<int>> &memo) {
//    if (moves_left <= 0 || current_stone >= stoneValue.size())
//        return INT_MIN;
//
//    if (memo[current_stone][moves_left - 1] != INT_MIN)
//        return memo[current_stone][moves_left - 1];
//
//    int current_best = INT_MIN;
//
//    for (int i = 0; i < 3; i++) {
//        current_best = max(current_best, dp(stoneValue, current_stone + i + 1, moves_left - i - 1, memo));
//    }
//
//    return memo[current_stone][moves_left - 1] = stoneValue[current_stone] + current_best;
//}
//
//string stoneGameIII(vector<int> &stoneValue) {
//    vector<vector<int>> memo(stoneValue.size(), vector<int>(3, INT_MIN));
//    int alice = dp(stoneValue, 0, 3, memo);
//    int best_first_move = 0;
//    for (int i = 1; i < 3; i++) {
//        if (memo[0][i] > memo[0][best_first_move])
//            best_first_move = 1;
//    }
//
//    if (best_first_move + 1 > stoneValue.size()) {
//        if (memo[0][best_first_move] > 0)
//            return "Alice";
//        else if (memo[0][best_first_move] < 0)
//            return "Bob";
//        else
//            return "Tie";
//    }
//    int bob = memo[best_first_move + 1][0];
//
//    for (int i = 1; i < 3; i++) {
//        bob = max(bob, memo[best_first_move + 1][i]);
//    }
//
//    if (alice > bob)
//        return "Alice";
//    else if (bob > alice)
//        return "Bob";
//    else
//        return "Tie";
//}


int main() {
    vector<int> v = {1, 2, 3, 7};
    cout << stoneGameIII(v);
}
