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


/**Day 18 Wednesday 26/4*/
// int addDigits(int num) {
//     if (num == 0) return 0;
//     if (num % 9 == 0) return 9;
//     return num % 9;
// }


/**Day 19 Thursday 27/4*/
//int bulbSwitch(int n) {
//    return (int) Math.sqrt(n);
//}


/**Day 20 Friday 28/4*/
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


/**Day 21 Saturday 29/4 REVISIT(DISJOINT-SET UNION)*/
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


/**Day 22 Sunday 30/4*/
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


/**Day 23 Monday 1/5*/
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


/**Day 24 Tuesday 2/5*/
int arraySign(vector<int>& nums) {
    int ans = 1;
    for (int num : nums) {
        if (num < 0)
            ans *= -1;
        else if (num == 0)
            return 0;
    }
    return ans;
}

int main() {
    Solution sol;
    vector<vector<int>> v({{3, 1, 2},
                           {3, 2, 3},
                           {1, 1, 3},
                           {1, 2, 4},
                           {1, 1, 2},
                           {2, 3, 4}});
    cout << sol.maxNumEdgesToRemove(4, v);
}
