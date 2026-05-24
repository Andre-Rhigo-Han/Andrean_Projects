#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <algorithm>

using namespace std;

// 表示地图上一个点的坐标
struct Point {
    int x, y;
};

// 用于BFS队列的节点，包含位置和到达该位置的步数
struct Node {
    Point pos;
    int step;
};

// 定义四个移动方向：上、右、下、左
int dx[4] = {-1, 0, 1, 0};
int dy[4] = {0, 1, 0, -1};

class WarehouseRobot {
private:
    vector<string> map; // 存储地图数据，每个字符代表一个格子
    int M, N, K; // 地图的行数(M)、列数(N)、机器人最大电量(步数K)
    Point S, T, E; // 起点(S)、货物点(T)、终点(E)的坐标
    vector<Point> finalPath; // 存储最终的完整路径坐标序列
    vector<bool> isFirstPart; // 标记路径中的每个点是否属于第一阶段(S->T)
    
public:
    // 读取输入数据并初始化地图
    bool readInput() {
        cin >> M >> N >> K;
        map.resize(M);
        
        for (int i = 0; i < M; i++) {
            cin >> map[i];
            for (int j = 0; j < N; j++) {
                if (map[i][j] == 'S') {
                    S.x = i;
                    S.y = j;
                } else if (map[i][j] == 'T') {
                    T.x = i;
                    T.y = j;
                } else if (map[i][j] == 'E') {
                    E.x = i;
                    E.y = j;
                }
            }
        }
        return true;
    }
    
    // 检查坐标(x,y)是否在地图范围内且可通行（不是障碍物'1'）
    bool canGo(int x, int y) {
        if (x < 0 || x >= M || y < 0 || y >= N) return false;
        return map[x][y] != '1';
    }
    
    // 使用BFS算法搜索从start到end的最短路径，返回路径坐标序列
    vector<Point> bfs(Point start, Point end) {
        vector<vector<bool>> visited(M, vector<bool>(N, false)); // 记录已访问的节点
        vector<vector<Point>> parent(M, vector<Point>(N, {-1, -1})); // 记录每个节点的前驱节点
        queue<Node> q; // BFS队列
        
        q.push({start, 0});
        visited[start.x][start.y] = true;
        
        while (!q.empty()) {
            Node cur = q.front();
            q.pop();
            // 如果到达目标点，回溯构建路径
            if (cur.pos.x == end.x && cur.pos.y == end.y) {
                vector<Point> path;
                Point p = end;
                // 通过parent数组从终点反向回溯到起点
                while (p.x != start.x || p.y != start.y) {
                    path.push_back(p);
                    p = parent[p.x][p.y];
                }
                path.push_back(start);
                reverse(path.begin(), path.end()); // 反转路径，得到从起点到终点的顺序
                return path;
            }
            
            // 向四个方向移动
            for (int i = 0; i < 4; i++) {
                int nx = cur.pos.x + dx[i];
                int ny = cur.pos.y + dy[i];
                if (canGo(nx, ny) && !visited[nx][ny]) {
                    visited[nx][ny] = true;
                    parent[nx][ny] = cur.pos; // 记录前驱节点
                    q.push({{nx, ny}, cur.step + 1});
                }
            }
        }
        return {}; // 如果无法到达终点，返回空路径
    }
    
    // 规划完整路径：S->T->E
    int findPath() {
        finalPath.clear();
        isFirstPart.clear();
        
        // S到T
        vector<Point> path1 = bfs(S, T);
        if (path1.empty()) return -1; // 无法到达货物点
        
        // T到E
        vector<Point> path2 = bfs(T, E);
        if (path2.empty()) return -1; // 无法到达终点
        
        // 检查步数
        if (path1.size() + path2.size() - 1 > K) return -2; // 电量不足
        
        // 合并路径
        finalPath = path1;
        for (int i = 1; i < path2.size(); i++) {
            finalPath.push_back(path2[i]); // 从path2的第二个点开始添加，跳过T点
        }
        
        // 标记第一阶段
        isFirstPart.resize(finalPath.size(), false);
        for (int i = 0; i < path1.size(); i++) {
            isFirstPart[i] = true; // 第一阶段路径标记为true
        }
        
        return 0; // 路径规划成功
    }
    
    // 输出最终结果
    void printResult() {
        // 输出路径坐标序列
        for (int i = 0; i < finalPath.size(); i++) {
            cout << "(" << finalPath[i].x << "," << finalPath[i].y << ")";
            if (i < finalPath.size() - 1) cout << "->";
        }
        cout << endl;
        
        cout << "Total Steps:" << finalPath.size() - 1 << endl;
        cout << "Status: Task Completed" << endl;
        
        showPath(); // 显示可视化路径图
    }
    
    void showPath() {
        cout << "Visualized Route:" << endl;
        
        vector<string> showMap = map; // 复制原始地图
        
        // 在地图上标记路径
        for (int i = 0; i < finalPath.size(); i++) {
            Point p = finalPath[i];
            char ch = showMap[p.x][p.y];
            // 只在普通空地(非S/T/E)上标记路径
            if (ch != 'S' && ch != 'T' && ch != 'E') {
                if (isFirstPart[i]) {
                    showMap[p.x][p.y] = '.'; // 第一阶段路径用'.'标记
                } else {
                    showMap[p.x][p.y] = '*'; // 第二阶段路径用'*'标记
                }
            }
        }
        
        // 输出标记后的地图
        for (int i = 0; i < M; i++) {
            cout << showMap[i] << endl;
        }
    }
};

int main() {
    WarehouseRobot robot;
    
    // 读取输入数据
    if (!robot.readInput()) {
        cout << "Input Error" << endl;
        return 1;
    }
    
    int result = robot.findPath();
    
    // 进行路径规划，获取结果状态码
    if (result == 0) {
        robot.printResult(); // 成功找到路径
    } else if (result == -1) {
        cout << "NO SOLUTION" << endl; // 无解
    } else if (result == -2) {
        cout << "LOW POWER" << endl; // 电量不足
    }
    
    return 0;
}