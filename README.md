# Smart Warehouse AGV Path Planning

A C++ implementation of a grid-based path planner for an automated guided vehicle (AGV) in a warehouse-like environment.

The planner computes a valid shortest route for an AGV that starts from a charging station, visits a target shelf, and then reaches a packing station while avoiding obstacles and respecting a maximum step budget.

## Overview

In a warehouse map, each cell represents either a walkable area, an obstacle, or a task-related location. The AGV must complete the workflow in the required order:

```text
Start / Charging Station (S) -> Target Shelf (T) -> End / Packing Station (E)
```

A direct route from `S` to `E` is not considered complete unless the robot visits `T` first. The program therefore solves the problem as a two-stage shortest-path search:

1. Find the shortest path from `S` to `T`.
2. Find the shortest path from `T` to `E`.
3. Merge the two path segments into one complete route.
4. Validate the total route length against the robot's maximum step budget `K`.
5. Print the route and visualize it in the terminal.

## Features

- **Two-stage route planning**: enforces the required `S -> T -> E` task sequence.
- **Shortest-path search**: uses Breadth-First Search (BFS) on an unweighted grid.
- **Obstacle handling**: treats cells marked as `1` as blocked and avoids them during search.
- **Battery / step-budget validation**: checks whether the planned route fits within the maximum allowed movement steps.
- **Path reconstruction**: records parent nodes during BFS and reconstructs the full coordinate sequence after reaching the target.
- **Terminal visualization**: displays the planned route directly on the input map.
  - `.` marks the first stage: `S -> T`
  - `*` marks the second stage: `T -> E`

## Map Symbols

| Symbol | Meaning |
|---|---|
| `S` | Start point / charging station |
| `T` | Target shelf / pickup point |
| `E` | End point / packing station |
| `0` | Walkable cell |
| `1` | Obstacle |

## Implementation Approach

The project is implemented in C++ using only the standard library.

### 1. Grid Representation

The warehouse map is stored as:

```cpp
vector<string> map;
```

Each string represents one row of the grid. This keeps the map compact and makes cell access straightforward through `map[x][y]`.

The core positions are represented by a simple `Point` structure:

```cpp
struct Point {
    int x, y;
};
```

During input parsing, the program scans the grid once to locate `S`, `T`, and `E`.

### 2. BFS Shortest-Path Search

The `bfs(start, end)` function performs Breadth-First Search from a given start point to a given destination.

For each BFS run, the implementation maintains:

```cpp
vector<vector<bool>> visited;
vector<vector<Point>> parent;
queue<Node> q;
```

- `visited` prevents revisiting the same cell.
- `parent` stores the previous cell of each visited position, allowing the final path to be reconstructed.
- `queue<Node>` supports level-order BFS traversal, which guarantees the shortest path in an unweighted grid.

Movement is limited to four directions:

```text
up, right, down, left
```

A helper function checks whether a candidate cell is inside the grid and not blocked by an obstacle.

### 3. Multi-Stage Path Planning

The full path is produced by running BFS twice:

```text
path1 = BFS(S, T)
path2 = BFS(T, E)
```

If either segment cannot be found, the planner reports that no valid route exists.

When both segments are available, the final route is created by appending `path2` after `path1` while skipping the duplicated `T` node at the beginning of `path2`.

### 4. Step Budget Check

After constructing the route, the program checks whether the planned movement can be completed within the maximum step budget `K`.

The output reports the total number of movement steps as:

```text
number of coordinates in the final route - 1
```

If the route exceeds the available step budget, the program reports `LOW POWER`.

### 5. Route Visualization

The visualization is generated from a copy of the original map. The program keeps the original task markers `S`, `T`, and `E`, then marks ordinary path cells as:

- `.` for the first stage from `S` to `T`
- `*` for the second stage from `T` to `E`

This makes it easy to distinguish the pickup route from the delivery route.

## Input Format

```text
M N K
<row 1>
<row 2>
...
<row M>
```

Where:

- `M` is the number of rows.
- `N` is the number of columns.
- `K` is the maximum number of movement steps supported by the AGV.
- The following `M` lines describe the warehouse grid.

## Output

If a valid route exists and satisfies the step budget, the program prints:

1. The full coordinate sequence.
2. The total number of steps.
3. The task status.
4. A visualized route map.

If planning fails, the program prints one of the following messages:

| Output | Meaning |
|---|---|
| `NO SOLUTION` | The target shelf or final destination cannot be reached. |
| `LOW POWER` | A route exists, but it exceeds the maximum step budget. |

## Example

### Input

```text
5 5 20
S0100
00100
00T00
11100
0000E
```

### Output

```text
(0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3)->(3,3)->(4,3)->(4,4)
Total Steps:8
Status: Task Completed
Visualized Route:
S0100
.0100
..T*0
111*0
000*E
```

## Complexity Analysis

For a grid with `M` rows and `N` columns:

- Each BFS traversal visits each cell at most once.
- The planner runs BFS twice: once for `S -> T` and once for `T -> E`.

Therefore:

```text
Time Complexity:  O(M * N)
Space Complexity: O(M * N)
```

## Build and Run

Compile with a C++20-compatible compiler:

```bash
g++ -Wall -std=c++20 -O2 main.cpp -o main
```

Run the executable:

```bash
./main
```

Or provide input from a file:

```bash
./main < input.txt
```

## File Structure

```text
.
├── main.cpp     # Core AGV path-planning implementation
└── README.md    # Project documentation
```

## Tech Stack

- Language: C++
- Standard: C++20
- Core data structures: `vector`, `queue`, grid matrix, parent matrix
- Algorithm: Breadth-First Search (BFS)
- External dependencies: none
