# Coding Interview Problems - Summary

This document summarizes all the coding problems solved with Python solutions.

---

## 1. Max Coins Collection (Google)

**Problem:** Tokens on a 1D board can move exactly 3 steps right. Maximize coins collected.

**Key Insight:** Chain decomposition - tokens at positions i, i+3, i+6, ... form independent chains.

**Solution:** Dynamic programming on each chain independently.

**Complexity:** O(n) time, O(1) space

**Files:**
- `max_coins_solution.py`
- `max_coins_explanation.md`

---

## 2. Secret Propagation Through Meetings (Google)

**Problem:** People meet at different times. If one knows a secret, both know it after meeting. Find all who know the secret.

**Key Insight:** Union-Find with time-based grouping. Process meetings by timestamp.

**Solution:** 
1. Group meetings by timestamp
2. For each timestamp, use Union-Find to propagate secrets
3. Reset Union-Find for people who don't know secret

**Complexity:** O(m log m + m·α(n)) time, O(n + m) space

**Files:**
- `secret_propagation_solution.py`
- `secret_propagation_explanation.md`

---

## 3. Factorial Calculation

**Problem:** Calculate factorial with considerations for large numbers, data types, and optimization.

**Key Insights:**
- Iterative vs recursive trade-offs
- Integer overflow handling
- Python's arbitrary precision integers
- Optimization strategies (memoization, Stirling's approximation)

**Solution:** 6 different approaches from basic to optimized

**Complexity:** O(n) time, O(1) space for iterative

**Files:**
- `factorial_solution.py`
- `factorial_explanation.md`

---

## 4. Waymo Path Finding Through Circular Obstacles

**Problem:** Can a vehicle travel from bottom to top of a region with circular obstacles?

**CRITICAL INSIGHT:** Don't merge circles into larger circles - this creates artificial blocking areas!

**Correct Approach:**
1. Use Union-Find to group overlapping circles
2. Check if UNION of circles (not merged circle) covers x-range
3. Merge RANGES at each y-level, not circles themselves

**Solution:**
- Build connectivity graph with Union-Find
- For each component, sample y-levels
- Check if ranges cover [x1, x2]

**Complexity:** O(n² + n·m) time, O(n) space

**Files:**
- `waymo_path_finding_solution.py` (original with merging)
- `waymo_improved_solution.py` (corrected approach)
- `waymo_path_finding_explanation.md` (updated with correct approach)
- `waymo_key_insight.md` (explains why not to merge)

**Key Mistake to Avoid:**
- ❌ Merging circles creates larger blocking areas
- ✅ Check UNION coverage preserves actual gaps

---

## 5. Parking Lot Path Finding with Floating-Point Coordinates (ABC)

**Problem:** Given start, end, and obstacles with floating-point coordinates, can car reach destination?

**Key Challenge:** Continuous space (not discrete grid) with floating-point precision issues.

**Solution:** Grid discretization + BFS
1. Convert continuous space to discrete grid
2. Mark obstacle cells (with radius)
3. Run BFS from start to end

**Optimization:** Check straight line path first before BFS

**Complexity:** O(W·H + O·R²) time, O(W·H) space

**Files:**
- `parking_lot_path_finding_solution.py`
- `parking_lot_path_finding_explanation.md`

**Key Techniques:**
- Discretization (continuous → discrete)
- Resolution parameter (accuracy vs speed)
- Obstacle radius (handle floating-point imprecision)
- Point-to-segment distance calculation

---

## Common Patterns Across Problems

### 1. Union-Find (Disjoint Set Union)
**Used in:**
- Secret Propagation
- Waymo Path Finding

**When to use:**
- Grouping/clustering elements
- Tracking connected components
- Dynamic connectivity queries

### 2. Graph Traversal (BFS/DFS)
**Used in:**
- Parking Lot Path Finding

**When to use:**
- Shortest path in unweighted graph
- Reachability queries
- Level-order processing

### 3. Dynamic Programming
**Used in:**
- Max Coins Collection

**When to use:**
- Optimal substructure
- Overlapping subproblems
- Optimization problems

### 4. Discretization
**Used in:**
- Parking Lot Path Finding
- Waymo Path Finding (y-level sampling)

**When to use:**
- Continuous space problems
- Floating-point coordinates
- Need to apply discrete algorithms

### 5. Sweep Line Algorithm
**Used in:**
- Waymo Path Finding

**When to use:**
- Geometric problems
- Range queries
- Interval processing

---

## Interview Tips Summary

### Before Coding

1. **Clarify the problem**
   - Ask about constraints
   - Confirm input/output format
   - Check edge cases

2. **Discuss approach**
   - Explain high-level strategy
   - Mention trade-offs
   - Get interviewer feedback

3. **Analyze complexity**
   - Time and space complexity
   - Justify your approach

### During Coding

1. **Write clean code**
   - Modular functions
   - Meaningful variable names
   - Comments for complex logic

2. **Handle edge cases**
   - Empty input
   - Single element
   - Boundary conditions

3. **Test as you go**
   - Simple test case
   - Edge case
   - Complex scenario

### After Coding

1. **Test thoroughly**
   - Walk through examples
   - Check edge cases
   - Verify correctness

2. **Discuss optimizations**
   - Time/space trade-offs
   - Alternative approaches
   - Real-world considerations

3. **Be ready to modify**
   - Handle follow-up questions
   - Adapt to new constraints
   - Explain design decisions

---

## Common Mistakes to Avoid

### 1. Jumping into Code Too Quickly
❌ Start coding immediately  
✅ Discuss approach first, get feedback

### 2. Not Handling Edge Cases
❌ Only test happy path  
✅ Test empty, single, boundary cases

### 3. Inefficient Algorithms
❌ O(n³) when O(n²) possible  
✅ Analyze and optimize complexity

### 4. Poor Code Organization
❌ One giant function  
✅ Modular, reusable functions

### 5. Not Testing
❌ Assume code works  
✅ Test with examples

### 6. Ignoring Constraints
❌ Treat all coordinates as integers  
✅ Handle floating-point properly

### 7. Overcomplicating
❌ Complex solution for simple problem  
✅ Start simple, optimize if needed

---

## Complexity Cheat Sheet

### Time Complexity
- **O(1):** Constant - hash lookup, array access
- **O(log n):** Logarithmic - binary search, balanced tree
- **O(n):** Linear - single pass through array
- **O(n log n):** Linearithmic - sorting, divide & conquer
- **O(n²):** Quadratic - nested loops
- **O(2ⁿ):** Exponential - recursive subsets
- **O(n!):** Factorial - permutations

### Space Complexity
- **O(1):** Constant - few variables
- **O(n):** Linear - array/list of size n
- **O(n²):** Quadratic - 2D grid
- **O(log n):** Logarithmic - recursion depth

### Union-Find Operations
- **Find:** O(α(n)) ≈ O(1) amortized
- **Union:** O(α(n)) ≈ O(1) amortized
- α(n) = inverse Ackermann function (very slow growing)

---

## Data Structure Selection Guide

### When to use:

**Array/List**
- Sequential access
- Index-based lookup
- Fixed or dynamic size

**Hash Map/Set**
- Fast lookup (O(1))
- Unique elements
- Key-value pairs

**Queue (BFS)**
- Level-order traversal
- Shortest path (unweighted)
- FIFO processing

**Stack (DFS)**
- Backtracking
- Recursion simulation
- LIFO processing

**Heap/Priority Queue**
- Always need min/max
- Dijkstra's algorithm
- K-th largest/smallest

**Union-Find**
- Dynamic connectivity
- Grouping elements
- Cycle detection

**Grid/2D Array**
- Spatial problems
- Matrix operations
- Discretization

---

## Python Tips for Interviews

### Useful Built-ins

```python
# Collections
from collections import deque, defaultdict, Counter

# Heap
import heapq

# Math
import math

# Sorting with custom key
sorted(items, key=lambda x: x[0])

# List comprehension
[x*2 for x in range(10) if x % 2 == 0]

# Dictionary comprehension
{k: v*2 for k, v in dict.items()}

# Set operations
set1 & set2  # intersection
set1 | set2  # union
set1 - set2  # difference

# Infinity
float('inf'), float('-inf')

# Enumerate with index
for i, val in enumerate(arr):
    ...

# Zip multiple lists
for a, b, c in zip(list1, list2, list3):
    ...
```

### Common Patterns

```python
# Two pointers
left, right = 0, len(arr) - 1
while left < right:
    ...

# Sliding window
window_sum = sum(arr[:k])
for i in range(k, len(arr)):
    window_sum += arr[i] - arr[i-k]

# Binary search
left, right = 0, len(arr)
while left < right:
    mid = (left + right) // 2
    if check(mid):
        right = mid
    else:
        left = mid + 1

# DFS (recursive)
def dfs(node):
    if base_case:
        return
    for neighbor in node.neighbors:
        dfs(neighbor)

# BFS (iterative)
queue = deque([start])
visited = {start}
while queue:
    node = queue.popleft()
    for neighbor in node.neighbors:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
```

---

## Summary

**Problems Solved:** 5  
**Companies:** Google (3), Waymo (1), ABC (1)  
**Key Topics:** Union-Find, BFS, DP, Geometry, Discretization  
**Total Lines of Code:** ~2000+  
**All Tests:** ✓ PASSING

**Most Important Lesson:** Always clarify the problem and discuss your approach before coding. The Waymo problem showed that understanding the correct approach (don't merge circles!) is more important than coding speed.
