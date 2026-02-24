# Waymo Interview Question: Path Finding Through Circular Obstacles

## Interview Context

**Company:** Waymo  
**Position:** Applied ML / Simulation Team  
**Difficulty:** Medium-Hard  
**Topics:** Computational Geometry, Path Finding, Union-Find

---

## Problem Statement

### Scenario

You have a rectangular region defined by:
- **X-range:** `[x1, x2]` (horizontal boundaries)
- **Y-range:** `[y1, y2]` (vertical boundaries)

Within this region, there are multiple **circular obstacles** (may or may not overlap).

### Question

Can a vehicle travel from the **bottom** of the region (`y = y1`) to the **top** (`y = y2`)?

### Constraints

1. Vehicle starts at any point on the bottom edge: `y = y1`, `x ∈ [x1, x2]`
2. Vehicle must reach any point on the top edge: `y = y2`, `x ∈ [x1, x2]`
3. Vehicle can move in **curves** (not restricted to straight lines)
4. Vehicle **cannot pass through** circular obstacles
5. Vehicle must stay within the x-range `[x1, x2]`

### Input Format

```python
x_range: Tuple[float, float]  # (x1, x2)
y_range: Tuple[float, float]  # (y1, y2)
circles: List[Tuple[float, float, float]]  # [(cx1, cy1, r1), (cx2, cy2, r2), ...]
```

### Output

```python
Boolean: True if path exists, False otherwise
```

---

## Examples

### Example 1: Single Small Circle

```
Input:
  x_range = (0, 10)
  y_range = (0, 10)
  circles = [(5, 5, 2)]

Visualization:
  10 ┤                    ← Top (goal)
     │
   5 ┤      ●●●           ← Circle (radius 2)
     │     ●   ●
     │      ●●●
   0 ┤                    ← Bottom (start)
     └─────────────────
     0     5     10

Output: True
Explanation: Vehicle can go around the circle on either left or right side
```

### Example 2: Large Blocking Circle

```
Input:
  x_range = (0, 10)
  y_range = (0, 10)
  circles = [(5, 5, 6)]

Visualization:
  10 ┤   ●●●●●●●●●●●     ← Top (goal)
     │  ●           ●
   5 ┤ ●      ●      ●    ← Circle (radius 6)
     │  ●           ●
   0 ┤   ●●●●●●●●●●●     ← Bottom (start)
     └─────────────────
     0     5     10

Output: False
Explanation: Circle blocks the entire width, no path possible
```

### Example 3: Two Circles with Gap

```
Input:
  x_range = (0, 10)
  y_range = (0, 10)
  circles = [(3, 5, 2), (7, 5, 2)]

Visualization:
  10 ┤                    ← Top (goal)
     │
   5 ┤  ●●●   ●●●        ← Two circles
     │ ●   ● ●   ●
     │  ●●●   ●●●
   0 ┤                    ← Bottom (start)
     └─────────────────
     0  3  5  7  10

Output: True
Explanation: Vehicle can pass through the gap between circles
```

### Example 4: Overlapping Circles Forming Barrier

```
Input:
  x_range = (0, 10)
  y_range = (0, 10)
  circles = [(3, 5, 3), (7, 5, 3)]

Visualization:
  10 ┤                    ← Top (goal)
     │
   5 ┤ ●●●●●●●●●●        ← Overlapping circles
     │●           ●
     │ ●●●●●●●●●●
   0 ┤                    ← Bottom (start)
     └─────────────────
     0  3  5  7  10

Output: False
Explanation: Overlapping circles form a continuous barrier
```

---

## Solution Approach

### High-Level Strategy

The interviewer confirmed this is the **right track**:

1. **Merge overlapping circles** into larger circles
2. **Sort circles** by x-coordinate
3. **Check if circles block** the entire x-range at any y-level

### Detailed Algorithm

#### Step 1: Merge Overlapping Circles

**Why?** Overlapping circles act as a single larger obstacle.

**Algorithm:**
1. Use **Union-Find** to group overlapping circles
2. For each group, merge all circles into one large circle

**How to check if two circles overlap:**
```python
def overlaps(circle1, circle2):
    distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    return distance < (r1 + r2)
```

**How to merge two circles:**
```python
def merge(circle1, circle2):
    # Find farthest points on both circles
    # Create new circle that encompasses both
    # New center = midpoint of farthest points
    # New radius = half distance between farthest points
```

#### Step 2: Check for Blocking

**Key Insight:** A path is blocked if circles form a continuous barrier spanning the entire x-range.

**Algorithm:**
1. Sample multiple y-levels between y1 and y2
2. At each y-level, find x-ranges covered by circles
3. Check if covered ranges span the entire [x1, x2]
4. If any level is fully covered, no path exists

**Sweep Line Algorithm:**
```python
for y in sample_y_levels:
    covered_ranges = []
    for circle in circles:
        if circle intersects y:
            x_range = circle.x_range_at_y(y)
            covered_ranges.append(x_range)
    
    if is_fully_covered(covered_ranges, x1, x2):
        return False  # Blocked!

return True  # Path exists
```

---

## Implementation Details

### Circle Class

```python
class Circle:
    def __init__(self, x, y, r):
        self.x = x  # Center x
        self.y = y  # Center y
        self.r = r  # Radius
    
    def overlaps(self, other):
        """Check if circles overlap."""
        distance = sqrt((self.x - other.x)^2 + (self.y - other.y)^2)
        return distance < (self.r + other.r)
    
    def x_range_at_y(self, y):
        """Get x-range covered at given y-level."""
        if abs(y - self.y) > self.r:
            return (inf, -inf)  # No intersection
        
        dy = abs(y - self.y)
        dx = sqrt(self.r^2 - dy^2)
        return (self.x - dx, self.x + dx)
```

### Union-Find for Merging

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        # Union by rank...
```

### Main Algorithm

```python
def can_reach_top(x_range, y_range, circles):
    # Step 1: Convert to Circle objects
    circle_objects = [Circle(cx, cy, r) for cx, cy, r in circles]
    
    # Step 2: Merge overlapping circles
    merged = merge_overlapping_circles(circle_objects)
    
    # Step 3: Check if any configuration blocks passage
    return not forms_continuous_barrier(merged, x_range, y_range)
```

---

## Complexity Analysis

### Time Complexity

**Overall: O(n² + n·m)**

Where:
- `n` = number of circles
- `m` = number of sample y-levels (typically 100)

**Breakdown:**
1. **Merging circles:** O(n²)
   - Check all pairs for overlap: O(n²)
   - Union-Find operations: O(n·α(n)) ≈ O(n)
   - Merge groups: O(n)

2. **Checking barrier:** O(n·m)
   - For each y-level: O(m)
   - Check each circle: O(n)
   - Merge ranges: O(n log n)
   - Total: O(m·n log n)

**Practical:** For typical inputs (n < 100, m = 100), this is very fast.

### Space Complexity

**O(n)**
- Store circles: O(n)
- Union-Find structure: O(n)
- Merged circles: O(n)

---

## Edge Cases

### Case 1: No Circles
```python
circles = []
Output: True
```
No obstacles, path always exists.

### Case 2: Circle Outside Region
```python
x_range = (0, 10)
y_range = (0, 10)
circles = [(20, 20, 5)]  # Far outside
Output: True
```
Irrelevant circles should be filtered out.

### Case 3: Circle Touching Boundaries
```python
x_range = (0, 10)
y_range = (0, 10)
circles = [(5, 5, 5)]  # Touches all edges
Output: False
```
Circle blocks passage.

### Case 4: Many Small Circles
```python
circles = [(i, 5, 0.5) for i in range(0, 11)]
Output: False
```
Many small circles can form a barrier.

### Case 5: Vertical Stack
```python
circles = [(5, 2, 3), (5, 5, 3), (5, 8, 3)]
Output: False
```
Vertically stacked circles block passage.

---

## Optimization Strategies

### 1. Early Termination

```python
# Check if any single circle blocks entire passage
for circle in circles:
    if circle.blocks_x_range(x1, x2):
        if circle.y - circle.r <= y1 and circle.y + circle.r >= y2:
            return False  # Early exit
```

### 2. Spatial Indexing

For many circles, use spatial data structures:
- **Quadtree** for fast overlap detection
- **R-tree** for range queries

### 3. Adaptive Sampling

Instead of fixed y-samples, sample more densely near circles:
```python
y_samples = []
for circle in circles:
    y_samples.extend([circle.y - circle.r, circle.y, circle.y + circle.r])
y_samples = sorted(set(y_samples))
```

### 4. Parallel Processing

Check multiple y-levels in parallel:
```python
from multiprocessing import Pool

with Pool() as pool:
    results = pool.map(check_y_level, y_samples)
    return not any(results)
```

---

## Common Mistakes

### Mistake 1: Not Merging Overlapping Circles

❌ **Wrong:** Check each circle independently
```python
for circle in circles:
    if circle.blocks_x_range(x1, x2):
        return False
```

This misses cases where multiple overlapping circles form a barrier.

✅ **Correct:** Merge first, then check

### Mistake 2: Only Checking Straight Line

❌ **Wrong:** Check if straight line from bottom to top is blocked
```python
# Only check x = (x1 + x2) / 2
```

Vehicle can move in curves, so must check all possible paths.

✅ **Correct:** Use sweep line to check all y-levels

### Mistake 3: Insufficient Sampling

❌ **Wrong:** Only check a few y-levels
```python
for y in [y1, (y1+y2)/2, y2]:
    # Check...
```

May miss barriers between sample points.

✅ **Correct:** Use sufficient samples (100+) or adaptive sampling

---

## Interview Tips

### What the Interviewer Looks For

1. **Problem Understanding**
   - Clarify constraints (can vehicle move in curves?)
   - Ask about edge cases (overlapping circles?)

2. **Approach**
   - Identify key insight (merging overlapping circles)
   - Explain high-level strategy before coding

3. **Implementation**
   - Clean, modular code
   - Handle edge cases
   - Test with examples

4. **Optimization**
   - Discuss time/space complexity
   - Suggest improvements

### Time Management

- **5 min:** Understand problem, ask clarifying questions
- **10 min:** Discuss approach, get feedback
- **25 min:** Implement solution
- **10 min:** Test and optimize

### Red Flags to Avoid

- ❌ Jumping into code without explaining approach
- ❌ Not handling overlapping circles
- ❌ Inefficient O(n³) or worse algorithm
- ❌ Not testing with examples

---

## Summary

**Key Concepts:**
1. **Merge overlapping circles** using Union-Find
2. **Sweep line algorithm** to check for barriers
3. **Computational geometry** for circle intersections

**Algorithm:**
1. Merge overlapping circles into larger circles
2. Sample y-levels from bottom to top
3. At each level, check if circles cover entire x-range
4. If any level is fully covered, no path exists

**Complexity:**
- Time: O(n² + n·m) where m is sample count
- Space: O(n)

**Why This Works:**
- Merging simplifies the problem
- Sweep line catches all possible barriers
- Handles curved paths automatically

This is a great problem that combines graph theory, computational geometry, and practical autonomous vehicle path planning!
