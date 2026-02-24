# Max Coins Collection Problem - Detailed Explanation

## Problem Statement

Given a 1D board represented by a string `s` of length `n`:
- Contains tokens (`T`), coins (`C`), and empty cells (`.`)
- Each token can move only to the right
- Each move must be **exactly 3 steps** (from index `i` to `i+3`)
- If a token lands on a coin, it collects it (coin is removed)
- A token can perform any number of moves as long as `i+3 < n`
- Find the **maximum number of coins** that can be collected

## Example

```
Input: "T..C..C.TC"
Output: 2
```

### Visualization

```
Index:  0  1  2  3  4  5  6  7  8  9
Board:  T  .  .  C  .  .  C  .  T  C
        ↓        ↓        ↓
        0  →  →  3  →  →  6  →  →  9 (out of bounds)
```

- Token at index 0 can move: `0 → 3 (collect C) → 6 (collect C)` = **2 coins**
- Token at index 8 can move: `8 → 11` (out of bounds, can't move)
- Token at index 9 can't move: `9 + 3 = 12 ≥ 10` (out of bounds)

**Maximum coins = 2**

---

## Key Insight: Chain Decomposition

### Observation 1: Independent Chains

Tokens can only reach positions that are **multiples of 3 steps away**:
- Token at position `i` can reach: `i, i+3, i+6, i+9, i+12, ...`

This means positions can be grouped into **3 independent chains** based on `index % 3`:

```
Chain 0: positions 0, 3, 6, 9, 12, ... (index % 3 = 0)
Chain 1: positions 1, 4, 7, 10, 13, ... (index % 3 = 1)
Chain 2: positions 2, 5, 8, 11, 14, ... (index % 3 = 2)
```

### Observation 2: Tokens in Same Chain

Tokens in the same chain can reach the same set of positions. For example:
- Token at position 0 can reach: 0, 3, 6, 9, ...
- Token at position 3 can reach: 3, 6, 9, 12, ...

The **leftmost token** in a chain can reach all positions that any token in that chain can reach.

### Observation 3: Greedy Strategy

For each chain, the **leftmost token** should collect all coins to its right in that chain. This is optimal because:
1. Any token can only collect coins in its own chain
2. The leftmost token can reach all positions that later tokens can reach
3. There's no benefit to using a later token instead

---

## Algorithm

### Step 1: Divide into Chains

```python
chains = [[], [], []]  # 3 chains for mod 0, 1, 2

for i in range(n):
    chains[i % 3].append((i, s[i]))
```

### Step 2: Process Each Chain

For each chain:
1. Find the **leftmost token** (first `T` in the chain)
2. Count all **coins after** that token in the chain
3. Add to total

```python
for chain_id in range(3):
    # Find leftmost token
    first_token_pos = -1
    for i in range(chain_id, n, 3):
        if s[i] == 'T':
            first_token_pos = i
            break
    
    if first_token_pos == -1:
        continue  # No token in this chain
    
    # Count coins after first token
    coins = 0
    for i in range(first_token_pos + 3, n, 3):
        if s[i] == 'C':
            coins += 1
    
    total += coins
```

### Step 3: Return Total

Sum up coins from all 3 chains.

---

## Complexity Analysis

### Time Complexity: O(n)
- We iterate through the string once to process each chain
- Each position is visited exactly once

### Space Complexity: O(1)
- We only use a constant amount of extra space (counters)
- No additional data structures needed

---

## Example Walkthrough

### Example 1: `"T..C..C.TC"`

**Step 1: Identify Chains**

```
Index:  0  1  2  3  4  5  6  7  8  9
Board:  T  .  .  C  .  .  C  .  T  C
Chain:  0  1  2  0  1  2  0  1  2  0

Chain 0: [0:T, 3:C, 6:C, 9:C]
Chain 1: [1:., 4:., 7:., ]
Chain 2: [2:., 5:., 8:T]
```

**Step 2: Process Each Chain**

**Chain 0:**
- Leftmost token: position 0
- Coins after position 0: positions 3, 6, 9
- Coins collected: 3 (but wait, position 9 is also 'C')
- Actually: positions 3:C, 6:C, 9:C → **3 coins**

Wait, let me recheck the example...

Actually, looking at `"T..C..C.TC"`:
```
Index:  0  1  2  3  4  5  6  7  8  9
Board:  T  .  .  C  .  .  C  .  T  C
```

Chain 0: positions [0, 3, 6, 9] → chars [T, C, C, C]
- Leftmost token at 0
- Coins after: 3:C, 6:C, 9:C → **3 coins**

But expected output is 2... Let me reconsider.

Oh! Position 9 is 'C', not reachable from position 0 because:
- 0 → 3 → 6 → 9 (this works!)

So it should be 3 coins, not 2. Unless the example has a typo, or I'm misunderstanding.

Let me recount the string:
```
"T..C..C.TC"
 0123456789
```
- Position 0: T
- Position 3: C
- Position 6: C
- Position 8: T
- Position 9: C

Chain 0: [0:T, 3:C, 6:C, 9:C]
Chain 1: [1:., 4:., 7:.]
Chain 2: [2:., 5:., 8:T]

Token at 0 can collect: 3:C, 6:C, 9:C = 3 coins
Token at 8 (chain 2) has no coins after it

**Total = 3 coins**

**Note:** The expected output of 2 might be incorrect, or there's a different interpretation of the problem.

---

## Edge Cases

### Case 1: No Tokens
```
Input: "C..C..C"
Output: 0
```
No tokens to collect coins.

### Case 2: No Coins
```
Input: "T..T..T"
Output: 0
```
No coins to collect.

### Case 3: Token After Coins
```
Input: "C..C..T"
Output: 0
```
Token can't move backward to collect coins.

### Case 4: Multiple Tokens in Same Chain
```
Input: "T..T..C"
Output: 1
```
Both tokens at positions 0 and 3 are in chain 0.
Leftmost token (position 0) collects coin at position 6.

### Case 5: Tokens in Different Chains
```
Input: "T.T...C.C"
Output: 2
```
- Chain 0: [0:T, 3:., 6:C, 9:.]
- Chain 1: [1:., 4:., 7:., ]
- Chain 2: [2:T, 5:., 8:C]

Token at 0 collects 6:C = 1 coin
Token at 2 collects 8:C = 1 coin
Total = 2 coins

---

## Python Implementation

See `max_coins_solution.py` for the complete implementation with:
- Main solution function
- Test cases
- Detailed analysis output
- Interactive input mode

### Usage

```bash
python max_coins_solution.py
```

The program will:
1. Run automated tests
2. Prompt for input
3. Display the maximum coins collectible
4. Show detailed chain analysis

---

## Summary

**Key Points:**
1. Divide the board into 3 independent chains based on `index % 3`
2. For each chain, find the leftmost token
3. Count all coins after that token in the chain
4. Sum up coins from all chains

**Why This Works:**
- Tokens can only move in steps of 3, creating independent chains
- The leftmost token in each chain can reach all positions that any token in that chain can reach
- Greedy approach: use the leftmost token to maximize coin collection

**Complexity:**
- Time: O(n) - single pass through the string
- Space: O(1) - constant extra space
