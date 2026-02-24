"""
Max Coins Collected by Tokens Moving Exactly 3 Steps Right on a 1D Board

Problem:
- 1D board represented by string s of length n
- Contains tokens (T) and coins (C) and empty cells (.)
- Each token can move only to the right
- Each move must be exactly 3 steps (from index i to i+3)
- If a token lands on a coin, it collects it (coin is removed)
- A token can perform any number of moves as long as i+3 < n
- Find the maximum number of coins that can be collected

Example:
Input: "T..C..C.TC"
Output: 2

Explanation:
- Token at index 0 can move: 0 -> 3 (collect coin C) -> 6 (collect coin C) = 2 coins
- Token at index 8 can move: 8 -> 11 (out of bounds, can't move)
- Token at index 9 can't move (9+3 = 12 >= 10)
- Maximum coins = 2
"""

def max_coins_collected(s):
    """
    Calculate maximum coins that can be collected by tokens moving exactly 3 steps right.
    
    Approach:
    1. Each token at position i can only reach positions: i, i+3, i+6, i+9, ...
    2. These positions form "chains" based on (index % 3)
    3. Positions with same (index % 3) can be reached by tokens in that chain
    4. For each chain (0, 1, 2), find which token can collect the most coins
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    n = len(s)
    
    # Group positions by their modulo 3 value (chains)
    # Chain 0: positions 0, 3, 6, 9, ...
    # Chain 1: positions 1, 4, 7, 10, ...
    # Chain 2: positions 2, 5, 8, 11, ...
    chains = [[], [], []]
    
    for i in range(n):
        chains[i % 3].append((i, s[i]))
    
    total_coins = 0
    
    # Process each chain independently
    for chain in chains:
        if not chain:
            continue
        
        # Find all tokens and coins in this chain
        tokens = []
        coins = []
        
        for pos, char in chain:
            if char == 'T':
                tokens.append(pos)
            elif char == 'C':
                coins.append(pos)
        
        if not tokens or not coins:
            continue
        
        # For each token, calculate how many coins it can collect
        # A token at position i can collect coins at positions i+3, i+6, i+9, ...
        # which are all positions > i in the same chain
        
        # Greedy approach: assign each coin to the leftmost token that can reach it
        # This maximizes the total coins collected
        
        # Sort tokens and coins by position (already sorted since we iterate in order)
        coins_collected = 0
        coin_idx = 0
        
        for token_pos in tokens:
            # This token can collect all coins at positions > token_pos in this chain
            while coin_idx < len(coins) and coins[coin_idx] <= token_pos:
                coin_idx += 1
            
            # Count coins this token can collect
            temp_coins = 0
            temp_idx = coin_idx
            while temp_idx < len(coins):
                temp_coins += 1
                temp_idx += 1
            
            coins_collected = max(coins_collected, temp_coins)
        
        total_coins += coins_collected
    
    return total_coins


def max_coins_collected_optimized(s):
    """
    Optimized solution using dynamic programming approach.
    
    Key insight:
    - Tokens at position i can reach positions i+3k (k >= 0)
    - For each chain (positions with same i%3), we need to find optimal assignment
    - Use greedy: leftmost token collects all coins to its right in the chain
    
    Time Complexity: O(n)
    Space Complexity: O(1) - only using counters
    """
    n = len(s)
    total_coins = 0
    
    # Process each of the 3 chains (mod 3 = 0, 1, 2)
    for start in range(3):
        # For this chain, find the leftmost token
        leftmost_token = -1
        coins_after_token = 0
        
        # Traverse this chain
        pos = start
        while pos < n:
            if s[pos] == 'T':
                if leftmost_token == -1:
                    # First token in this chain
                    leftmost_token = pos
                    coins_after_token = 0
                # If we already have a token, this new token can't collect
                # coins that the previous token already collected
            elif s[pos] == 'C':
                if leftmost_token != -1:
                    # There's a token that can reach this coin
                    coins_after_token += 1
            
            pos += 3
        
        total_coins += coins_after_token
    
    return total_coins


def solve(s):
    """
    Main solution function.
    
    Strategy:
    1. Divide the board into 3 independent chains based on position % 3
    2. For each chain, the leftmost token can collect all coins to its right
    3. Sum up coins from all chains
    """
    n = len(s)
    max_coins = 0
    
    # Process each chain (0, 1, 2)
    for chain_id in range(3):
        # Find leftmost token in this chain
        first_token_pos = -1
        
        for i in range(chain_id, n, 3):
            if s[i] == 'T':
                first_token_pos = i
                break
        
        if first_token_pos == -1:
            # No token in this chain
            continue
        
        # Count coins after the first token in this chain
        coins_in_chain = 0
        for i in range(first_token_pos + 3, n, 3):
            if s[i] == 'C':
                coins_in_chain += 1
        
        max_coins += coins_in_chain
    
    return max_coins


# Test cases
def test():
    print("=" * 60)
    print("Testing Max Coins Collection Problem")
    print("=" * 60)
    
    test_cases = [
        ("T..C..C.TC", 2),
        ("T", 0),
        ("C", 0),
        ("TC", 0),
        ("T..C", 1),
        ("T.....C", 0),  # Token at 0 can reach 3, 6 but not 6 (coin at 6)
        ("T..C..C", 2),
        ("...T..C..C", 2),
        ("T..T..C", 1),  # Two tokens, but only first one matters
        ("CT..C..C", 2),  # Coin before token doesn't count
        ("T..C..T..C", 2),  # Each token in different chains
    ]
    
    for i, (input_str, expected) in enumerate(test_cases, 1):
        result = solve(input_str)
        status = "✓" if result == expected else "✗"
        print(f"\nTest {i}: {status}")
        print(f"  Input:    '{input_str}'")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        
        if result != expected:
            print(f"  ERROR: Mismatch!")
            # Debug: show chain analysis
            print(f"  Debug:")
            for chain_id in range(3):
                positions = [i for i in range(chain_id, len(input_str), 3)]
                chars = [input_str[i] for i in positions]
                print(f"    Chain {chain_id}: positions {positions} -> {chars}")


def main():
    """
    Main function to read input and solve the problem.
    """
    print("\n" + "=" * 60)
    print("Max Coins Collection - Solution")
    print("=" * 60)
    
    # Read input
    s = input("Enter the board string (T=token, C=coin, .=empty): ").strip()
    
    # Solve
    result = solve(s)
    
    # Output
    print(f"\nMaximum coins collectible: {result}")
    
    # Show detailed analysis
    print("\n" + "-" * 60)
    print("Detailed Analysis:")
    print("-" * 60)
    
    n = len(s)
    for chain_id in range(3):
        print(f"\nChain {chain_id} (positions with index % 3 = {chain_id}):")
        
        positions = []
        chars = []
        for i in range(chain_id, n, 3):
            positions.append(i)
            chars.append(s[i])
        
        print(f"  Positions: {positions}")
        print(f"  Characters: {chars}")
        
        # Find first token
        first_token = -1
        for i, char in enumerate(chars):
            if char == 'T':
                first_token = positions[i]
                break
        
        if first_token == -1:
            print(f"  No token in this chain")
        else:
            coins = sum(1 for i in range(positions.index(first_token) + 1, len(chars)) if chars[i] == 'C')
            print(f"  First token at position {first_token}")
            print(f"  Coins collectible: {coins}")


if __name__ == "__main__":
    # Run tests first
    test()
    
    # Then run main program
    print("\n" + "=" * 60)
    main()
