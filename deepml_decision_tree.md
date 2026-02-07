
Decision Tree Learning
Hard
Machine Learning

Write a Python function that implements the decision tree learning algorithm for classification. The function should use recursive binary splitting based on entropy and information gain to build a decision tree. It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as input, and return a nested dictionary representing the decision tree.

Tie-Breaking Rules:

If multiple attributes have equal information gain, choose the one that appears first in the attributes list.
If a leaf node has equal counts of different classes, return the class that comes first alphabetically.
Process attribute values in sorted order to ensure consistent tree structure.
Example:
Input:
examples = [
    {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},
    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
],
attributes = ['Outlook', 'Wind'],
target_attr = 'PlayTennis'
Output:
{'Outlook': {'Overcast': 'Yes', 'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}}, 'Sunny': 'No'}}
Reasoning:
The algorithm first calculates information gain for each attribute. 'Outlook' has the highest gain, so it becomes the root. For 'Overcast', all outcomes are 'Yes' (pure leaf). For 'Sunny', all outcomes are 'No' (pure leaf - no need to split further). For 'Rain', the data has mixed labels, so it recurses and splits on 'Wind'.

This is a classic ID3 implementation challenge. Your initial code had a few hiccups, specifically in the entropy calculation (using the label value instead of the probability) and the lack of handling for the recursive base cases.

I've fixed the entropy logic, implemented the information gain calculation, and added the recursive structure for `learn_decision_tree`.

### Implementation of ID3 Decision Tree

```python
import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if not labels:
        return 0
    cnts = Counter(labels)
    entropy = 0
    total = len(labels)
    for count in cnts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    total_entropy = calculate_entropy([ex[target_attr] for ex in examples])
    
    # Group examples by attribute value
    attr_values = {}
    for ex in examples:
        val = ex[attr]
        if val not in attr_values:
            attr_values[val] = []
        attr_values[val].append(ex[target_attr])
    
    # Calculate weighted entropy of subsets
    subset_entropy = 0
    total_count = len(examples)
    for val in attr_values:
        prob = len(attr_values[val]) / total_count
        subset_entropy += prob * calculate_entropy(attr_values[val])
        
    return total_entropy - subset_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    labels = [ex[target_attr] for ex in examples]
    if not labels:
        return ""
    
    cnts = Counter(labels)
    # Sort by frequency (desc) and then by name (asc) for tie-breaking
    # counts.items() -> [('No', 3), ('Yes', 3)] 
    # sorted -> [('No', 3), ('Yes', 3)] because 'N' < 'Y'
    sorted_classes = sorted(cnts.items(), key=lambda x: (-x[1], x[0]))
    return sorted_classes[0][0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    labels = [ex[target_attr] for ex in examples]

    # Base Case 1: All examples have the same label
    if len(set(labels)) == 1:
        return labels[0]

    # Base Case 2: No more attributes to split on
    if not attributes:
        return majority_class(examples, target_attr)

    # 1. Choose the best attribute based on Information Gain
    # Tie-break: if gains are equal, use the one first in the list (max() does this naturally)
    best_attr = max(attributes, key=lambda attr: calculate_information_gain(examples, attr, target_attr))

    # 2. Build the tree structure
    tree = {best_attr: {}}
    
    # Get all unique values for the chosen attribute (sorted for consistency)
    attr_values = sorted(list(set(ex[best_attr] for ex in examples)))
    
    remaining_attrs = [a for a in attributes if a != best_attr]

    for val in attr_values:
        # Create subset for this attribute value
        subset = [ex for ex in examples if ex[best_attr] == val]
        
        if not subset:
            # If subset is empty, add a leaf with the majority class of the parent
            tree[best_attr][val] = majority_class(examples, target_attr)
        else:
            # Recursively build the sub-tree
            tree[best_attr][val] = learn_decision_tree(subset, remaining_attrs, target_attr)

    return tree

```

---

### Key Logic & Tie-Breaking

* **Entropy:** Used `math.log2` which is the standard for ID3. Note that in your initial code, you were using `np.log(l/len(labels))`, where `l` was the label itself rather than its frequency.
* **Tie-Breaking (Attributes):** Python's `max()` function returns the **first** occurrence of the maximum value. Since we iterate through the `attributes` list in order, this naturally satisfies the requirement to pick the first attribute in the list if gains are tied.
* **Tie-Breaking (Labels):** In `majority_class`, I used a double-sort: `(-x[1], x[0])`. This sorts by count descending (the negative sign) and then alphabetically ascending.
* **Sorted Values:** The `attr_values` are sorted before iteration to ensure the dictionary structure remains deterministic.

Would you like me to write a test script with the 'PlayTennis' example to verify the output structure matches exactly?
