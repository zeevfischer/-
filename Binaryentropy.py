import numpy as np
from collections import Counter


# Node class definition
class Node:
    def __init__(self, feature=None, split_value=None, left=None, right=None, label=None):
        self.feature = feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.label = label


# Function to calculate binary entropy
def binary_entropy(labels):
    total = len(labels)
    if total == 0:
        return 0
    count = Counter(labels)
    probs = [count[label] / total for label in count]
    return -sum(p * np.log2(p) for p in probs if p > 0)


# Function to find the best split based on binary entropy
def best_split(X, y):
    best_entropy = float('inf')
    best_feature = None
    best_split_value = None

    n_features = X.shape[1]
    for feature in range(n_features):
        left_indices = X[:, feature] == 0
        right_indices = X[:, feature] == 1
        left_labels = y[left_indices]
        right_labels = y[right_indices]

        left_entropy = binary_entropy(left_labels)
        right_entropy = binary_entropy(right_labels)
        split_entropy = (left_entropy * len(left_labels) + right_entropy * len(right_labels)) / len(y)

        if split_entropy < best_entropy:
            best_entropy = split_entropy
            best_feature = feature
            best_split_value = 0  # Binary split, so split value is 0

    return best_feature, best_split_value, best_entropy


# Function to build the tree
def build_tree(X, y, depth, max_depth):
    if depth == max_depth - 1 or len(set(y)) == 1:
        label = Counter(y).most_common(1)[0][0]
        return Node(label=label)

    feature, split_value, entropy = best_split(X, y)

    left_indices = X[:, feature] == split_value
    right_indices = X[:, feature] != split_value

    left_node = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
    right_node = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)

    return Node(feature=feature, split_value=split_value, left=left_node, right=right_node)


# Function to predict the label for a given data point
def predict(node, x):
    while node.label is None:
        if x[node.feature] == node.split_value:
            node = node.left
        else:
            node = node.right
    return node.label


# Function to calculate the error of the tree on the dataset
def calculate_error(tree, X, y):
    predictions = [predict(tree, x) for x in X]
    # count = 0
    # for prediction,label in zip(predictions,y):
    #     if prediction != label:
    #         count = count+1
    # return count
    return np.mean(predictions != y)


# Function to print the tree structure
def print_tree(node):
    """Helper function to print a binary tree in the specified format."""
    lines, *_ = display_aux(node)
    for line in lines:
        print(line)


def display_aux(node):
    """Returns list of strings, width, height, and horizontal coordinate of the root."""
    if node.left is None and node.right is None:
        line = f'{node.label}' if node.label is not None else 'None'
        width = len(line)
        height = 1
        middle = width // 2
        return [line], width, height, middle

    left, n, p, x = display_aux(node.left) if node.left else ([], 0, 0, 0)
    right, m, q, y = display_aux(node.right) if node.right else ([], 0, 0, 0)
    s = f'Feature {node.feature}'
    u = len(s)
    first_line = (x + 2) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y + 1) * ' '
    second_line = (x) * ' ' + '0/' + (n - x - 1 + u + y) * ' ' + '\\1' + (m - y - 1) * ' '
    if p < q:
        left += [n * ' '] * (q - p)
    elif q < p:
        right += [m * ' '] * (p - q)
    zipped_lines = zip(left, right)
    lines = [first_line, second_line] + [a + (u + 2) * ' ' + b for a, b in zipped_lines]
    return lines, n + m + u + 2, max(p, q) + 2, n + u // 2


def main(features, labels):
    # Build the tree
    tree = build_tree(features, labels, 0, max_depth=3)

    # Calculate the error
    error = calculate_error(tree, features, labels)
    print(f"Error: {error}" + "%")

    # Print the tree structure
    print("tree")
    print_tree(tree)
