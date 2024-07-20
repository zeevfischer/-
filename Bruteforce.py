import itertools


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def generate_full_tree(level, features_values, current_level=1, leaf_values=None, index=0):
    """Generate a full binary tree of a given level with specified leaf values and node labels from list_of_features."""
    if leaf_values and current_level == level:
        # Use the first value of leaf_values and remove it from the list
        return TreeNode(leaf_values.pop(0))

    # Determine node label based on list_of_features
    node_label = features_values[index] if index < len(features_values) else f'c{current_level}'
    node = TreeNode(node_label)

    if current_level < level:
        node.left = generate_full_tree(level, features_values, current_level + 1, leaf_values, index + 1)
        node.right = generate_full_tree(level, features_values, current_level + 1, leaf_values, index + 2)

    return node


def run_and_classify(tree, Features_list, label):
    current_node = tree
    while current_node.right is not None and current_node.left is not None:
        if Features_list[current_node.val - 1] == 0:
            if current_node.left is not None:
                current_node = current_node.left
        elif Features_list[current_node.val - 1] == 1:
            if current_node.right is not None:
                current_node = current_node.right

    # We should be at a leaf node now
    if current_node.left is None and current_node.right is None:
        return current_node.val == label
    else:
        return False  # The node is not a leaf node


def generate_leafs_options(n):
    """Generate a list of binary numbers from 0 to n-1."""
    binary_list = []
    for i in range(n):
        # Format the number as a binary string with leading zeros
        binary_str = format(i, f'0{n.bit_length() - 1}b')
        binary_list.append(binary_str)
    return binary_list


def print_tree(root):
    """Helper function to print a binary tree in the specified format."""
    lines, *_ = display_aux(root)
    for line in lines:
        print(line)


def display_aux(node):
    """Returns list of strings, width, height, and horizontal coordinate of the root."""
    if node.left is None and node.right is None:
        line = f'{node.val}'
        width = len(line)
        height = 1
        middle = width // 2
        return [line], width, height, middle

    left, n, p, x = display_aux(node.left) if node.left else ([], 0, 0, 0)
    right, m, q, y = display_aux(node.right) if node.right else ([], 0, 0, 0)
    s = f'Feature:{node.val}'
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


def tree_permutations(number_of_features, choose):
    numbers = list(range(1, number_of_features + 1))
    permutations = list(itertools.permutations(numbers, choose))

    return permutations


def main(permutations, options, Features, Labels, K):
    best_tree = None
    best_error = None
    for permutation in permutations:
        for leaf_option in options:
            # creat the tree
            list_leaf_option = list(leaf_option)
            list_leaf_option = [int(char) for char in list_leaf_option]
            tree = generate_full_tree(K, permutation, leaf_values=list_leaf_option)
            Error = 0
            # now for every tree run over all the vecturs
            for features, lable in zip(Features, Labels):
                check = run_and_classify(tree, features, lable)
                if check == False:
                    Error = Error + 1

            # now check the error of the tree
            if best_error == None or best_error > Error:
                best_error = Error
                best_tree = tree

    percent = best_error / len(Features) * 100
    # print("Best error: " + str(best_error) + " out of: " + str(len(Features)) + " which is " + str(percent) + " percent")
    print(f"Error: {percent}" + "%")
    print("best tree:")
    print_tree(best_tree)
