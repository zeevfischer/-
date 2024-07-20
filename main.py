import Binaryentropy
import Bruteforce
import numpy as np


def load_data(file_path):
    data = np.loadtxt(file_path, dtype=int)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

Features, Labels = load_data('vectors.txt')
K = 3

# BruteForce data
print("Brute Force data")
number_of_options = pow(2, K + 1)
options = Bruteforce.generate_leafs_options(number_of_options)
permutations = Bruteforce.tree_permutations(8, 3)
Bruteforce.main(permutations, options, Features, Labels, K)

print("")

# BinaryEntropy data
print("Binary Entropy data")
Binaryentropy.main(Features,Labels)