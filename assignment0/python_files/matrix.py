import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np


def calculate_matrix_C():
    ##############################################################################
    # TODO:     [[1, 2, 4],       [[0, 1, 3]                                    #
    #       A = [2, 3, 1],    B = [2, 0, 4]                                     #
    #           [0, 1, 2]]        [4, 3, 1]]                                    #
    #                                                                           # 
    # Return C, the product of A and B                                          #
    # Hints:                                                                    #
    #    1. `np.array([[0,0],[0,0]])` creates a 2x2 numpy array of zeros.       #
    #    2. In numpy, `@` is the symbol for matrix multiplication               #
    ##############################################################################
    C = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    A = np.array([[1,2,4],[2,3,1],[0,1,2]])
    B = np.array([[0,1,3],[2,0,4],[4,3,1]])
    C = A @ B

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return C