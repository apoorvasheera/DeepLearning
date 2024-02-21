import os
from re import X
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np


def value_and_derivative_of_g(x):
    ##############################################################################
    # TODO:  Return the value and derivative of g(x) = (3x^2-4)^5                  #
    ##############################################################################
    value = 0
    derivative = 0
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    value = (3*x**2-4)**5 
    derivative = 30*x*((3*x**2 - 4)**4)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return value, derivative