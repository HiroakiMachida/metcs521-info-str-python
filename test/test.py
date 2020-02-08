#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 00:52:14 2019

@author: hiroakimachida
"""

import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.layers import Affine

"""
Initialize Affine

Neuron that has two inputs and one output.
"""
weights = np.array([[1], [2]])
biases = np.array([3])
affine = Affine(weights, biases)
x = np.array([5, 6])

"""
Test forward()

Test:
Input [5, 6] and get [20]
    
x1 * w1 + x2 * w2 + b = y
5 * 1 + 6 * 2 + 3 = 20
    
Expected:
20
"""
result = affine.forward(x)
expected = [20]
assert np.array_equal(result, expected), "forward() error."
print("forward() ok!")


"""
Test backward()

Test:
differentiation of w1 and w2,
that means if I change w1 or w2, how much the output will change?

d(x1 * w1 + x2 * w2 + b)/d(x1 * w1 + x2 * w2) = 1
d(x1 * w1 + x2 * w2)/d(x1 * w1) = 1
d(x1 * w1 + x2 * w2)/d(x2 * w2) = 1
d(x1 * w1)/d(w1) = 5
d(x2 * w2)/d(w2) = 6

Expected:
5
6
"""
dout = np.array([1])
affine.backward(dout)
result = affine.dW
expected = [[5], [6]]
assert np.array_equal(result, expected), "backward() error."
print("backward() ok!")