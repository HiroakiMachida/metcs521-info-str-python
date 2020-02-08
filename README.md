#################################
#   1. Overview
#################################
Handwriting recognition.

Take handwriting number images and predict what the number is.

The structure of fully-connected neural network is as follows:
Input layer: 28 x 28 pixels(array of 784 elements) with 256 gray scale.
Layer1: 50 fully-connected neurons.
Layer2: Activation function of Rectified Liner Unit.
Layer3: 10 fully-connected neurons.
Output layer: Softmax-with-Loss of an one-hot array from 0 to 9.


#################################
#   2. Getting started
#################################
Use python 3.
Install NumPy and Matplotlib.
(See https://scipy.org/install.html)
Run train.py.
(Just on Spider or run "python train.py" on terminal.)


#################################
#   3. Requirements
#################################
container type (list, tuple, set, or dictionary)
  => list: train.py#31 test_acc_list
  => tuple: train.py#46 iteration
  => set: train.py#75 iteration
  => dictionary: network.py#26 params
  => string: train.py#71 dot

iteration type (for, while)
  => train.py#47

conditional (if)
  => train.py#53

try blocks
  => train.py#2

user-defined functions
  => train.py#68

input and/or output file (submit input data)
  => train.py#23 Automatically downloaded

user-defined class. The class must be imported by your main program and have
the following required structures.
− at least 1 private and 2 public self attributes
− at least 1 private and 1 public method that take arguments, return values and
are used by your program
− an init() method that takes at least 1 argument
− a repr() method 
  => network.py

Invalid case
  => Delete *.gz files and disconnect the internet, and run.


#################################
#   4. References
#################################
https://github.com/oreilly-japan/deep-learning-from-scratch


#################################
#   5. Author
#################################
Machida Hiroaki
