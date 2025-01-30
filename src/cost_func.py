import numpy as np
 
def calculateCost(x, y, w, b):
    cost = 0
    for i in range(len(x)):
        c = w * x[i] + b
        cost += (y[i] - c) ** 2   # Squared error
        m = len(x) //number of training examples
    return cost / (2 * m))   # Divide by 2*m for MSE

 
'''
What is Gradient Descent 
It is an iterative process that finds the best weight and bias that minimize the loss.
The loss function of linear model always produce the convex shape 
Simple :if we take just 1 feature . It produce the bowl or U shape  
while x is weight y is bias .
In multiple the loss function remains convex in the higher-dimensional space defined by all the weights and bias, Mean just one Global minimum.
'''
 
