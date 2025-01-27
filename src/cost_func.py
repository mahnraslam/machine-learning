import numpy as np
def calculate(x,y,w,b) :
    cost = 0
    for i in range(len(x)):
        c = w*x[i] + b
        cost += abs(y[i]- c)   # MAE
    return cost

x = np.arange(1,12)
 
#generate numbers b/w 0,20  np.random.randint(0, 20, 12)
y = [2,4,6,8,10,12,14,13,11,12,9]
w = 1.0199
b = 3
 
loss = calculate(x,y,w,b)
print(loss)
'''
What is Gradient Descent 
It is an iterative process that finds the best weight and bias that minimize the loss.
The loss function of linear model always produce the convex shape 
Simple :if we take just 1 feature . It produce the bowl or U shape  
while x is weight y is bias .
In multiple the loss function remains convex in the higher-dimensional space defined by all the weights and bias, Mean junt one Global minimum.
'''
 