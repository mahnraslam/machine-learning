#SLOPE
#Using first principle
# slope = f(xo + delta) - f(x0) / delta
# Funrion is x**2+2x +2 
import numpy as np
#using python code
import pandas as pd
import matplotlib.pyplot as plt
def f(x):
    return x**2 + 2*x + 1

def  deriv(f,x,delta):
    return  (f(x+delta) - f(x)) / delta
 
def bias(x,y,w):
    return y-w*x

def plot(x,y) :
    # fig,ax  = plt.subpots(nrows = 1, ncols = 1)
    # ax.plot(x,y)
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Function')
    ax.scatter(x,y,f(x))
    plt.show()
def main():
    data = pd.read_csv('student_scores.csv')
    x = np.array(data['Hours'])
    y = np.array(data['Scores'])
    plot(x,y)
main()
