import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cost_func import calculate
df = pd.read_csv("student_scores.csv")

x = np.array(df['Hours'])
y = np.array(df['Scores'])
x_mean , y_mean = np.mean(x) , np.mean(y)
print(np.mean(x_mean))
print(np.mean(y_mean))

#Variance
var  = np.sum(np.square(x-x_mean)) / len(x)

size = len(x)
#cov(x,y)
cov = np.sum((x-x_mean)*(y-y_mean)) / size

#corelation = cov / std 
cor = np.sum((x - x_mean) * (y - y_mean)) / np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))

 
 #Linear regression 
# beat1 = cov / var 

beta1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)

beta0 = y_mean  - (beta1 * x_mean)
#sns.scatterplot(x= "Hours" , y = "Scores" ,data =df)
plt.scatter(x,y)
xPointsOfLine = np.linspace(0,9)
yPointsOfLine  = beta0 + beta1*xPointsOfLine

sns.lineplot(x = xPointsOfLine, y = yPointsOfLine)

print("Loss is",calculate(x,y,beta1 , beta0))
print("beta0" , beta0 , "beta1" ,beta1)
#Test x = 4.5 y =41
x_1 = 4.5
y_1 = beta0 + beta1 * x_1
print("predicted value" , y_1)
plt.scatter(x_1, y_1, marker = 'x', s = 150 ,color = 'red')  


# Example 2 : x = 3.3, y = 42 predicted 36
#example 3 ; 8.9,95
x_1 =  8.9 
y_1 = beta0 + beta1 * x_1
print("predicted value" , y_1)
plt.scatter(x_1, y_1, marker = 'x', s = 150, color = 'red')  

# plt.axvline()
# plt.axhline()
plt.grid()
plt.show()