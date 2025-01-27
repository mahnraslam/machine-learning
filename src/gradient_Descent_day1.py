from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
from cost_func import calculate
"""
    Compute the gradients of the cost function with respect to the model parameters.

    This function calculates the gradients of the cost function with respect to the weight (`w`) and bias (`b`) 
    for a simple linear regression model. The gradients are computed using the Mean Squared Error 
    (MSE) cost function.

    Parameters:
    ----------
    x : list or array-like
        The input features of the training dataset. It should be a list or array where each element corresponds to 
        one training example.
    y : list or array-like
        The true output values of the training dataset. It should be a list or array where each element corresponds 
        to the true value for the corresponding input feature in `x`.
    w : float
        The weight parameter of the linear regression model.
    b : float
        The bias parameter of the linear regression model.

    Returns:
    -------
    dj_dw : float
        The gradient of the cost function with respect to the weight `w`. This value indicates how much the cost 
        function would change with a small change in `w`.
    dj_db : float
        The gradient of the cost function with respect to the bias `b`. This value indicates how much the cost 
        function would change with a small change in `b`.

    Notes:
    -----
    The gradients are computed as follows:
    - For each training example, compute the prediction `f_wb_i` using the formula `w * x[i] + b`.
    - Calculate the error `f_wb_i - y[i]` and use it to compute the partial derivatives with respect to `w` and `b`.
    - Average these partial derivatives over all training examples to get the final gradients.

    

"""
def compute_gradient(x, y, w, b) :
    
    m = len(x)
    dj_dw = 0
    dj_db = 0
    for i in range(m) : 
        #CALCULATE PREDICTION 
        f_wb_i = w*x[i] + b
        #Calculate partial derivative of this cost 
        dj_db_i = f_wb_i - y[i]
        dj_dw_i  = (f_wb_i - y[i]) * x[i]
        dj_dw  += dj_dw_i
        dj_db  += dj_db_i
    #Devide by number of examples 
    dj_db = 1/m * dj_db
    dj_dw = 1/m * dj_dw

    return dj_dw, dj_db
 
# Example data
df = pd.read_csv("student_scores.csv")
x =  df['Hours']
y = df['Scores'] 
w = 0
b = 0

# Initialize lists to store gradients
grad_w = []
grad_b = []
learning_rate = 0.001
cost = []
# Simulate multiple iterations
 
for i in range(20):
    dj_dw, dj_db = compute_gradient(x, y, w, b)
    
    grad_w.append(dj_dw)
    grad_b.append(dj_db)
    
    cost.append(calculate(x,y,dj_dw,dj_db))
    
     
    w -= learning_rate * dj_dw
    b -= learning_rate * dj_db

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))

# Plot the gradient with respect to w vs iteration
ax1.plot(grad_w, label='Gradient w.r.t  w')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Gradient')
ax1.set_title('Gradient w.r.t w vs Iteration')
ax1.legend()

# Plot the gradient with respect to b vs iteration
ax2.plot(grad_b, label='Gradient w.r.t b')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Gradient')
ax2.set_title('Gradient w.r.t b vs Iteration')
ax2.legend()
 
    

#Second way 
# fig = plt.figure(figsize=(12,7))
# ax = fig.add_subplot(111, projection = '3d') 


# Create a meshgrid for w and b
# W, B = np.meshgrid(grad_w, grad_b)
# # Plot the cost function
# ax.plot(grad_w, grad_b, cost, marker='o', linestyle='-', color='b')
# Initialize a 2D array to store cost values
# Cost = np.zeros(W.shape)

 
# Create the contour plot
# plt.figure(figsize=(10, 7))
# cp = plt.contourf(W, B, Cost, cmap='viridis', levels=50)
# plt.colorbar(cp)
    
# # Labels and title
# ax.set_xlabel('w')
# ax.set_ylabel('b')
# ax.set_zlabel('Cost')
# ax.set_title('Cost Function over Iterations')
# w_values = np.array(grad_w)  # These should be populated during your gradient descent loop
# b_values = np.array(grad_b)
# plt.plot(grad_w, grad_b, 'r-o')  # Path of the gradient descent
# Show the plots
plt.show()