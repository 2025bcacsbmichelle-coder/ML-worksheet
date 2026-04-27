# Cost Function and Gradient Descent
import numpy as np
import math, copy
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])           #(size in 1000s square feet)
y_train = np.array([30.0, 50.0, 68.0, 75.0, 100.0])           #(price in INR lakhs)

def compute_cost(x, y, b, w):
    """
    Computes the cost function for linear regression.

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      b, w (scalar)    : theta_0 and theta_1, model parameters

    Returns
        total_cost (float): The cost of using theta_0, theta_1 as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    cost_sum = 0

    for i in range(m):
        # Calculate h(x) and then calculate the cost from y
        # YOUR CODE HERE (find h_x then use it to find cost):
        h_x = w * x[i] + b

        cost = (h_x - y[i]) ** 2

        cost_sum += cost


    total_cost = (1 / (2*m)) * cost_sum

    return total_cost


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        h_x = w * x[i] + b
        dj_dw_i = (h_x - y[i]) * x[i]
        dj_db_i = h_x - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b]
      """

    w = copy.deepcopy(w_in) # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha *dj_db



        # Save cost J and parameters at each iteration
        if i<100000:
            cost = cost_function(x, y, w, b)

            J_history.append(cost)
            p_history.append([w, b])

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history #return w and J,w history for graphing


# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()


# Use the hypothesis equation and parameters obtained. Get predictions for 1000 sqft, 2000sqft and 1750 sqft.
# Note: your model is trained on per thousand data, so 1 = 1000

# YOUR CODE HERE:
# Use the hypothesis equation and parameters obtained

pred_1000 = w_final * 1.0 + b_final
pred_2000 = w_final * 2.0 + b_final
pred_1750 = w_final * 1.75 + b_final


print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand INR")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand INR")
print(f"1750 sqft house prediction {w_final*1.75 + b_final:0.1f} Thousand INR")



# Expected:
# For 1000 = 31.6 Thousand INR
# For 2000 = 48.1 Thousand INR
# For 1750 = 44.0 Thousand INR