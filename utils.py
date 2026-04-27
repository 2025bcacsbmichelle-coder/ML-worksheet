"""
Utility Functions
"""
import matplotlib.pyplot as plt


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'green', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='.', s=s, label=neg_label, facecolors='none', edgecolors='#0096ff', lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with your actual X, y)
X = np.array([[1,2], [2,3], [3,3], [5,6], [6,7]])
y = np.array([0, 0, 1, 1, 1])

# Create figure and axis
fig, ax = plt.subplots()

# Call your function
plot_data(X, y, ax)

# Show plot
plt.show()