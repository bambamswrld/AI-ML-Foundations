import numpy as np
import matplotlib.pyplot as plt

# the data (swap these for any x, y you want)
x = [1, 2, 3]
y = [5, 1, 3]
m = 1                      # frozen slope

# the loss function for THIS data, as a function of b
def loss(b):
    total = 0
    for xi, yi in zip(x, y):
        residual = yi - (m * xi + b)
        total += residual ** 2
    return total / len(x)

# sweep b across a range, scoring each candidate
b_values = np.linspace(-3, 6, 200)      # 200 candidate b's between -3 and 6
losses = [loss(b) for b in b_values]    # the loss at each one

# plot: x-axis is b, y-axis is loss
plt.plot(b_values, losses)
plt.xlabel("b (intercept)")
plt.ylabel("loss")
plt.show()