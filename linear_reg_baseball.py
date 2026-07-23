import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0

  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))

  b_gradient = -(2/N) * diff  

  return b_gradient

def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0

  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))

  m_gradient = -(2/N) * diff  

  return m_gradient

# Step Gradient Function:
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)

    return [b, m]
  
# Gradient Descent Function:
def gradient_descent(x, y, learning_rate, num_iterations):
  b = 0
  m = 0

  for i in range(num_iterations):
    b, m = step_gradient(b, m, x, y, learning_rate)
    
  return [b, m]

here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "baseball_players.csv"))
df = df.dropna(subset=["Height(inches)", "Weight(pounds)"])
df = df.reset_index(drop=True)

X = df["Height(inches)"]
y = df["Weight(pounds)"]
print(df)

plt.plot(X, y, "o")

b, m = gradient_descent(X, y, num_iterations=1000, learning_rate=0.0001)

# See the line that was settled upon
y_pred = [(m * x) + b for x in X]

plt.plot(X, y_pred)
plt.show()