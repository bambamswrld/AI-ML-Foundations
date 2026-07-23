# AI / ML Foundations

Building a machine learning foundation from the ground up — implementing the
core ideas by hand, without leaning on ML libraries, to actually understand
*why* they work.

## Linear Regression

Fitting a straight line `y = m*x + b` to data using **gradient descent**, written
from scratch. This is the seed the rest of the field grows from: a neural
network is, in a real sense, millions of these lines stacked together and
trained by the exact same procedure.

**Files**
- `linear_reg_basic.py` — the line model and plotting the fit over the data.
- `loss_b.py` — measuring loss and computing the gradient of the loss.

### The idea in four moves

1. **Model.** A line, `y = m*x + b`. `m` and `b` are the *parameters* — the
   knobs we get to choose. Everything is about picking them well.

2. **Loss.** For each data point, the *residual* is `actual - predicted`. Square
   the residuals and average them → **mean squared error**, one number that says
   how wrong the whole line is. Squaring kills the sign, punishes big misses
   disproportionately, and (crucially) makes the loss a smooth bowl we can
   descend with calculus.

3. **Gradient.** The derivative of the loss with respect to each parameter — the
   local slope of that bowl. Tells us which way is downhill and how steep.
   - `∂L/∂b = -2/N · Σ (y - (m*x + b))`
   - `∂L/∂m = -2/N · Σ x · (y - (m*x + b))`  ← the extra `x` means far-out points
     pull harder on the slope.

4. **Descent.** Step each parameter *against* its gradient:
   `new = current - learning_rate * gradient`. Repeat many times, re-measuring
   the slope at each new spot, until the gradient flattens to ~0 at the bottom.

### Learning rate

The `learning_rate` is the step size. Too small → converges too slowly to be
useful. Too large → overshoots the minimum and can spiral outward and diverge.
The safe ceiling depends on the *scale of the data*: the loss curvature in the
`m` direction is `(2/N)·Σx²`, so large `x` values (squared) make a narrow,
touchy bowl and force a smaller learning rate. This is exactly why feature
scaling / normalization matters in practice.

### Run it

```bash
python3 linear_reg_basic.py
```

On the sample lemonade-stand data, gradient descent discovers a best-fit line of
roughly `y = 10.46*x + 49.6` — a slope it *found*, not one it was given.

---

*Worked through as part of building ML intuition — concepts over copy-paste.*
