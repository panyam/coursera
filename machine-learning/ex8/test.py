
from math import exp
import ipdb
"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    if x.ndim == 1:
        s = sum(map(exp, x))
        return np.array([10 * exp(v) / s for v in x])
    else:
        return np.array(map(softmax, x.transpose())).transpose()


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
