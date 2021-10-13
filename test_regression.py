from nuq import NuqRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
sns.set()

x_distribution = sps.Beta(5, 2)
xi_distribution = sps.norm
mean = lambda x : np.sin(x * 2 * np.pi)
sigma = lambda x : np.exp(-np.abs)

n = 100
sample = x_distribution.rvs(size=n)
y = mean(sample) + sigma(sample) * xi_distribution(size=n)

plt.figure(figsize=(10, 20))
plt.scatter(sample, y)

grid = np.linspace(0, 1, 100)
plt.plot(grid, mean(grid), label='mean')
plt.savefig('fig.jpg')



