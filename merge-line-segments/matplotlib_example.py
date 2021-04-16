import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("orientation_compatibility_plt.csv", index_col = 0 )
print(df.head())
x = []
y = []
s = []
for i, index in enumerate(df.index):
    for j,column in enumerate( df.columns):
        x.append(int(column))
        y.append(int(index))
        s.append(df.iloc[i][j])

plt.scatter(x,y,s = 50, c = s)
plt.show()
"""
# Fixing random state for reproducibility
np.random.seed(19680801)


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
"""
