"""
Chapter1 32.    平均0, 標準偏差2の正規分布から生成したサンプルのヒストグラムを表示せよ.
"""

import numpy as np
import matplotlib.pyplot as plt

# 正規分布に従う1000個の点を生成（平均0，標準偏差2）
mu = 0
sigma = 2
num_points = 1000
x = np.random.normal(mu, sigma, num_points)

# ヒストグラムの表示
plt.hist(x, bins=30, color='blue' , alpha=0.7, edgecolor='black')
plt.title('Chapter 1 Exercise 32: Histogram of 1000 Random Points from Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid()
print("ヒストグラムを表示します...")
plt.show()
print("ヒストグラムウィンドウを閉じました．")