"""
Chapter1 30.    10個の点を表示せよ.
                各点の座標は-3から3までの一様分布でランダムに生成
                散布図として図示する
"""

import numpy as np
import matplotlib.pyplot as plt

# 10個の点を-3から3までの一様分布でランダムに生成
x = np.random.uniform(-3, 3, 10)
y = np.random.uniform(-3, 3, 10)

# 散布図として図示
plt.scatter(x, y, color='blue', marker='o')
plt.title('Chapter 1 Exercise 30: Scatter Plot of Random 10 Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
print("散布図を表示します...")
plt.show()
print("散布図ウィンドウを閉じました．")