"""
Chapter1 31.    100個の点を透明度のある点として表示せよ.
                各点の座標は前の問題と同様にランダムで生成
                色は0から1までの値でランダムに設定
"""

import numpy as np
import matplotlib.pyplot as plt

# 100個の点を-3から3までの一様分布でランダムに生成
x = np.random.uniform(-3, 3, 100)
y = np.random.uniform(-3, 3, 100)
value_color = np.random.random(100)

# 散布図として図示
plt.scatter(x, y, c=value_color, cmap='Blues', marker='o')
plt.title('Chapter 1 Exercise 31: Scatter Plot of Random 100 Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
print("散布図を表示します...")
plt.show()
print("散布図ウィンドウを閉じました．")