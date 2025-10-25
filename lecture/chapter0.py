import math
import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
# %matplotlib inline


print(math.sqrt(2))
print(math.exp(2))
print(math.sin(2))
print(math.pi, math.e)
vec = np.array([[1, 2]]) # 1次元の配列をNumpy配列(ベクトル)にする
matrix = np.array([[1, 3], [6, 4]]) # 2次元の配列をNumpy配列(行列)にする
print(vec)
print(matrix)

print(np.ones(3)) # 3次元の1ベクトル
print(np.zeros(3)) # 3次元の0ベクトル
print(np.ones((2, 3))) #2x3次元の1行列
print(np.zeros((2, 3))) #2x3次元の0行列

print("dtype", matrix.dtype) # 型を表示する
print("shape", matrix.shape) # サイズ(次元)

A = np.array([[1, 3], [6, 4]])
print("*演算:\n", A*A) # 行列Aどうしの要素積
print("np.dot:\n", np.dot(A, A)) # 行列Aどうしの行列積

B = np.matrix([[1, 3], [6, 4]])
print("*演算:\n", B*B) # 行列Bどうしの行列積
print("np.dot:\n", np.dot(B, B)) # 行列Bどうしの行列積


def f(x):
    return x ** 2 + 17 * np.sin(x)

x = np.arange(-10, 10, 0.1)
x_star = fmin_bfgs(f, 0) # 関数f(x)の最適化(最小値をとるxを求める)
plt.xlim(-10, 10)
plt.plot(x, f(x))
plt.scatter(x_star, f(x_star), c='red')

# これがグラフウィンドウを表示するための命令
print("グラフを表示します...")
plt.show()

print("グラフウィンドウを閉じました。")