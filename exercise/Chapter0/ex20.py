"""
Chapter 0 20.   余接関数を求める関数cot(theta)を実装せよ．
                thetaはラジアンで入力されるとする.
                三角関数の値はmathやnumpy等を用いてよい.
"""

import math

def cot(theta):
    return 1 / math.tan(theta)

input_angle = float(input("角度を度数法で入力してください: "))
theta = math.radians(input_angle)
cotangent = cot(theta)
print(f"{input_angle}度の余接は: {cotangent}")