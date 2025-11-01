"""
Chapter0 24.    標準入力から数値列を受け取り，その平均と標準偏差(母標準偏差)を標準出力せよ．
                例題
                    入力: 81.65 97.52 95.25 92.98 86.18 88.45
                    出力: 平均: 90.34, 標準偏差: 5.46
"""

import math

input_numbers = input("数値列をスペース区切りで入力してください: ")
numbers = list(map(float, input_numbers.split()))
n = len(numbers)

mean = sum(numbers) / n

dev = 0
for num in numbers:
    dev += (num - mean) ** 2

std_dev = math.sqrt(dev / n)

print(f"平均: {mean:.2f}, 標準偏差: {std_dev:.2f}")