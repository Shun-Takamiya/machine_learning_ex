"""
Ex2 25. 2次元ベクトルを入力して, それを行列Aで変換した値を出力する関数transA(x)を実装せよ．
        行列Aは固定で, A=[[1, 3], [7, 6]]
        Axを返す関数を設計する. 
        例題
            入力: 2 3
            出力: 11 32
"""

import numpy as np

def transA(x):
    vec = np.array(x)
    A = np.array([[1, 3], [7, 6]])
    return np.dot(A, vec)

imput_vec = input("数値列をスペース区切りで入力してください: ")
numbers = list(map(int, imput_vec.split()))

trans_A = transA(numbers)
print(f"{trans_A}")
