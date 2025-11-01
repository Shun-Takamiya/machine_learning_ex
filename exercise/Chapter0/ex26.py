"""
Chapter0 26.    25を書き換えて, 行列Aも入力として受け取るような形にせよ. 
                ベクトルの次元数d, 変換後の次元数mとする.
                入力フォーマットは次のようにする．
                d m
                e1 e2 ... ed
                A11 A12 ... A1d
                A21 A22 ... A2d
                :
                Am1 Am2 ... Amd

                例題
                    入力: 
                        2 3
                        1 7
                        2 4 
                        1 7
                        10 4
                    出力: 
                        30 50 38
"""

import numpy as np

def transA(x):
    numbers = list(map(int, x.split()))
    d = numbers [0]
    m = numbers [1]
    vec = np.array(numbers[2 : 2 + d])
    A_elements = np.array(numbers[2 + d :])
    A = A_elements.reshape(m, d)
    return np.dot(A, vec)

imput_vec = input("数値列をスペース区切りで入力してください: ")

trans_A = transA(imput_vec)
print(f"{trans_A}")