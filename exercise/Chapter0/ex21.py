"""
Chapter0 21.    平方根を求める関数square_root(x)を実装せよ．
                xに負の値が入力された場合には-1を返すようにせよ.
                xに数値以外が入力された場合にはNoneを返すようにせよ.
"""

import math

def square_root(x):
    try:
        value = float(x)
        if value < 0:
            return -1
        return math.sqrt(value)
    
    except ValueError:
        return None

input_value = input("平方根を求めたい数値を入力してください: ")
result = square_root(input_value)
print(f"{input_value}の平方根は: {result}")