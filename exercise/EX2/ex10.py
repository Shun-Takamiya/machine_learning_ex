"""
Ex2 10. 2つの点の座標(x1, y1), (x2, y2)を標準入力から受け取り，その点の間のユークリッド距離を標準出力せよ．
        座標は小数で与えられ, ユークリッド距離は小数点以下1桁以上表示する.
        例題
            入力: 1 2
                1 3
            出力: 1.0
"""

import math

input_value_1 = input("１つ目の座標を入力してください: ")
x1, y1 = map(float, input_value_1.split())
input_value_2 = input("２つ目の座標を入力してください: ")
x2, y2 = map(float, input_value_2.split())

distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
print(f"{distance:.1f}")