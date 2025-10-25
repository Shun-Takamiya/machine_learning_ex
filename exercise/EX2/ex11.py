"""
Ex2 11. 3つの座標(x1, y1), (x2, y2), (x3, y3)をそれぞれ標準入力から受け取って，三角形の面積を標準出力せよ．
        各行に座標がx yの順にスペース区切りで入力される.
        例題
            入力: 1 3
                5 7
                9 2          
            出力: 18
"""

import math

input_value_1 = input("１つ目の座標を入力してください: ")
x1, y1 = map(float, input_value_1.split())
input_value_2 = input("２つ目の座標を入力してください: ")
x2, y2 = map(float, input_value_2.split())
input_value_3 = input("３つ目の座標を入力してください: ")
x3, y3 = map(float, input_value_3.split())

area = abs((x1*(y2 - y3) - x3*(y2 - y3) - x2*(y1 - y3) + x3*(y1 - y3)) )/ 2
print(f"{area:.0f}")