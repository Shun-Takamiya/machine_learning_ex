"""
Chapter0 23.    整数0から100までの一様乱数をn個のリストを返す関数random_nを実装せよ.
                整数nは標準入力で受け取る.
"""

import random

def random_n(n):
    list = []
    i = 0
    for i in range(n):
        list.append(random.randint(1, 100))
    return list

n = int(input("整数nを入力してください: "))
print(random_n(n))
