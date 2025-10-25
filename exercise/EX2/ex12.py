"""
Ex2 12. 標準入力から数値を1つ受け取って，その数値が素数かどうか判定せよ．
        is_primeという関数を作る．
"""

import math

def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(math.sqrt(n))

    for i in range(3, limit + 1, 2):
        if n % i == 0:
            return False
            
    return True


input_value = input("数値を入力してください: ")
n = int(input_value)

if is_prime(n):
    print(f"{n}は素数です")
else:
    print(f"{n}は素数ではありません")
