"""
Ex2 13. 標準入力で受け取った整数nまでの素数を全て列挙する関数get_all_primesを作成し, そのリストをカンマ区切りで標準出力せよ．
        例題
            入力: 12
            出力: 2,3,5,7,11
"""

import math

def get_all_primes(n):
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        if num == 2:
            primes.append(num)
            continue
        if num % 2 == 0:
            continue

        limit = int(math.sqrt(num))
        for i in range(3, limit + 1, 2):
            if num % i == 0:
                is_prime = False
                break
        if is_prime == True:
            primes.append(num)
    return primes

input_value = input("数値を入力してください: ")
n = int(input_value)
primes = get_all_primes(n)
print(",".join(map(str, primes)))
