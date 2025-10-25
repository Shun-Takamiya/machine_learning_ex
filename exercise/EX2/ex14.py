"""
Ex2 14. 空白で区切られた整数を受け取って昇順に並び替えて標準出力せよ．
        並び替えにはsortメソッドなどを使用しても良い.
        例題
            入力: 10 5 2 7 8
            出力: 2 5 7 8 10
"""

input_value = input("整数を空白で区切って入力してください: ")

numbers = list(map(int, input_value.split()))
numbers.sort()
print(" ".join(map(str, numbers)))