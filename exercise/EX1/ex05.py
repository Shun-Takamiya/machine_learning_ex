"""
EX1 5. 入力された数値に3桁区切りでカンマ","を入れて標準出力せよ．
    1234が入力された場合，"1,234"と出力する．123が入力された場合，"123"と出力する．
"""

input_value = input("数値を入力してください: ")

for i in range(len(input_value)):
    if i != 0 and i % 3 == 0:
        print(",", end="")
    print(input_value[i], end="")
print()  # 最後に改行を追加