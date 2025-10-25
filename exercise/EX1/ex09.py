"""
Ex1 9. 標準入力された文字列の長さを標準出力せよ．
        例題
            入力: "abc"
            出力: 3
"""

input_string = input("文字列を入力してください: ")

count = 0
for char in input_string:
    count += 1
print(count)