"""
EX1 4. 標準入力で受け取った文字列を1行に1文字ずつ標準出力せよ．そのとき，何番目の文字かを同時に表示．
        例題
            入力: abc
            出力:    1 a
                    2 b
                    3 c
"""

input_string = input("文字列を入力してください: ")

length = len(input_string)
for i in range(length):
    print(i + 1, input_string[i])