"""
Ex1 8. 標準入力された文字列を逆順で標準出力せよ．
        例題
            入力: "abcdefg"
            出力: "gfedcba"
"""

input_string = input("文字列を入力してください: ")

char_list = list(input_string)
length = len(char_list)

for i in range(length // 2):
    char_list[i], char_list[length - 1 - i] = char_list[length - 1 - i], char_list[i]

reversed_string = ''.join(char_list)
print(reversed_string)