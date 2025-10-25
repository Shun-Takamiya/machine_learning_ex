"""
Ex2 17. 標準入力から文字列を2つ受け取り, 1つ目の文字列中に現れる2つ目の文字列の個数を数えよ.
        例題
            入力: abcdeaabccd cd
            出力: 2
"""

input_line = input("検索対象の文字列と検索文字列をスペース区切りで入力してください: ")

main_string, substring = input_line.split()
count = main_string.count(substring)
print(count)