"""
Ex2 15. 文字列を受け取って，それぞれの文字の出現回数を標準出力するプログラムを作成せよ．
        文字列は[a-zA-Z]で構成されるものとする．
        大文字小文字の区別は行わない．
        出現回数はアルファベット順に以下のように出力する．
        例題
            入力: Hello world
            出力: D: 1
                E: 1
                H: 1
                L: 3
                O: 2
                R: 1
                W: 1
"""

input_string = input("文字列を入力してください: ")

alpha_chars = []

for char in input_string:
    if char.isalpha():
        alpha_chars.append(char.lower())

counts = {}
for char in alpha_chars:
    counts[char] = counts.get(char, 0) + 1

sorted_chars = sorted(counts.keys())
for char in sorted_chars:
    count = counts[char]
    print(f"{char.upper()}: {count}")