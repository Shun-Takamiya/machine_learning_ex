"""
EX1. 6 ファイルに書かれている文字を標準出力せよ．
        bashでいう"cat"コマンドを実装すればよい．
        ファイル名は"sample.txt"とプログラム中で指定する．
"""

file_name = "sample.txt"

with open(file_name, "r", encoding="utf-8") as file:
    content = file.read()
    print(content)