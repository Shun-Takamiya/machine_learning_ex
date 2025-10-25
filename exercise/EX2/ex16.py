"""
Ex2 16. ファイル名をコマンドライン引数で受け取り，そのファイル内の単語の頻度を降順に標準出力せよ．
        ファイル内には単語がスペース区切りで書かれている．
        ファイルは複数行あり，全ての行にわたっての単語の頻度を求める．
        単語の頻度が同じ場合は，単語の辞書順に出力する．
        例題
            ファイル内のデータ: sample.txt
                            aaa aa ab aa
                            aa ba aaa ab
            出力: aa 3
                aaa 2
                ab 2
                ba 1
"""

import sys

if len(sys.argv) < 2:
    print("エラー: ファイル名を指定してください。")
    print("使い方: python ex16.py <ファイル名>")
    sys.exit(1) # プログラムを終了

file_name = sys.argv[1]

with open(file_name, 'r', encoding='utf-8') as f:

    words = f.read().split()

    word_counts = {}

    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    sorted_counts = sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))

    for word, count in sorted_counts:
        print(f"{word} {count}")