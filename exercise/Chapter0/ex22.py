"""
Chapter0 22.    サイコロシミュレータ関数diceを実装せよ.
                出目は1-6でそれぞれの出目の確率は同様に確からしいとする.
                (一度の実行で1-6の整数が1つ返る関数を作成する. ）
"""

import random

def dice():
    return random.randint(1, 6)

roll_result = dice()
print(f"サイコロの出目は: {roll_result}")
