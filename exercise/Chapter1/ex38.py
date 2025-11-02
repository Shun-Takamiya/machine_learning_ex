"""
Chapter1 38.    sample.csvについて, 欠損値を補完して表示せよ．
                補完の方法は，いくつか存在する．
                それらを調べて確認してみよう．
"""

import pandas as pd

df = pd.read_csv('./data/sample.csv')

print("元のデータフレーム:")
print(df)
print("\n")

# 補完方法 1: 欠損値を含む行を削除
df_dropped = df.dropna()

print("補完方法 1: 欠損値を含む行を削除(dropna)")
print(df_dropped)
print("\n")

# 補完方法 2: 欠損値を0で埋める
df_filled_zero = df.copy() # 元のdfを変更しないようにコピー
df_filled_zero['age'] = df_filled_zero['age'].fillna(0)
df_filled_zero['score'] = df_filled_zero['score'].fillna(0)

print("補完方法 2: 欠損値を0で埋める(fillna(0))")
print(df_filled_zero)
print("\n")

# 補完方法 3: 欠損値を各列の平均値で埋める
# 'age' と 'score' の列（欠損値を除いた）の平均値を計算し，それでNaNを埋める。
df_filled_mean = df.copy()

# 'age' の平均値を計算 (NaNを無視して計算)
age_mean = df_filled_mean['age'].mean()
df_filled_mean['age'] = df_filled_mean['age'].fillna(age_mean)

# 'score' の平均値を計算 (NaNを無視して計算)
score_mean = df_filled_mean['score'].mean()
df_filled_mean['score'] = df_filled_mean['score'].fillna(score_mean)

print("補完方法 3: 欠損値を各列の平均値で埋める(fillna(平均値))")
print(df_filled_mean)
print("\n")

# 補完方法 4: 前の値で埋める（前方補完, ffill）
df_filled_ffill = df.copy()

# df_filled_ffill = df_filled_ffill.fillna(method='ffill') # <- この書き方は古い (deprecated)
# 警告メッセージに従い、 .ffill() メソッドを使用するのが現在の推奨される方法
df_filled_ffill = df_filled_ffill.ffill()

print("補完方法 4: 前の値で埋める（前方補完, ffill）")
print(df_filled_ffill)
print("\n")