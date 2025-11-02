"""
Chapter1 35.    最近1ヶ月にしぼって, データフレームを作成せよ．
"""

import pandas as pd

df = pd.read_csv('./data/GOOG.csv')

# 'Date'列を、「単なる文字列」から「日付として計算や処理ができる専用の型（datetime型）」に変換する
df['Date'] = pd.to_datetime(df['Date'])

# 日付データの確認とデバッグ用出力
# print(df['Date'])
# print(df['Date'].max())
# print(df['Date'].max() - pd.DateOffset(months=1))

# 最近1ヶ月のデータにフィルタリング
recent_df = df[df['Date'] >= (df['Date'].max() - pd.DateOffset(months=1))]

print("最近1ヶ月のデータフレーム:")
print(recent_df)