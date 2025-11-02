"""
Chapter1 37.    前日比が大きく変動している（"DoD"が±1USDよりも大きい）行を抽出せよ.
"""

import pandas as pd

df = pd.read_csv('./data/GOOG.csv')

# "Adj Close"列の前日との差分(diff()メソッド)を計算し，"DoD"列として追加
df['DoD'] = df['Adj Close'].diff()

# 前日比が±1USDよりも大きい行を抽出
significant_changes_df = df[df['DoD'].abs()>1.0]

print("前日比が±1USDよりも大きい行:")
print(significant_changes_df[['Date', 'Adj Close', 'DoD']])