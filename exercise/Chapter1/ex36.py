"""
Chapter1 36.    "Adj Close"の列に関して前日との差分をとって株価の前日比を計算し，"DoD"という列を作成せよ．
"""

import pandas as pd

df = pd.read_csv('./data/GOOG.csv')

# "Adj Close"列の前日との差分(diff()メソッド)を計算し，"DoD"列として追加
df['DoD'] = df['Adj Close'].diff()
print("前日比を計算したデータフレーム:")
print(df[['Date', 'Adj Close', 'DoD']])