"""
Chapter1 33.    csvにある列名をリストで標準出力せよ.
"""

import pandas as pd

df = pd.read_csv('./data/GOOG.csv')
column_list = list(df.columns)
print(column_list)