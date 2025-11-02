"""
Chapter1 34.    "Adj Close"の列をグラフとしてプロットせよ．
                ただし，時系列順に並び替えて表示する．
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

df = pd.read_csv('./data/GOOG.csv')
df_date_list = list(df['Date'])
df_adj_close_list = list(df['Adj Close'])

sorted_date = sorted(df_date_list)

plt.figure(num ='Chapter 1 Exercise 34')

plt.plot(sorted_date, df_adj_close_list, color='blue', marker='o', markersize=2, linestyle='-')
plt.title('Adjusted Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.grid()

# gca() = "Get Current Axis" (現在の軸を取得)
ax = plt.gca() 

# X軸のロケーター（目盛りの位置決め）を設定／MaxNLocator(10)は「最大10個の目盛り（ラベル）を表示する」という意味
ax.xaxis.set_major_locator(ticker.MaxNLocator(10))

# 日付ラベルを見やすく回転させる
plt.gcf().autofmt_xdate()

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました．")
