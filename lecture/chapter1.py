import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
plt.plot(X, np.cos(X))
plt.plot(X, np.sin(X))

# これがグラフウィンドウを表示するための命令
print("グラフを表示します...")
plt.show()

print("グラフウィンドウを閉じました。")


plt.figure(figsize=(8,6), dpi=80) # リサイズ
plt.xlim(-np.pi, np.pi) # x軸方向の最大値，最小値を決める

plt.xticks( [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.yticks([-1, 0, +1])


X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
plt.plot(X, np.cos(X), color='blue', linewidth=2.0, linestyle='-', label='cos') # 線のスタイル，幅などを変更できる
plt.plot(X, np.sin(X), color='green', linewidth=2.0, linestyle=':', label='sin')
plt.legend(loc='upper left')

# これがグラフウィンドウを表示するための命令
print("グラフを表示します...")
plt.show()

print("グラフウィンドウを閉じました。")

name = pd.Series(['Mike', 'John', 'Ken', 'Maki', 'Lisa'], name='name') # RDBの1列
country = pd.Series(['USA', 'UK', 'Japan', 'China', 'USA'], name='country')
score = pd.Series([1, 3, 5, 4, 3], name='score')
df = pd.DataFrame({'name':name, 'country':country, 'score':score}) # RDB全体
df.head()

print(df)
print(df.describe())

print(df[['country', 'score']])
print(df[df['score'] >= 3] )

df = pd.read_csv('./data/sample.csv')
print(df.head())