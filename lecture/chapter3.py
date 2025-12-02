import math
import numpy as np
import matplotlib.pyplot as plt

# シグモイド関数
def sigmoid(x):
    y = 1 / (1 + math.e**(-x))
    return y

x = np.linspace(-10, 10, 100)
y = sigmoid(x)
plt.plot(x, y)
plt.show()

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# irisデータセットの読み込み
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Species'])
df = pd.concat([X, y], axis=1)
df.head()

# データの分割
print(df.head())

df.tail()

# データの分割
print(df.tail())

# 品種 setosa、versicolorを抽出
df = df[(df['Species']==0) | (df['Species']==1)]
# 説明変数
X = df.iloc[:, [2,3]]
# 目的変数
y = df.iloc[:, 4]
# 学習データと検証データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LogisticRegression()
# 学習
model.fit(X_train, y_train)

from sklearn import metrics
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# 切片
print(model.intercept_)
# 傾き
print(model.coef_)


w_0 = model.intercept_
w_1 = model.coef_[0, 0]
w_2 = model.coef_[0, 1]

x1 = np.linspace(0, 6, 30)
x2 = (-w_1 * x1 - w_0) / w_2
# 決定境界およびデータ点を描画
plt.plot(x1, x2, color='gray')
plt.scatter(X.iloc[:, 0][y==0], X.iloc[:, 1][y==0], color='lightskyblue', label=data.target_names[0])
plt.scatter(X.iloc[:, 0][y==1], X.iloc[:, 1][y==1], color='sandybrown', label=data.target_names[1])
plt.ylim(-0.25, 2)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.legend()
plt.show()

# irisデータセットの読み込み
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Species'])
df = pd.concat([X, y], axis=1)
df.head()

# データの分割
print(df.head())

df.tail()

# データの分割
print(df.tail())

# 学習データと検証データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LogisticRegression()
# 学習
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


from sklearn.datasets import make_blobs 
X, y = make_blobs(centers=2, n_samples=1000, n_features=5, cluster_std=5, random_state=121)

# 学習データと検証データを分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# データを標準化するための準備
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)

# モデル定義
# transformで標準化を行い，学習
from sklearn.svm import SVC
clf = SVC(max_iter=100)
clf.fit(scaler.transform(X_train), y_train)

# テストデータも標準化を実行し，predict(予測)を行う
y_pred = clf.predict(scaler.transform(X_test))

#正解率を算出
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#x^2の関数
def f(x):
    return x**2
#x^2の1階微分
def diff_f(x):
    return 2*x

t=np.linspace(-5,5,1000)
fig, ax = plt.subplots()
ax.plot(t,f(t), color='blue') # グラフを作成
ims = []

x=5 #xの初期値
epsilon=0.01 #収束判定
eta=0.1 #学習率

i=0
while abs(diff_f(x)*eta)>epsilon:
    
    img = ax.plot(x,f(x),'*',markersize=10, color='red')
    img2=ax.text(-0.5, 27,f"n={i}")
    ims.append(img+[img2]) # グラフを配列に追加
    
    #ここに更新式を書く
    x=x-eta*diff_f(x)
    i+=1
print("収束までの繰り返し回数:",i)
print("極小値:x=",x)
ani = animation.ArtistAnimation(fig, ims, interval=100)
#ani.save('x^2.gif', writer='imagemagick')
plt.plot()

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました。")

