"""
    Chapter3 42.    ロジスティック回帰の実装に取り組み，コードを実装せよ．
                    fit関数を完成させることで再急降下法を用いたパラメータ推定ができるようにする.
                    (オプション課題)newton_fit関数を完成させることでニュートン法を用いたパラメータ推定を行えるようにする.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from copy import deepcopy
class MyLogisticRegression:
    """ロジスティック回帰実行クラス

    Attributes
    ----------
    eta : float
        学習率
    epoch : int
        エポック数
    random_state : int
        乱数シード
    is_trained : bool
        学習完了フラグ
    num_samples : int
        学習データのサンプル数
    num_features : int
        特徴量の数
    w : NDArray[float]
        パラメータベクトル
    costs : NDArray[float]
        各エポックでの損失関数の値の履歴
    W: list
        各エポックでのパラメータベクトルの履歴（グラフ描画用）

    Methods
    -------
    fit -> None
        学習データについてパラメータベクトルを適合させる
    predict -> NDArray[int]
        予測値を返却する
    """
    def __init__(self, eta=0.01, epsilon=0.05, random_state=42):
        self.eta = eta
        self.epsilon = epsilon
        self.random_state = random_state
        self.is_trained = False

    def fit(self, X, y):
        """
        学習データについてパラメータベクトルを適合させる

        Parameters
        ----------
        X : NDArray[NDArray[float]]
            学習データ: (num_samples, num_features)の行列
        y : NDArray[int]
            学習データの教師ラベル: (num_features, )のndarray
        """
        self.num_samples = X.shape[0]  # サンプル数
        self.num_features = X.shape[1]  # 特徴量の数
        
        # 乱数生成器
        rgen = np.random.RandomState(self.random_state)
        
        # 正規乱数を用いてパラメータベクトルを初期化
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1+self.num_features)
        self.W = []  # 各エポックでのパラメータベクトルの値を格納する配列
        
        net_input = self._net_input(X)
        output = self._activation(net_input)
        self.costs = [(-y @ np.log(output)) - ((1-y) @ np.log(1-output))]  # 各エポックでの損失関数の値を格納する配列
        #xの先頭に1を追加する
        Xd=np.hstack((np.ones((self.num_samples,1)),X))
        # パラメータベクトルの更新
        while True:
            
            
            # 式(2)
            #更新式を埋める
            self.w+=self.eta * (Xd.T @ (y - output))
            
            net_input = self._net_input(X)
            output = self._activation(net_input)
            
            # 損失関数: 式(1.3)
            # 損失関数を求めるコードで埋める
            cost = (-y @ np.log(output)) - ((1-y) @ np.log(1-output))

            #更新が一定以下であれば終了
            if abs(self.costs[-1]-cost)<self.epsilon:
                break
                
            self.costs.append(cost)
            #本来は必要ないが，グラフの描画用
            self.W.append(deepcopy(self.w))
            
        # 学習完了のフラグを立てる
        self.is_trained = True
    
    #ニュートン法を用いたパラメータ推定
    def newton_fit(self, X, y):
        """
        学習データについてパラメータベクトルを適合させる

        Parameters
        ----------
        X : NDArray[NDArray[float]]
            学習データ: (num_samples, num_features)の行列
        y : NDArray[int]
            学習データの教師ラベル: (num_features, )のndarray
        """
        self.num_samples = X.shape[0]  # サンプル数
        self.num_features = X.shape[1]  # 特徴量の数
        
        # 乱数生成器
        rgen = np.random.RandomState(self.random_state)
        
        # 正規乱数を用いてパラメータベクトルを初期化
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1+self.num_features)
        self.W = []  # 各エポックでのパラメータベクトルの値を格納する配列
        
        net_input = self._net_input(X)
        output = self._activation(net_input)
        self.costs = [(-y @ np.log(output)) - ((1-y) @ np.log(1-output))]  # 各エポックでの損失関数の値を格納する配列
        
        #xの先頭に1を追加する
        Xd=np.hstack((np.ones((self.num_samples,1)),X))
        # パラメータベクトルの更新
        while True:
            #ヘッセ行列を求める
            #コードを埋める
            H = Xd.T @ (np.diag(output * (1 - output))) @ Xd
            #ヘッセ行列の逆行列を求める
            #コードを埋める
            H_inv = np.linalg.inv(H)
            # 更新式
            #コードを埋める
            self.w += H_inv @ (Xd.T @ (y - output))
            
            net_input = self._net_input(X)
            output = self._activation(net_input)
            
            # 損失関数: 式(1)
            #コードを埋める
            cost = (-y @ np.log(output)) - ((1-y) @ np.log(1-output))

            if abs(self.costs[-1]-cost)<self.epsilon:
                break
                
            self.costs.append(cost)
            #本来は必要ないが，グラフの描画用
            self.W.append(deepcopy(self.w))
            
        # 学習完了のフラグを立てる
        self.is_trained = True
        
    def predict(self, X):
        """
        予測値を返却する

        Parameters
        ----------
        X : NDArray[NDArray[float]]
            予測するデータ: (any, num_features)の行列

        Returens
        -----------
        NDArray[int]
            0 or 1 (any, )のndarray
        """
        if not self.is_trained:
            raise Exception('This model is not trained.')
        return np.where(self._activation(self._net_input(X)) >= 0.5, 1, 0)

    def _net_input(self, X):
        """
        データとパラメータベクトルの内積を計算する

        Parameters
        --------------
        X : NDArray[NDArray[float]]
            データ: (any, num_features)の行列

        Returns
        -------
        NDArray[float]
            データとパラメータベクトルの内積の値
        """
        return X @ self.w[1:] + self.w[0]

    def _activation(self, z):
        """
        活性化関数（シグモイド関数）

        Parameters
        ----------
        z : NDArray[float]
            (any, )のndarray

        Returns
        -------
        NDArray[float]
            各成分に活性化関数を適応した (any, )のndarray
        """
        #シグモイド関数を求める
        #コードを埋める
        return 1.0 / (1.0 + np.exp(-z))

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Species'])
df = pd.concat([X, y], axis=1)
# 品種 setosa、versicolorを抽出
df = df[(df['Species']==0) | (df['Species']==1)]
# 説明変数
X = df.iloc[:, [2,3]]
# 目的変数
y = df.iloc[:, 4]
# 学習データと検証データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = MyLogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

fig, ax = plt.subplots()
ims = []
ax.set_ylim(-0.25, 2)
ax.set_xlabel(X.columns[0])
ax.set_ylabel(X.columns[1])
ax.scatter(X.iloc[:, 0][y==0], X.iloc[:, 1][y==0], color='lightskyblue', label=data.target_names[0])
ax.scatter(X.iloc[:, 0][y==1], X.iloc[:, 1][y==1], color='sandybrown', label=data.target_names[1])
for n,w in enumerate(model.W):
    w_0,w_1,w_2 = w
    x1 = np.linspace(0, 6, 30)
    x2 = (-w_1 * x1 - w_0) / w_2
    # 決定境界およびデータ点を描画
    img = ax.plot(x1, x2, color='gray')
    img2=ax.text(2.5, 2.1,f"n={n}")
    ims.append(img+[img2])
print("収束までの繰り返し回数:",len(model.W))
ani = animation.ArtistAnimation(fig, ims, interval=100)
#ani.save('MyLogisticRegression.gif', writer='imagemagick')
plt.plot()

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました．")


model = MyLogisticRegression()
model.newton_fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


fig, ax = plt.subplots()
ims = []
ax.set_ylim(-0.25, 2)
ax.set_xlabel(X.columns[0])
ax.set_ylabel(X.columns[1])
ax.scatter(X.iloc[:, 0][y==0], X.iloc[:, 1][y==0], color='lightskyblue', label=data.target_names[0])
ax.scatter(X.iloc[:, 0][y==1], X.iloc[:, 1][y==1], color='sandybrown', label=data.target_names[1])
for n,w in enumerate(model.W):
    w_0,w_1,w_2 = w
    x1 = np.linspace(0, 6, 30)
    x2 = (-w_1 * x1 - w_0) / w_2
    # 決定境界およびデータ点を描画
    img = ax.plot(x1, x2, color='gray')
    img2=ax.text(2.5, 2.1,f"n={n}")
    ims.append(img+[img2])
print("収束までの繰り返し回数:",len(model.W))
ani = animation.ArtistAnimation(fig, ims, interval=500)
#ani.save('MyLogisticRegression.gif', writer='imagemagick')
plt.plot()

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました．")