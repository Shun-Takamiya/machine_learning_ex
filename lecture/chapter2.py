import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn

wine = pd.read_csv("./data/winequality-red.csv", sep=";") # sepは区切り文字の指定

# display() の代わりに print() を使う
print(wine.head(5))

X = wine[["density"]].values # 説明変数 X
T = wine["alcohol"].values # 目的変数 T

# 平均を引く
X = X - X.mean()
T = T - T.mean()

# データを訓練データとテストデータに分割
X_train = X[:1000, :]
T_train = T[:1000]
X_test = X[1000:, :]
T_test = T[1000:]

fig, axes = plt.subplots(ncols=2, figsize=(12, 4)) # ncols=2 は横に2つ並べる指定, figsize は図全体のサイズ指定

axes[0].scatter(X_train, T_train, marker=".") # axes[0] は左側のグラフ領域を指す
axes[0].set_title("train")
axes[1].scatter(X_test, T_test, marker=".") # axes[1] は右側のグラフ領域を指す
axes[1].set_title("test")
# fig.show() は非推奨なので plt.show() を使う

print("グラフを表示します...")
plt.show() # これがグラフウィンドウを表示するための命令
print("グラフウィンドウを閉じました。")


# このままではデータの並び順が偏っている可能性があるので，ランダムにシャッフルする

np.random.seed(0) # random の挙動を固定

p = np.random.permutation(len(X)) # random な index のリスト
X = X[p]
T = T[p]

X_train = X[:1000, :]
T_train = T[:1000]
X_test = X[1000:, :]
T_test = T[1000:]

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

axes[0].scatter(X_train, T_train, marker=".")
axes[0].set_title("train")
axes[1].scatter(X_test, T_test, marker=".")
axes[1].set_title("test")
# fig.show() は非推奨なので plt.show() を使う

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました。")


class MyLinearRegression(object):
    def __init__(self):
        """
        Initialize a coefficient and an intercept.
        """
        self.a = 0
        self.b = 0
        
    def fit(self, X, y):
        """
        X: data, array-like, shape (n_samples, n_features)
        y: array, shape (n_samples,)
        Estimate a coefficient and an intercept from data.
        """
        # 単回帰の最小二乗法の公式（解析解）を用いて係数を計算するコードを実装

        # scikit-learn準拠のためXは2次元配列(n_samples, 1)で渡されると想定
        # 計算のため1次元配列(n_samples,)に変換
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()
        
        # yも1次元配列(n_samples,)であることを確認
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        # ステップ1: 平均を求める
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # ステップ2: 傾き a を求める
        # 公式: a = Sxy / Sxx
        # Sxy (共分散の分子) = Σ(X - x_mean)(y - y_mean)
        # Sxx (分散の分子)   = Σ(X - x_mean)^2

        numerator = np.sum((X - x_mean) * (y - y_mean)) # 分子 Sxy
        denominator = np.sum((X - x_mean)**2)           # 分母 Sxx

        # 0除算の回避（念のため）
        if denominator == 0:
            self.a = 0
        else:
            self.a = numerator / denominator

        # ステップ3: 切片 b (w0) を求める
        # 公式: b = y_mean - a * x_mean
        self.b = y_mean - self.a * x_mean

        return self
    
    def predict(self, X):
        """
        Calc y from X
        """

        # Xが2次元配列(n_samples, 1)の場合，1次元に変換
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()

        y = self.a * X + self.b

        return y


clf = MyLinearRegression()
clf.fit(X_train, T_train)
# 回帰係数
print("係数: ", clf.a)
# 切片
print("切片: ", clf.b)

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

axes[0].scatter(X_train, T_train, marker=".")
axes[0].plot(X_train, clf.predict(X_train), color="red")
axes[0].set_title("train")

axes[1].scatter(X_test, T_test, marker=".")
axes[1].plot(X_test, clf.predict(X_test), color="red")
axes[1].set_title("test")
# fig.show() は非推奨なので plt.show() を使う

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました。")


print(sklearn.__version__)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()

# 予測モデルを作成
clf.fit(X_train, T_train)

# 回帰係数
print("係数: ", clf.coef_)

# 切片
print("切片: ", clf.intercept_)

# 決定係数
print("決定係数: ", clf.score(X_train, T_train))

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

axes[0].scatter(X_train, T_train, marker=".")
axes[0].plot(X_train, clf.predict(X_train), color="red")
axes[0].set_title("train")

axes[1].scatter(X_test, T_test, marker=".")
axes[1].plot(X_test, clf.predict(X_test), color="red")
axes[1].set_title("test")
# fig.show() は非推奨なので plt.show() を使う

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました。")


from sklearn import metrics
T_pred = clf.predict(X_test)
print("MAE: ", metrics.mean_absolute_error(T_test, T_pred))
print("MSE: ", metrics.mean_squared_error(T_test, T_pred))
print("決定係数: ", metrics.r2_score(T_test, T_pred))


# 1. データセットを用意する
from sklearn import datasets
iris = datasets.load_iris() # ここではIrisデータセットを読み込む
print(iris.data[0], iris.target[0]) # 1番目のサンプルのデータとラベル


# 2.学習用データとテスト用データに分割する
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

# 3. 線形SVMという手法を用いて分類する
from sklearn.svm import SVC, LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train) # 学習

# 4. 分類器の性能を測る
y_pred = clf.predict(X_test) # 予測
print(metrics.classification_report(y_true=y_test, y_pred=y_pred)) # 予測結果の評価

print('accuracy: ', metrics.accuracy_score(y_test, y_pred))
print('precision:', metrics.precision_score(y_test, y_pred, average='macro'))
print('recall:   ', metrics.recall_score(y_test, y_pred, average='macro'))
print('F1 score: ', metrics.f1_score(y_test, y_pred, average='macro'))


from sklearn.decomposition import PCA
from sklearn import datasets
iris = datasets.load_iris()

pca = PCA(n_components=2)
X, y = iris.data, iris.target
X_pca = pca.fit_transform(X) # 次元圧縮
print(X_pca.shape)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("PCA of IRIS dataset")
plt.show()

# 次元圧縮したデータを用いて分類してみる
X_train, X_test, y_train, y_test = train_test_split(X_pca, iris.target)
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_test)

from sklearn import metrics
print(metrics.classification_report(y_true=y_test, y_pred=y_pred2)) # 予測結果の評価



from sklearn.utils.estimator_checks import check_estimator
check_estimator(MyLinearRegression)