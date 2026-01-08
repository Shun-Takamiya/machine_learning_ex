import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets

# ワインデータのダウンロード
wine = datasets.load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
target = pd.DataFrame(wine.target, columns=["class"])

df = pd.concat([data, target], axis=1)

df.head(5)

# データの先頭5行を表示
print(df.head())


X = df[["color_intensity","proline"]]
y = wine.target

X.head(5)

# データの先頭5行を表示
print(X.head())


#k-means法で必要なライブラリ
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 特徴量の標準化
sc = StandardScaler()
X_std = sc.fit_transform(X)

# K-Meansのモデルを作成
model2 = KMeans(n_clusters=2, random_state=7)
model3 = KMeans(n_clusters=3, random_state=7)
model4 = KMeans(n_clusters=4, random_state=7)

#モデルの訓練
model2.fit(X_std)
model3.fit(X_std)
model4.fit(X_std)

plt.figure(figsize=(8,12)) #プロットのサイズ指定

color2 = np.array(['r','g'])
color3 = np.array(['r','g','b'])
color4 = np.array(['r','g','b','y'])
# クラスタ数2のK-Meansの散布図
plt.subplot(3, 1, 1)
plt.scatter(X_std[:,0], X_std[:,1], c=color2[model2.labels_])
plt.scatter(model2.cluster_centers_[:,0], model2.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.title('K-means(n_clusters=2)')

# クラスタ数3のK-Meansの散布図
plt.subplot(3, 1, 2)
plt.scatter(X_std[:,0], X_std[:,1], c=color3[model3.labels_])
plt.scatter(model3.cluster_centers_[:,0], model3.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.title('K-means(n_clusters=3)')

# クラスタ数4のK-Meansの散布図
plt.subplot(3, 1, 3)
plt.scatter(X_std[:,0], X_std[:,1], c=color4[model4.labels_])
plt.scatter(model4.cluster_centers_[:,0], model4.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.title('K-means(n_clusters=4)')

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました．")

plt.figure(figsize=(8,8)) #プロットのサイズ指定

# 色とプロリンの散布図
plt.subplot(2, 1, 1)
plt.scatter(X_std[:,0], X_std[:,1],c=y)
plt.title('training data y')

# K-Meansの散布図
plt.subplot(2, 1, 2)
plt.scatter(X_std[:,0], X_std[:,1], c=model3.labels_)
plt.scatter(model3.cluster_centers_[:,0], model3.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.title('K-means(n_clusters=3)')

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました．")


import numpy as np
import matplotlib.pyplot as plt

# ガウス関数
def gauss(x, a=1, mu=0, sigma=1):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))

x = np.arange(-4, 4, 0.1)
f = gauss(x)

# Figureを作成
fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111)
ax.set_title("Gaussian Function", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
ax.grid()
ax.set_xlim([-4, 4])
ax.set_ylim([0, 1.2])

f1 = gauss(x)

# Axesにガウス関数を描画
ax.plot(x, f, label="a=1.0, μ=0, σ=1")
ax.legend(fontsize=14)

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました．")


# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# ワインデータのダウンロード
wine = datasets.load_wine()
X = wine.data[:,[9,12]]
y = wine.target

# 特徴量の標準化
sc = StandardScaler()
X_std = sc.fit_transform(X)

# covariance_typeに'diag'を指定しGMMのモデルを作成
model = GaussianMixture(n_components=3, covariance_type='diag', random_state=1)

#モデルの訓練
model.fit(X_std)

# covariance_typeに'full'を指定し GMMのモデルを作成
model2 = GaussianMixture(n_components=3, covariance_type='full', random_state=1)

#モデルの訓練
model2.fit(X_std)

plt.figure(figsize=(8,8)) #プロットのサイズ指定
# 色とプロリンの散布図のGMM(diag)によるクラスタリング
plt.subplot(2, 1, 1)

x = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
y = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -model.score_samples(XX)
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z, levels=[0.5, 1, 2 ,3 ,4, 5]) # 等高線のプロット
plt.scatter(X_std[:,0], X_std[:,1], c=model.predict(X_std))
plt.scatter(model.means_[:,0], model.means_[:,1],s=250, marker='*',c='red')
plt.title('GMM(covariance_type=diag)')

# 色とプロリンの散布図のGMM(full)によるクラスタリング
plt.subplot(2, 1, 2)

x = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
y = np.linspace(X_std[:,0].min(), X_std[:,0].max(), 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -model2.score_samples(XX)
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z, levels=[0.5, 1, 2 ,3 ,4, 5]) # 等高線のプロット
plt.scatter(X_std[:,0], X_std[:,1], c=model2.predict(X_std))
plt.scatter(model2.means_[:,0], model2.means_[:,1],s=250, marker='*',c='red')
plt.title('GMM(covariance_type=full)')

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました．")

model.predict(X_std) #予測