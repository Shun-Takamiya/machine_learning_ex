"""
    Chapter7 52.    それぞれの実験の結果をレポートにて報告せよ．
                    コードを変えることでどのような結果の変化が起こったか，またそれはなぜだと考えられるか考察せよ．
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings('ignore') # 実行に影響のない　warninig　を非表示にします. 非推奨.

### データの読み込み ###
wine = pd.read_csv("../text/data/winequality-red.csv", sep=";") # sepは区切り文字の指定
print(wine.head(5))

### データの準備 ###
np.random.seed(0) # random　の挙動を固定

X = wine[["density"]].values
T = wine["alcohol"].values
X = X - X.mean()
T = T - T.mean()

#X, Tそれぞれが最小値0, 最大値1になるように標準化
#追加の前処理部分
X = (X - X.min()) / (X.max() - X.min())
T = (T - T.min()) / (T.max() - T.min())

# データのシャッフルと分割
p = np.random.permutation(len(X))
X = X[p]
T = T[p]

X_train = X[:1000, :]
T_train = T[:1000]
X_test = X[1000:, :]
T_test = T[1000:]

data = dict(x=X_train, y=T_train)

### データの可視化 ###
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")
ax.scatter(X_train, T_train, marker=".", label="sampled data")
plt.legend(loc=0)
plt.show()

# 無限大の定義
inf = 10000
#inf = 100 #【実験】

### ベイズ線形回帰モデルの定義 ###
with pm.Model() as model:#以下のコードブロック中でそれぞれの分布を定義する
    
    #このコードブロック内で定義されたパラメータが順にモデルに追加されていく
    
    #https://docs.pymc.io/api/distributions/continuous.html
    #他にも色々な分布形が用意されている
    
    #各パラメータの分布定義
    sigma = pm.Gamma("sigma", mu=1, sigma=0.2)#データの分散に対する事前分布
    intercept = pm.Normal("Intercept", 0, sigma=1)#切片に対する事前分布
    x_coeff = pm.Normal("x_coeff", 0, sigma=1)#係数に関する事前分布
    
    #観測値
    x = pm.Data("x", X_train.reshape(-1))
    t = pm.Data("t", T_train.reshape(-1))
    
    
    #尤度定義
    likelihood = pm.Normal("y", mu=intercept + x_coeff * x, sigma=sigma, observed=t)
    idata = pm.sample(return_inferencedata=True)

### 事後分布の可視化 ###
plt.figure(figsize=(7, 7))
pm.plot_trace(idata)
plt.tight_layout()
plt.show()

#### 事後予測分布の取得 ###
lin_x = np.linspace(0, 1, 11)
with model:
    pm.set_data(
        {
            "x": lin_x,#新たな入力値
            "t": np.zeros_like(lin_x),#ダミーで目標値をおいておく必要がある（なんでもいい）数値を変えても結果が変わらないことを確認するのも良い
        }
    )
    post_pred = pm.sample_posterior_predictive(idata.posterior)

### 予測分布の可視化 ###
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")


#学習，推論に用いたデータの描写
ax.scatter(X_train, T_train, marker=".", label="sampled data", alpha=0.2)

#予測線の描写
#事後分布（MCMCの探索履歴）から切片と傾きをランダムにサンプリングして100個表示する
for _ in range(100):
    chain = np.random.randint(0, 3)
    draw = np.random.randint(0, 999)
    coeff = float(idata.posterior.x_coeff[chain][draw])
    inter = float(idata.posterior.Intercept[chain][draw])
    ax.plot(lin_x, lin_x*coeff+inter, c='g', alpha=0.02)

#各入力値における確率分布の描写
#今回の例では原理的にどんな入力に対しても同じ種類，　同じ分散の分布になる（平均は違う）

# stackして得た配列の軸順が (features, samples) になっていることがあるため転置して
# (samples, features) に揃える。これで pp_y[:, i] がサンプルの1次元配列になる。
pp_y = post_pred.posterior_predictive['y'].stack(sample=('chain', 'draw')).values.T

"""
エラーが出る古いコード
for i in range(1, 11, 2):
    dist = np.array(np.histogram(post_pred.y[:, i], bins=30))
    mu, var = np.average(pp_y[:, i]), np.var(pp_y[:, i])
    print(f'x: {lin_x[i]}, mu: {mu}, var: {var}')
    dist[0] = dist[0] / len(pp_y[:, i]) + lin_x[i]
    ax.plot(dist[0], dist[1][:-1], c = 'black')
    ind = np.argmax(dist[0])
    ax.scatter(lin_x[i], dist[1][ind], c = 'red')
    ax.axvline(x=lin_x[i], c = 'black')
"""

for i in range(1, 11, 2):
    #print(len(post_pred['y'][:, i]))
    counts, bin_edges = np.histogram(pp_y[:, i], bins=30)
    mu, var = np.average(pp_y[:, i]), np.var(pp_y[:, i])
    print(f'x: {lin_x[i]:.2f}, mu: {mu:.4f}, var: {var:.4f}')
    if counts.max() > 0:
        dist_x = counts / counts.max() * 0.1 + lin_x[i]
    else:
        dist_x = np.full_like(counts, lin_x[i], dtype=float)
        
    dist_y = bin_edges[:-1]
    
    ax.plot(dist_x, dist_y, c = 'black')
    
    ind = np.argmax(dist_x)
    ax.scatter(lin_x[i], dist_y[ind], c = 'red')
    ax.axvline(x=lin_x[i], c = 'black', alpha=0.3)

plt.legend(loc=0)
plt.show()

#パラメータの事後分布がどんな形かを確認
inters = np.array(idata.posterior.Intercept).reshape(-1)
coeffs = np.array(idata.posterior.x_coeff).reshape(-1)
print(f'intercept distribution: mu = {np.mean(inters)}, var = {np.var(inters)}')
print(f'x-coeff distribution  : mu = {np.mean(coeffs)}, var = {np.var(coeffs)}')