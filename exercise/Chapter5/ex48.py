"""
    Chapter4 48.    クラスmy_GMMを完成させることで混合ガウス分布を推定するEMアルゴリズムを実装せよ.
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import animation
def Gaus(x,mu,S): #正規分布の値を取得
    #xがデータ点，muが正規分布の平均値，Sが共分散行列である
    #正規分布の値を求める
    #コードを埋める
    g = np.exp(-0.5 * (x - mu) @ np.linalg.inv(S) @ ((x - mu).T)) / np.sqrt((2 * np.pi) ** len(x) * np.linalg.det(S))
    return g

#混合ガウスモデル
class my_GMM:
    def __init__(self, n_cluster=3, random_state=42,epsilon=0.1):
        self.n_cluster = n_cluster
        self.epsilon = epsilon
        self.random_state = random_state
        self.is_trained = False
        
    def fit(self,X):
        self.X = X
        
        #パラメータの初期化
        
        #合計が1になるように初期化
        self.pi = np.ones(self.n_cluster)/self.n_cluster
        
        #平均はランダムに選んだデータ点で初期化する
        idx = np.random.choice(len(self.X), self.n_cluster, replace=False)
        self.mu = X[idx, :]
        
        #共分散行列は単位行列にする
        self.S = np.array([np.eye(len(self.X[0])) for _ in range(self.n_cluster)])
        LF=0
        
        #アニメーション描画用
        self.params=[]
        
        for _ in range(300):
            self.params.append([deepcopy(self.pi),deepcopy(self.mu),deepcopy(self.S)])
            
            #Eステップ
            new_LF = self.E_Step()

            #更新量が基準以下になれば終了
            if abs(new_LF-LF)<self.epsilon:
                return 
            LF = new_LF
            #Mステップ
            self.M_Step()
            
    #事後分布の計算(Eステップ)
    def E_Step(self):
        #割引率の計算
        #コードを埋める
        gam = np.zeros((len(self.X),self.n_cluster))
        #正規化定数
        #コードを埋める
        nom = np.sum(gam, axis=1).reshape(-1, 1)
        gam/= nom
        self.gam = gam
        #誤差関数の計算
        LF = sum(np.log(nom))[0]
        return LF
    
    #パラメーターの更新式（Mステップ）
    def M_Step(self):
        #各ガウス分布から生成されたと推測されるのサンプル数
        #コードを埋める
        N_k = np.sum(self.gam, axis=0)
        N = np.sum(N_k)
        #各ガウス分布の平均値
        #コードを埋める
        mu = (self.gam.T @ self.X) / N_k[:, np.newaxis]
        S =[]
        #各kについてΣを求める
        for k in range(self.n_cluster):
            #データ点について和をとる
            new=np.zeros((len(self.X[0]),(len(self.X[0]))))
            for n in range(len(self.X)):
                #共分散行列
                #コードを埋める
                cov=np.outer(self.X[n] - mu[k], self.X[n] - mu[k])
                #重みづけ
                #コードを埋める
                new+=(self.gam[n, k] * cov) / N_k[k]
            S.append(new)
        S=np.array(S)

        #混合率を求める
        #コードを埋める
        pi = N_k / N
        #パラメータを更新
        self.pi, self.mu, self.S = pi , mu, S


np.random.seed(0)
# テストデータ
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [2, 2], [2, 1], [1, 2], [3, 3], [3, 2], [2, 3]])
points1 = np.random.randn(80, 2)
points2 = np.random.randn(80, 2) + np.array([4,0])
points3 = np.random.randn(80, 2) + np.array([5,8])

X = np.r_[points1, points2, points3]

model=my_GMM(n_cluster=5)
model.fit(X)

plt.scatter(X[:,0],X[:,1])
plt.scatter(model.mu[:,0],model.mu[:,1])

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import animation


data=model.X
fig, ax = plt.subplots()
ax.scatter(data[:,0],data[:,1])
#関数に投入するデータを作成
x =  np.arange(min(data[:,0])-1, max(data[:,0])+1, 0.5)
y =  np.arange(min(data[:,1])-1, max(data[:,1])+1, 0.5)
X, Y = np.meshgrid(x, y)
z = np.c_[X.ravel(),Y.ravel()]
ims=[]

for n,params in enumerate(model.params):
    text=[ax.text(1.3, 11.2,f"n={n}")]
    #混合比,平均,共分散行列
    pi,mu,S = params
    #分散共分散行列の行列式
    dets = [np.linalg.det(s) for s in S]
    #分散共分散行列の逆行列
    invs = [np.linalg.inv(s) for s in S]
    #二次元正規分布の確率密度を返す関数
    def gaussian(x):        
        n = x.ndim
        i=0
        p= sum([pi[i]*np.diag(np.exp(-(x - mu[i])@invs[i]@((x - mu[i]).T)/2.0)) / (np.sqrt((2 * np.pi) ** n * dets[i])) for i in range(model.n_cluster)])
        return p
    Z = gaussian(z)
    shape = X.shape
    Z = Z.reshape(shape)
    CS = ax.contourf(x,y, Z, alpha=0.4)
    ims.append(CS.collections+text)
ani = animation.ArtistAnimation(fig, ims, interval=200)
#ani.save('MyGMM.gif', writer='imagemagick')
plt.show()