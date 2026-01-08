"""
    Chapter4 47.    k-means法を実装せよ.
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import animation
class My_KMeans:
    def __init__(self, n_cluster=2, random_state=42,n_iter=300):
        self.n_cluster = n_cluster
        self.n_iter = n_iter
        self.random_state = random_state
        self.is_trained = False
    #各クラスタの中心点を計算
    def fit(self,X):
        
        #結果保存用
        self.centroids_and_labels = []
        self.features = len(X[0])
        
        #代表地点をランダムに選んだデータ点で初期化する
        idx = np.random.choice(len(X), self.n_cluster, replace=False)
        #各クラスタの中心の座標を格納
        self.centroids = X[idx, :]
        
        #代表地点を基に各データ点にラベル付け
        label = [self.dist(x) for x in X]

        #n_iter回の更新、もしくは分類結果が変わらなくなれば終了
        for n in range(self.n_iter):
            self.labels = label
            
            #分類結果を基に代表地点を更新する（各クラスに属するデータ点の重心を求める）

            #代表地点を格納
            self.centroids = np.zeros((self.n_cluster,self.features))
            #いくつのデータがクラスタに含まれるか数える
            self.counts = np.zeros((self.n_cluster,1))

            #代表地点を求める
            #点の重心を求めるので座標を合計してからその平均をとればよい
            for i,x in enumerate(X):
                #コードを埋める
                self.centroids[self.labels[i]] += x
                self.counts[self.labels[i]] += 1
            #平均をとる
            self.centroids /= self.counts
            
            #animation描画用に結果を保存しておく
            self.centroids_and_labels.append([deepcopy(self.centroids),deepcopy(self.labels)])
            
            #更新された代表地点を基に再分類
            label=[self.dist(x) for x in X]
            
            #分類結果が変わらなければ終了する
            if label==self.labels:
                break
                
        #学習を終了する
        self.is_trained=True
        
    #与えられた点のクラスタを予測
    def predict(self,X):
        #学習済みでなければ処理をしない
        assert self.is_trained,"fitが実行されていません"
        
        #予測値を格納した配列を作成
        pred=[self.dist(x) for x in X]
        return pred
    
    #xと各クラスタの中心点との距離を計算し、最も距離が近いクラスタのラベルを返す
    def dist(self,x):
        #xが属するクラスタのラベル
        label=-1
        #最小の距離を格納
        min_distance=float('inf')
        for i in range(self.n_cluster):
            #距離を求める
            #コードを埋める
            distance= np.linalg.norm(x - self.centroids[i])
            #もしxからより近いクラスタを発見したら更新を行う
            #if文を埋める
            if distance < min_distance:
                label=i
                min_distance=distance
        return label


np.random.seed(10)
# テストデータ
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [2, 2], [2, 1], [1, 2], [3, 3], [3, 2], [2, 3]])
points1 = np.random.randn(80, 2)
points2 = np.random.randn(80, 2) + np.array([4,0])
points3 = np.random.randn(80, 2) + np.array([5,8])

X = np.r_[points1, points2, points3]
#実行
model = My_KMeans(n_cluster=4,n_iter=20)
model.fit(X)
# 結果を結果をplot
plt.scatter(X[:,0], X[:,1])
plt.show()
colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
fig, ax = plt.subplots()
ims=[]
#結果を可視化可視化(ラベル付き) 
for n,(centroids,label) in enumerate(model.centroids_and_labels):
    u_labels = np.unique(label)
    new=[ax.text(1.3, 3.2,f"n={n}")]
    for i in u_labels:
        new.append(plt.scatter(X[label == i , 0] , X[label == i , 1] ,color=colors[i%10], label = i))
    for i in range(model.n_cluster):
        new.append(plt.scatter(centroids[i][0] , centroids[i][1] ,color=colors[i%10], marker="*"))
    ims.append(new)
ani = animation.ArtistAnimation(fig, ims, interval=500)

plt.plot()

print("グラフを表示します...")  
plt.show()
print("グラフウィンドウを閉じました．")