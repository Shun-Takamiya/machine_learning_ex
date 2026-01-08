"""
    Chapter4 44.    ジニ係数をもとにベストなデータ分割を行う関数を作成せよ．
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

#ジニ係数の計算を行う関数
def gini_score(data, target, feat_idx, threshold):
    # 課題1で作成したものを張り付ければよい

    gini = 0
    sample_num = len(target)

    #print(data) #確認用出力
    #print(target) #確認用出力
    #print(sample_num) #確認用出力
    
    #閾値をもとにデータを分ける
    div_target = [target[data[:, feat_idx] >= threshold], target[data[:, feat_idx] < threshold]]

    #print(div_target) #確認用出力
    
    #閾値で分けたデータをもとにジニ係数を計算
    #(演習)プログラムを補完し，ジニ係数が計算できるようにせよ．
    for group in div_target:
        score = 0

        #print(group) #確認用出力
        #print(len(group)) #確認用出力
        
        #クラスの種類を求める(今回は[0,1]の二つ)
        classes = np.unique(group)
        for cls in classes:
            #groupの点がクラスclsである確率を求める
            #コードを埋める
            p = len(group[group == cls]) / len(group)
            #確率の二乗和を求める
            #コードを埋める
            score += p ** 2
        #各グループについてgini係数の期待値をとる
        #コードを埋める
        gini += (1 - score) * (len(group) / sample_num)

    return gini

#最もよいデータの分類方法をジニ係数をもとに決定する
def search_best_split(data, target):   
    features = data.shape[1]
    #gini係数を最小にする閾値
    best_thrs = None
    #gini係数を最小にするために分割基準とする特徴量
    best_f = None
    gini = None
    #最小のgini係数を格納
    gini_min = 1
    
    #すべての特徴量で探索
    for feat_idx in range(features):
        values = sorted(list(set(data[:, feat_idx])))
        values = [(values[i]+values[i-1])/2 for i in range(1,len(values))]
        #すべてのデータ点の値を閾値として探索
        for val in values:
            gini = gini_score(data, target, feat_idx, val)
            
            #より良いジニ係数であれば更新
            #if文を埋める
            if gini < gini_min:
                gini_min = gini
                best_thrs = val
                best_f = feat_idx
                
    return gini_min, best_thrs, best_f       

#決定木のメイン処理
class DecisionTreeNode(object):
    def __init__(self, data, target):
        self.left = None
        self.right = None
        self.data = data
        self.target = target
        self.threshold = None
        self.feature = None
        self.gini_min = None

    def split(self):
        #最もよいデータの分割基準を探索
        self.gini_min, self.threshold, self.feature = search_best_split(self.data, self.target)
        
        #決定した基準をもとにデータを二つに分割する
        idx_left = self.data[:, self.feature] >= self.threshold
        idx_right = self.data[:, self.feature] < self.threshold
        
        #分割したそれぞれの条件に対して，どのラベルを付与するか決定
        #分割後のデータ集合で最も数の多いクラスに分類
        self.left = np.argmax(np.bincount(self.target[idx_left]))
        self.right = np.argmax(np.bincount(self.target[idx_right]))

    def predict(self, data):
        #条件を基に学習した結果を返す
        if data[self.feature] > self.threshold:
            return self.left
        else:
            return self.right

#決定木の根を表現
class my_DecisionTreeClassifier(object):
    def __init__(self):
        self.tree = None

    def fit(self, data, target):
        self.tree = DecisionTreeNode(data, target)
        self.tree.split()

    def predict(self, data):
        pred = []
        for s in data:
            pred.append(self.tree.predict(s))
        return np.array(pred)

# irisデータセットの読み込み
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Species'])
df = pd.concat([X, y], axis=1)
# 品種 setosa、versicolorを抽出
df = df[(df['Species']==0) | (df['Species']==1)]
# 説明変数
X = df.iloc[:, [2,3]].values
# 目的変数
y = df.iloc[:, 4].values
# 学習データと検証データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#決定木を呼び出し，学習を行う
clf = my_DecisionTreeClassifier()
clf.fit(X_train,y_train)

#学習結果を基に線を引く
if clf.tree.feature==0:
    plt.vlines(x=clf.tree.threshold,ymin=int(min(X[:, 0]))-1,ymax=int(min(X[:, 0]))+1)
else:
    plt.hlines(y=clf.tree.threshold,xmin=int(min(X[:, 1]))-1,xmax=int(min(X[:, 1]))+1)
plt.scatter(X[:, 0][y==0], X[:, 1][y==0], color='lightskyblue', label=data.target_names[0])
plt.scatter(X[:, 0][y==1], X[:, 1][y==1], color='sandybrown', label=data.target_names[1])

#予測結果の確認
y_pred=clf.predict(X_test)
print(y_pred)
#正解率の計算
print(metrics.accuracy_score(y_test, y_pred))

#グラフの表示
plt.xlabel(data.feature_names[2])
plt.ylabel(data.feature_names[3])
plt.legend()
plt.show()