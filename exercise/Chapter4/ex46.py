"""
    Chapter4 46.    my_RandomForestClassifierを完成させ, ランダムフォレストを実装せよ．
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.datasets import make_moons
import random
import matplotlib.pyplot as plt # グラフ等の描画用モジュール
from matplotlib.colors import ListedColormap # 描画時のカラー指定に使用

#決定木をフルスクラッチで実装した場合は先に実行しておく
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
    # 課題2で作成したものを張り付ければよい
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
    def __init__(self, data, target,depth,max_depth,is_leaf=False):
        self.left = None
        self.right = None
        self.data = data
        self.target = target
        self.threshold = None
        self.feature = None
        self.gini_min = None
        
        #深さの管理用
        self.max_depth = max_depth
        self.depth = depth
        #自身が葉であるかのbool値
        self.is_leaf = is_leaf
        self.label = np.argmax(np.bincount(target))
    def split(self):
        #同じラベルのデータしか含まれないなら分割を終了
        if len(self.target)==max(np.bincount(self.target)):
            self.leaf=True
            
        #最もよいデータの分割基準を探索
        self.gini_min, self.threshold, self.feature = search_best_split(self.data, self.target)
        #決定した基準をもとにデータを二つに分割する
        idx_left = self.data[:, self.feature] >= self.threshold
        idx_right = self.data[:, self.feature] < self.threshold
        
        #深さが最大に達するか、分割の必要がない場合は終了
        if self.depth==self.max_depth-1 or self.gini_min==0:
            #分割したそれぞれの条件に対して，どのラベルを付与するか決定
            #分割後のデータ集合で最も数の多いクラスに分類
            #コードを埋める
            self.left = DecisionTreeNode(self.data[idx_left], self.target[idx_left], self.depth+1, self.max_depth, is_leaf=True)
            self.right = DecisionTreeNode(self.data[idx_right], self.target[idx_right], self.depth+1, self.max_depth, is_leaf=True)
        #分岐を続ける
        else:
            #コードを埋める
            self.left = DecisionTreeNode(self.data[idx_left], self.target[idx_left], self.depth+1, self.max_depth)
            if len(set(self.target[idx_left]))>1:
                self.left.split()
            else:
                self.left.is_leaf = True
            #コードを埋める
            self.right = DecisionTreeNode(self.data[idx_right], self.target[idx_right], self.depth+1, self.max_depth)
            if len(set(self.target[idx_right]))>1:
                self.right.split()
            else:
                self.right.is_leaf = True
            
    def predict(self, data):
        #葉に到達した場合
        if self.is_leaf:
            return self.label
        
        #条件を基に学習した結果を返す
        if data[self.feature] >= self.threshold:
            return self.left.predict(data)
        else:
            return self.right.predict(data)

#決定木の根を表現
class my_DecisionTreeClassifier(object):
    def __init__(self,max_depth):
        self.tree = None
        
        #深さを表すパラメータ
        self.max_depth = max_depth
        
    def fit(self, data, target):
        #最初の深さは0
        initial_depth = 0
        self.tree = DecisionTreeNode(data,target,initial_depth,self.max_depth)
        self.tree.split()

    def predict(self, data):
        pred = []
        for s in data:
            pred.append(self.tree.predict(s))
        return np.array(pred)

#ランダムフォレストのフルスクラッチ
class my_RandomForestClassifier(object):
    def __init__(self,tree_num,sample_num,max_depth):
        self.trees = []
        self.datas = []
        self.tree_num = tree_num
        self.sample_num = sample_num
        self.max_depth = max_depth
        
    def fit(self, data, target):
        for _ in range(self.tree_num):
            #ブートストラップを行って、データをサンプリングする
            #コードを埋める
            sample_data,sample_target = data[(indices := np.random.choice(len(data), self.sample_num, replace=True))], target[indices]
            #tree = DecisionTreeClassifier(max_depth=self.max_depth)
            #もしフルスクラッチを完成させていたら，コメントアウト
            tree = my_DecisionTreeClassifier(self.max_depth)
            tree.fit(sample_data,sample_target)
            self.trees.append(tree)
            self.datas.append([sample_data,sample_target])
            
    def predict(self, data):
        results=[]
        #各決定木について分類結果を求める
        for tree in self.trees:
            results.append(tree.predict(data))
        pred = []
        for counter in list(zip(*results)):
            #分類結果をもとに多数決を行って，その結果をpredに格納
            #例えば[0,0,1,1,0]であれば，多数決の結果により0を予測結果とする
            #コードを埋める
            pred_result = max(set(counter), key=list(counter).count)
            pred.append(pred_result)
        return np.array(pred)

moon = make_moons(n_samples=200, noise=0.1, random_state=0)
X = moon[0]
y = moon[1]

# 学習データと検証データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = my_RandomForestClassifier(tree_num=8,sample_num=len(X_train)//2,max_depth=2)
clf.fit(X_train, y_train)

#学習データに対する予測
y_pred = clf.predict(X_train)
score = sum(y_pred == y_train)/float(len(y_train))
print('Classification accuracy: {}'.format(score))

#テストデータに対する予測
y_pred = clf.predict(X_test)
score = sum(y_pred == y_test)/float(len(y_test))
print('Classification accuracy: {}'.format(score))

# 学習結果の可視化
def plot_decision_boundary(model, x, t):
    # サンプルデータのプロット
    plt.plot(x[:, 0][t==0], x[:, 1][t==0], 'bo')
    plt.plot(x[:, 0][t==1], x[:, 1][t==1], 'r^')
    plt.xlabel('x') # x 軸方向に x を表示
    plt.ylabel('y', rotation=0) # y 軸方向に y を表示
    
    # 描画範囲の設定
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
    
    # 用意した間隔を使用してグリッドを作成
    _x, _y = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
    
    # 多次元配列の結合
    xy = np.array([_x.ravel(), _y.ravel()]).T
    # 予測結果を算出し、分類境界線を図示
    y_pred = model.predict(xy).reshape(_x.shape)
    custom_cmap = ListedColormap(['mediumblue', 'orangered'])
    plt.contourf(_x, _y, y_pred, cmap=custom_cmap, alpha=0.2)
fig = plt.figure(figsize = (12,8), facecolor='lightblue')
row = len(clf.trees)//3+1
for n in range(len(clf.trees)):
    ax = fig.add_subplot(row, 3, n+1)
    plot_decision_boundary(clf.trees[n], clf.datas[n][0],clf.datas[n][1])
plt.figure(figsize=(12, 8))
plot_decision_boundary(clf, X_train, y_train)

# グラフの表示
plt.show()