"""
    Chapter4 43.    ジニ係数を求める関数を作成せよ．
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

#ジニ係数の計算を行う関数
def gini_score(data, target, feat_idx, threshold):
    
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

#データを分割する基準（数値を変更してみるとよい）
feat_idx=0
threshold=3.1

#関数の実行
print(gini_score(X, y, feat_idx, threshold))

#データ分割を図示する
if feat_idx==0:
    plt.vlines(x=threshold,ymin=int(min(X[:, 0]))-1,ymax=int(min(X[:, 0]))+1)
else:
    plt.hlines(y=threshold,xmin=int(min(X[:, 1]))-1,xmax=int(min(X[:, 1]))+1)
plt.scatter(X[:, 0][y==0], X[:, 1][y==0], color='lightskyblue', label=data.target_names[0])
plt.scatter(X[:, 0][y==1], X[:, 1][y==1], color='sandybrown', label=data.target_names[1])

#グラフの表示
plt.xlabel(data.feature_names[2])
plt.ylabel(data.feature_names[3])
plt.legend()
plt.show()