"""
    Chapter6 51.    それぞれの実験の結果をレポートにて報告せよ．
                    コードを変えることでどのような結果の変化が起こったか，またそれはなぜだと考えられるか考察せよ．
                    （例）多項式回帰にしてみる, 標準化せずにやってみる, bias=Falseにしてみる
                        epoch数, batch_size, loss関数, optimizer（の種類, またはパラメータ）を変更してみる
                        chapter3で紹介した分類問題をpytorchで解いてみる
                        など
"""

import warnings
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib import cm

warnings.filterwarnings('ignore') # 実行に影響のない　warninig　を非表示にします. 非推奨.

# PyTorchのバージョン確認
print(torch.__version__)

### データの読み込み ###
wine = pd.read_csv("../text/data/winequality-red.csv", sep=";") # sepは区切り文字の指定
print(wine.head(5))

### データの準備 ###
np.random.seed(0) # random　の挙動を固定

X = wine[["density"]].values
T = wine["alcohol"].values

# 平均0にする前処理部分
X = X - X.mean() # 【実験】標準化の有無
T = T - T.mean() # 【実験】標準化の有無

# X, Tそれぞれが最小値0, 最大値1になるように標準化
# 追加の前処理部分
X = (X - X.min()) / (X.max() - X.min()) # 【実験】標準化の有無
T = (T - T.min()) / (T.max() - T.min()) # 【実験】標準化の有無

# データのシャッフルと分割
p = np.random.permutation(len(X))
X = X[p]
T = T[p]

X_train = X[:1000, :]
T_train = T[:1000]
X_test = X[1000:, :]
T_test = T[1000:]

### データの可視化 ###
fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

axes[0].scatter(X_train, T_train, marker=".")
axes[0].set_title("train")
axes[1].scatter(X_test, T_test, marker=".")
axes[1].set_title("test")
fig.show()
plt.show()

#### モデル構造 ###
class MyLinearRegression(torch.nn.Module):
    
    #初期化
    def __init__(self, n_input, n_output):#入力の次元と出力の次元を与える
        super().__init__()
        self.l1 = torch.nn.Linear(n_input, n_output, bias = True) # 【実験】bias = False　と変更
        """
        torch.nn.Linear
            (第一引数の入力次元数*第二引数に出力次元数の大きさの行列を定義することができる
            特に指定をしない限り学習過程で内部のパラメータは変化していく（初期値はランダム）
        """
        
    #sklearnで言うところの predict関数に相当
    def forward(self, x):
        h1 = self.l1(x)
        """
        self.l1(x)
            __init__で作った行列を使って入力xを予測値に変換する
        """
        return h1

### モデルの作成 ###
model = MyLinearRegression(1, 1)

# loss関数の設定
criterion = torch.nn.MSELoss()
#criterion = torch.nn.L1Loss()

# 最適化手法の設定
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

### データセットの作成 ###
class Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        if train:
            self.X = torch.FloatTensor(X_train)
            self.t = torch.FloatTensor(T_train).view(-1, 1)
        else:
            self.X = torch.FloatTensor(X_test)
            self.t = torch.FloatTensor(T_test).view(-1, 1)
        self.data_length = len(self.t)

    def __getitem__(self, index):
        return self.X[index], self.t[index]

    def __len__(self):
        return self.data_length

### 学習の実行 ###
batch_size = 1000
trainset = Dataset(train=True)
testset  = Dataset(train=False)
trainloader = DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)
testloader  = DataLoader(dataset = testset,  batch_size = batch_size, shuffle = True)

epochs = 1000
loss_list = []
param_list = []
num_batchs = len(trainloader) // batch_size + 1
for epoch in range(epochs):
    loss_sum = 0
    for X, t in trainloader:
        t_pred = model(X)
        loss = criterion(t_pred, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach()
    loss_sum /= num_batchs
    loss_list.append(loss_sum)
    st_dict = model.state_dict()
    param_list.append([st_dict['l1.weight'].detach().numpy()[0, 0], st_dict['l1.bias'].detach().numpy()[0], epoch])
    # 【実験】bias = False　と変更
    #param_list.append([st_dict['l1.weight'].detach().numpy()[0, 0], epoch])
    print(f'\repoch: {epoch+1}/{epochs}, train_loss: {loss_sum}', end='')
param_list = np.array(param_list)

### 学習結果の可視化 ###
plt.plot(loss_list)
plt.xlabel('Training Iteration') 
plt.ylabel('Loss') 
plt.show()

# 学習におけるパラメータの推移を可視化
plt.scatter(param_list[:, 0], param_list[:, 1],
            c=param_list[:, 2], # 【実験】bias = False　と変更
            cmap=cm.jet,#カラーマップの種類
            marker='.',lw=0)
plt.xlabel('係数') 
plt.ylabel('Bias') 
ax=plt.colorbar()#カラーマップの凡例
ax.set_label('time [epoch]')#カラーバーのラベルネーム
plt.show()

### テストデータでの評価 ###
model.eval()
num_batchs = len(testloader) // batch_size + 1
with torch.no_grad():
    loss_sum = 0
    for X, t in testloader:
        t_pred = model(X)
        loss_sum += criterion(t_pred, t).detach()
        
    loss_sum /= num_batchs
    
print('test_loss: ', loss_sum.detach().numpy())

### 予測結果の可視化 ###
predict = model(torch.FloatTensor(X_train)).detach().numpy()

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

axes[0].scatter(X_train, T_train, marker=".")
axes[0].plot(X_train, predict, color="red")
axes[0].set_title("train")

axes[1].scatter(X_test, T_test, marker=".")
axes[1].plot(X_train, predict, color="red")
axes[1].set_title("test")
fig.show()
plt.show()

### 学習後のパラメータの表示 ###
st_dict = model.state_dict()
print('係数: ', st_dict['l1.weight'].detach().numpy()[0, 0])
print('bias: ', st_dict['l1.bias'].detach().numpy()[0])
# 【実験】bias = False　と変更
#print('bias: ', 0) # 【実験】bias = False　のため常に0