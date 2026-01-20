"""
    Chapter10 55.    それぞれの実験の結果をレポートにて報告せよ．
                    コードを変えることでどのような結果の変化が起こったか，またそれはなぜだと考えられるか考察せよ．
"""

import warnings
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

warnings.filterwarnings('ignore') # 実行に影響のない　warninig　を非表示にします. 非推奨.

# PyTorchのバージョン確認
print(torch.__version__)

### データの読み込みと前処理 ###
#データの前処理を行うクラスインスタンス
transform = transforms.Compose(
    [transforms.Resize((16, 16)),
     transforms.ToTensor(),
     #transforms.Normalize((0.5, ), (0.5, ))
    ])

batch_size = 100

#使用するtrainデータセット
trainset = torchvision.datasets.MNIST(root='../text/data', 
                                        train=True,
                                        download=True,
                                        transform=transform)
#データ分割
#trainset, _ = torch.utils.data.random_split(trainset, [10000, len(trainset)-10000])
print(len(trainset))

#trainデータをbatchごとに逐次的に取り出してくれるクラスインスタンス
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True)

### VAEモデル構造 ###
class VAE(torch.nn.Module):
    def __init__(self, z_dim, n_input=256):
        super(VAE, self).__init__()
        self.n_input = n_input
        self.dense_enc1 = torch.nn.Linear(self.n_input, 128)
        self.dense_enc2 = torch.nn.Linear(128, 64)
        self.dense_encmean = torch.nn.Linear(64, z_dim)
        self.dense_encvar = torch.nn.Linear(64, z_dim)
        self.dense_dec1 = torch.nn.Linear(z_dim, 64)
        self.dense_dec2 = torch.nn.Linear(64, 128)
        self.dense_dec3 = torch.nn.Linear(128, self.n_input)
    
    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = F.sigmoid(self.dense_encmean(x))
        #mean = self.dense_encmean(x)
        #この後の確認作業上sigmoidに通しておくと便利なだけで絶対必要ではない
        var = F.softplus(self.dense_encvar(x))
        return mean, var
    
    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape)
        return mean + torch.sqrt(var) * epsilon

    def _decoder(self, z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        x = F.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x):
        x = x.view(-1, self.n_input)
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, z
    
    def loss(self, x):
        x = x.view(-1, self.n_input)
        mean, var = self._encoder(x)
        KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
        z = self._sample_z(mean, var)
        y = self._decoder(z)
        reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
        lower_bound = [-KL, reconstruction]
        return -sum(lower_bound)

### 学習の実行 ###
hidden_size = 20
model = VAE(hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#最適化手法の設定
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  #【実験】
model.train()
for i in range(20):
    losses = []
    for x, t in trainloader:
        model.zero_grad()
        y = model(x)
        loss = model.loss(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
    print("EPOCH: {} loss: {}".format(i, np.average(losses)))

### 生成画像の表示 ###
fig = plt.figure(figsize=(10, 3))

model.eval()
zs = []
for x, t in trainloader:
    # original
    for i, im in enumerate(x.view(-1, 16, 16).detach().numpy()[:10]):
        ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
        ax.imshow(im, 'gray')
    # generate from x
    y, z = model(x)
    zs.append(z)
    y = y.view(-1, 16, 16)
    for i, im in enumerate(y.cpu().detach().numpy()[:10]):
        ax = fig.add_subplot(2, 10, i+11, xticks=[], yticks=[])
        ax.imshow(im, 'gray')
    break
plt.show()

# generate from z
fig = plt.figure(figsize=(10, 10))
for i in range(10):
    array_i = i*0.1*torch.ones(10, 1)
    z1to0_ = torch.cat([torch.arange(0, 1, 0.1).view(10, 1), array_i], dim=1)
    z1to0 = torch.clone(z1to0_)
    if hidden_size>2:
        for j in range(hidden_size//2-1):
            z1to0 = torch.cat([z1to0_, z1to0], dim=1)
    y2 = model._decoder(z1to0).view(-1, 16, 16)
    for j, im in enumerate(y2.cpu().detach().numpy()):
        ax = fig.add_subplot(10, 10, j+10*i+1, xticks=[], yticks=[])
        ax.imshow(im, 'gray')
plt.show()