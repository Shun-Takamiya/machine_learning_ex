"""
    Chapter3 41.    ニュートン法の実装に取り組み，コードを実装せよ．
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#x^2の関数
def f(x):
    #埋める
    return x**2
#x^2の1階微分
def diff_f(x):
    #埋める 
    return 2*x
#x^2の2階微分
def diff2_f(x):
    #埋める
    return 2

t=np.linspace(-5,5,1000)
fig, ax = plt.subplots()
ax.plot(t,f(t), color='blue') # グラフを作成
ims = []

x=5 #xの初期値
epsilon=0.001 #ループ回数
eta=0.1 #学習率

#最急降下法
i=0

#ニュートン法
x2=x
i2=0
while abs(diff_f(x)*eta)>epsilon or abs(diff_f(x2)/diff2_f(x2))>epsilon:
    
    img = ax.plot(x,f(x),'*',markersize=10, color='red')
    img2=ax.text(-0.5, 27,f"n={i}")
    img3 = ax.plot(x2,f(x2),'*',markersize=10, color='green')
    ims.append(img+[img2]+img3) # グラフを配列に追加
    
    #勾配降下法
    if abs(diff_f(x)*eta)>epsilon:
        x=x-eta*diff_f(x)
        i+=1
    
    #ニュートン法
    if abs(diff_f(x2)/diff2_f(x2))>epsilon:
        #更新式を埋める
        x2=x2 - diff_f(x2)/diff2_f(x2)
        i2+=1
    
print("[最急降下法]")
print("収束までの繰り返し回数:",i)
print("極小値:x=",x)
print("[ニュートン法]")
print("収束までの繰り返し回数:",i2)
print("極小値:x=",x2)
ani = animation.ArtistAnimation(fig, ims, interval=100)
#ani.save('x^2.gif', writer='imagemagick')
plt.plot()

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました．")