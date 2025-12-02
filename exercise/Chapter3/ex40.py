"""
    Chapter3 40.    最急降下法の実装に取り組み，コードを実装せよ．
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#x^4-20x^2+30x+20の関数
def f(x):
    #埋める
    return x**4 - 20*x**2 + 30*x + 20

#x^4-20x^2+30x+20の1階微分
def diff_f(x):
    #埋める
    return 4*x**3 - 40*x + 30

t=np.linspace(-5,5,1000)
fig, ax = plt.subplots()
ax.plot(t,f(t), color='blue') # グラフを作成
ims = []

x=0 #xの初期値
epsilon=0.001 #収束判定
eta=0.01 #学習率

i=0
while abs(diff_f(x)*eta)>epsilon:
    
    img = ax.plot(x,f(x),'*',markersize=10, color='red')
    img2=ax.text(-0.5, 320,f"n={i}")
    ims.append(img+[img2]) # グラフを配列に追加
    
    #ここに更新式を書く
    #埋める
    x=x - eta*diff_f(x)
    i+=1

print("収束までの繰り返し回数:",i)
print("極小値:x=",x)
ani = animation.ArtistAnimation(fig, ims, interval=100)
#ani.save('x^4.gif', writer='imagemagick')
plt.plot()

print("グラフを表示します...")
plt.show()
print("グラフウィンドウを閉じました．")