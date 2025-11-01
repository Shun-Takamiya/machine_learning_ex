"""
Chapter0 27.    標準入力で, 点数を表す数値nと各点の座標(x, y)を受け取り，それらの間の距離を全て求め，行列表記で返せ．
                行列の(i,j)要素はi番目とj番目の点のユークリッド距離とする.
                ユークリッド距離は小数点以下4桁まで出力
                すべての要素について愚直に計算するようにせよ．(28との測度比較のため)
                入力される座標の各要素は実数である．
                例題
                    入力: 
                        3
                        1 2
                        1 3
                        2 3
                    出力: 
                        0.0000 1.0000 1.4142
                        1.0000 0.0000 1.0000
                        1.4142 1.0000 0.0000
"""

import numpy as np

def compute_distances(points):
    n = len(points)
    euclidean_distance_matrix = np.zeros((n, n))
    
    i = 0
    j = 0
    for i in range(n):
        for j in range(n):
            distance = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            euclidean_distance_matrix[i][j] = distance
            
    return euclidean_distance_matrix

n = int(input("点の数を入力してください: "))
points = []
i = 0
for i in range(n):
    x, y = map(float, input(f"{i+1} 番目の点の座標をスペース区切りで入力してください: ").split())
    points.append((x, y))

distance_matrix = compute_distances(points)

for row in distance_matrix:
    print(" ".join(f"{dist:.4f}" for dist in row))