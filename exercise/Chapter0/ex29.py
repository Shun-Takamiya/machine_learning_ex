"""
Chapter0 29.    πを求める関数calculate_piを実装せよ.
                math.piなど円周率をすでに計算された値を使ってはいけない.
                例えば，モンテカルロ法を用いてシミュレーション的に求める方法がある．
                a. 正方形の中に内接するような円を描く．
                b. 正方形の内部にランダムに$n$個の点を打つ．
                c. n点の中で円の内部に含まれる数を数えて, これを$k$個とする．
                d. π ≈ 4k / n が成り立つ.
"""

import random
def calculate_pi(n_points):
    n_inside_circle = 0
    
    for i in range(n_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        if x**2 + y**2 <= 1:
            n_inside_circle += 1
            
    pi_estimate = 4 * n_inside_circle / n_points
    return pi_estimate

n = int(input("モンテカルロ法で使用する点の数を入力してください: "))

pi_value = calculate_pi(n)
print(f"推定されたπの値: {pi_value}")