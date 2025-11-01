"""
Chapter0 28.    27で作った関数の実行速度を計測し, それよりも高速な手法を検討せよ．
                実行速度を求めるために，データサイズを工夫した入力例を作るとよい．
                また, 1回の実行速度ではなく10回以上の実行の平均速度を使う.
                実行速度の計測にはtimeモジュールを利用するなどが考えられる.
"""

import numpy as np
import time

# --- 関数定義 ---

def compute_distances_naive(points):
    """
    Ex27 の愚直な実装（n*n回すべて計算）
    """
    n = len(points)
    euclidean_distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            distance = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            euclidean_distance_matrix[i][j] = distance
            
    return euclidean_distance_matrix

def compute_distances_optimized(points):
    """
    Ex28 対称性を利用し，計算量を半分にした実装．(j > i のペアのみ計算)
    """
    n = len(points)
    # n x n の行列を0で初期化．対角成分 D[i, i] は 0 のままでよい．
    euclidean_distance_matrix = np.zeros((n, n))
    
    # 外側のループ (i) は 0 から n-2 まででよい
    for i in range(n):
        # 内側のループ (j) は i+1 から n-1 まで
        # これで (j > i) のペアだけが網羅される
        for j in range(i + 1, n):
            
            # 距離を1回だけ計算
            distance = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            
            # 対称性を利用し，2箇所に同時に代入
            euclidean_distance_matrix[i][j] = distance
            euclidean_distance_matrix[j, i] = distance # (j, i) にも同じ値を入れる
            
    return euclidean_distance_matrix

# --- ベンチマーク（実行速度計測）部分 ---

# 1. ベンチマーク設定
N_POINTS = 500      # 点の数 (データサイズ: この値を大きくすると計測しやすい)
REPEAT_COUNT = 10   # 試行回数

print(f"ベンチマーク開始...")
print(f"データサイズ (点の数): {N_POINTS}")
print(f"試行回数: {REPEAT_COUNT} 回")

# 2. テストデータの自動生成
# 乱数のシードを固定し，毎回同じデータで計測できるようにする
np.random.seed(42) 
# (N_POINTS, 2) の形状を持つNumpy配列 (0から100のランダムな座標)
points_arr = np.random.rand(N_POINTS, 2) * 100
# 関数がリスト入力を想定しているためリストに変換
points_list = points_arr.tolist()

# 3. 実行時間の計測 (愚直版)
print("\n[1] Ex27 愚直な実装 (n*n) の速度を計測中...")
execution_times_naive = [] # 各回の実行時間を格納するリスト
for _ in range(REPEAT_COUNT):
    start_time = time.perf_counter() # 高精度のタイマーを開始
    
    # 計測対象の関数を実行
    compute_distances_naive(points_list)
    
    end_time = time.perf_counter()   # タイマーを終了
    execution_times_naive.append(end_time - start_time) # 実行時間をリストに追加

# 4. 実行時間の計測 (最適化版)
print("[2] Ex28 対称性を利用した実装 (n*n / 2) の速度を計測中...")
execution_times_optimized = [] # 各回の実行時間を格納するリスト
for _ in range(REPEAT_COUNT):
    start_time = time.perf_counter() # 高精度のタイマーを開始
    
    # 計測対象の関数を実行
    compute_distances_optimized(points_list)
    
    end_time = time.perf_counter()   # タイマーを終了
    execution_times_optimized.append(end_time - start_time) # 実行時間をリストに追加

# 5. 平均実行時間の計算と出力
avg_time_naive = sum(execution_times_naive) / REPEAT_COUNT
avg_time_optimized = sum(execution_times_optimized) / REPEAT_COUNT

print("\n--- ベンチマーク結果 ---")
print(f"[1] Ex27 愚直な実装 (n*n)      : {avg_time_naive:.6f} 秒")
print(f"[2] Ex28 対称性を利用した実装 : {avg_time_optimized:.6f} 秒")