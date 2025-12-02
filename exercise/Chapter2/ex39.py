"""
    Chapter2 39.    (任意の)回帰モデルをscikit-learn準拠でコーディングし, 実験せよ.
                    まずはMyLinearRegression()クラスを完成させて，単回帰モデルを実装してみよ.
                    重回帰モデルなどの他の回帰モデルを実装してもよい．
                    データはX_train.csvとy_train.csvを用いよ.
                    評価はX_test.csvで行う. 予測結果を../text/data/y_pred.csvとして保存せよ.
                    提出は完成したMyLinearRegression()クラスのみで構わない．
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- scikit-learn準拠の単回帰モデルクラス ---

class MyLinearRegression(object):
    def __init__(self):
        """
        Initialize a coefficient and an intercept.
        """
        # 係数（傾き）a と 切片 b を初期化
        self.a = None
        self.b = None
        
    def fit(self, X, y):
        """
        X: data, array-like, shape (n_samples, 1)
        y: array, shape (n_samples,)
        Estimate a coefficient and an intercept from data.
        
        単回帰の最小二乗法の公式（解析解）を用いて係数を計算する．
        """
        
        # scikit-learn準拠のためXは2次元配列(n_samples, 1)で渡されると想定
        # 計算のため1次元配列(n_samples,)に変換
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()
        
        # yも1次元配列(n_samples,)であることを確認
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()
            
        # 最小二乗法の公式
        # a = Cov(x, y) / Var(x)
        # b = mean(y) - a * mean(x)
        
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # 傾き a の計算
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean)**2)
        
        # 0除算を回避
        if denominator == 0:
            self.a = 0
        else:
            self.a = numerator / denominator
            
        # 切片 b の計算
        self.b = y_mean - (self.a * x_mean)
        
        return self
    
    def predict(self, X):
        """
        Calc y from X
        y = aX + b
        """
        if self.a is None or self.b is None:
            raise ValueError("fit()メソッドを先に呼び出してモデルを学習させてください．")
            
        # Xが2次元配列(n_samples, 1)の場合，1次元に変換
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()
            
        # y = aX + b を計算して返す
        return self.a * X + self.b

# --- 実験の実行 ---

# def create_dummy_data(): ... (この関数全体を削除)

def run_experiment():
    """
    データ読み込み，学習，予測，保存，描画を実行する．
    """
    
    # --- 1. データの準備 ---
    
    # データ読み込みパス
    data_dir = "../text/data/"
    
    # CSVファイルからデータを読み込み
    try:
        X_train_df = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
        y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
        X_test_df = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    except FileNotFoundError:
        print(f"エラー: データファイルが '{data_dir}' に見つかりません．")
        return

    # --- ★★★ エラー修正 (データクリーニング) ★★★ ---
    # 1. 訓練データを数値に変換 (数値以外はNaNにする)
    
    # 単回帰モデルのため、CSVの *最初の列* (.iloc[:, 0]) のみを使用する
    # pd.to_numeric で数値に変換 (文字列などは NaN になる)
    X_train_series = pd.to_numeric(X_train_df.iloc[:, 0], errors='coerce')
    
    # y_trainも同様に最初の列のみを使用
    y_train_series = pd.to_numeric(y_train_df.iloc[:, 0], errors='coerce')
    
    # 2. X_train と y_train を横に結合し，NaNを含む行をまとめて削除
    #    (XかyのどちらかがNaNなら，そのペアは学習に使えないため)
    train_df_combined = pd.concat([X_train_series, y_train_series], axis=1)
    # わかりやすいように列名を変更
    train_df_combined.columns = ['X_train_col', 'y_train_col']
    train_df_combined.dropna(inplace=True)
    
    # 3. テストデータも同様にクリーニング (最初の列のみ使用)
    X_test_series = pd.to_numeric(X_test_df.iloc[:, 0], errors='coerce')
    X_test_series.dropna(inplace=True) # NaNになった行を削除
    # -----------------------------------------------

    # Pandas DataFrame から Numpy 配列に変換
    # 結合したDataFrameから再度分離
    # .values でNumpy配列にし, .reshape(-1, 1) で (n_samples, 1) の2D配列に整形
    X_train = train_df_combined['X_train_col'].values.reshape(-1, 1)
    y_train = train_df_combined['y_train_col'].values  # yは (n_samples,) の1D配列
    
    X_test = X_test_series.values.reshape(-1, 1)   # (n_samples, 1) の2D配列

    # --- 2. 学習 ---
    clf = MyLinearRegression()
    clf.fit(X_train, y_train)

    # 回帰係数と切片の表示
    print("\n--- 学習結果 ---")
    print(f"係数 (a): {clf.a:.4f}")
    print(f"切片 (b): {clf.b:.4f}")

    # --- 3. 予測と保存 ---
    y_pred = clf.predict(X_test)
    
    # 予測結果をDataFrameに変換
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted'])
    
    # 保存先ディレクトリの準備
    output_dir = "../text/data/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "y_pred.csv")
    
    # CSVファイルとして保存
    y_pred_df.to_csv(output_path, index=False)
    print(f"\n予測結果を {output_path} に保存しました．")

    # --- 4. グラフ描画 ---
    print("\nグラフを表示します...")
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

    # 学習データ
    axes[0].scatter(X_train, y_train, marker=".", label="Actual Data")
    axes[0].plot(X_train, clf.predict(X_train), color="red", label="Regression Line")
    axes[0].set_title("Train Data")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Target")
    axes[0].legend()
    axes[0].grid(True)

    # テストデータ
    axes[1].plot(X_test, clf.predict(X_test), color="red", label="Regression Line (Prediction)")
    axes[1].set_title("Test Data (Prediction Only)")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Target")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# このスクリプトが直接実行された場合のみ，実験コードを実行
if __name__ == "__main__":
    run_experiment()