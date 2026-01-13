"""
    Chapter 最終問題 49.      Kaggleのデータ分析コンテストに挑戦する.
                            ・提出はデータ分析に使用したコード(.py)と予測結果の出力(.txt)の二つを提出せよ．
                            ・予測精度の高さとコードの内容を総合的に評価する．
                            House Prices: Advanced Regression Techniquesのデータ分析に挑戦せよ．
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import norm
import warnings
import re

# 警告抑制とOptunaのログレベル設定
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# スタイルの設定
plt.style.use('ggplot')
sns.set_palette('pastel')

# --- 1. データ読み込み ---
print("Loading data...")
train = pd.read_csv("./house_prices_data/train.csv")
test = pd.read_csv("./house_prices_data/test.csv")

# --- 1.5 データの可視化 (EDA: Exploratory Data Analysis) ---
print("\n--- Starting EDA ---")
print("グラフを表示します。確認したらウィンドウを閉じてください。")

# (1) 目的変数 SalePrice の分布確認
# 正規分布に従っているかを確認する（QQプロットも併用）
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(train['SalePrice'], kde=True, ax=ax[0])
ax[0].set_title('SalePrice Distribution')
stats.probplot(train['SalePrice'], plot=ax[1])
ax[1].set_title('SalePrice QQ Plot')
plt.show()

# (2) 相関行列（ヒートマップ）
# SalePriceと相関が高い上位10個の特徴量を確認
corrmat = train.select_dtypes(include=[np.number]).corr() # 数値型のみで相関計算
k = 10 # 表示する変数の数
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(10, 8))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values, cmap='coolwarm')
plt.title('Top 10 Correlated Features with SalePrice')
plt.show()

# (3) 外れ値の確認 (GrLivArea vs SalePrice)
# 居住面積が広いのに価格が安い物件はノイズになる可能性がある
plt.figure(figsize=(10, 6))
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.title('GrLivArea vs SalePrice (Outlier Check)')
plt.show()

# (4) 欠損値の確認
# 欠損が多いカラムを可視化
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data_top20 = missing_data.head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=missing_data_top20.index, y=missing_data_top20['Percent'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Percent of Missing Values')
plt.title('Top 20 Features with Missing Values')
plt.show()


# --- 2. データ前処理 & 特徴量エンジニアリング ---
print("\nFeature Engineering...")

# 外れ値の処理 (Clipping & Rule-based)
# GrLivAreaが4000以上かつSalePriceが300000以下のデータは外れ値として除外（グラフで確認した点）
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# 目的変数の対数変換 (正規分布に近づける)
# 歪度を解消するため log(1+x) 変換を行う
train["SalePrice"] = np.log1p(train["SalePrice"])

# ★重要★ 外れ値除去で行が減ったため、インデックスをリセットして0から振り直す
y = train["SalePrice"].reset_index(drop=True)

# ID列の保存と削除
train_id = train["Id"]
test_id = test["Id"]
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)
train.drop("SalePrice", axis=1, inplace=True)

# データの結合 (前処理を一括で行うため)
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

# (1) 欠損値処理
# LightGBMは数値の欠損を扱えるが、カテゴリ変数の欠損は明示的に埋める必要がある場合がある
# ここではオブジェクト型の欠損を "None" という新しいカテゴリとして扱う
for col in all_data.select_dtypes(include='object').columns:
    all_data[col] = all_data[col].fillna("None")

# (2) 新たな特徴量の作成 (ドメイン知識に基づく)
# 総床面積
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# 築年数 + 改築年数 (建物の新しさの指標)
all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']
# バ​​スルーム総数
all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))
# 設備有無フラグ
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBasement'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# データを再分割
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]


# --- 3. Target Encoding with Smoothing (CV内実装) ---
# リーク（カンニング）を防ぐため、Cross Validationのループ内で、
# 「学習データから計算した統計量」を「検証データ」に適用する手法をとる。

def target_encoding(train_df, test_df, target, cat_cols, n_splits=10):
    """
    Target Encoding with Smoothing applied in an Out-of-Fold manner.
    """
    # 結果格納用データフレーム
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # KFold設定 (Shuffleしてランダム性を入れる)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # ターゲット名（Seriesに名前がない場合の対応）
    target_name = target.name if target.name else 'target'
    
    # カテゴリ列ごとに処理
    for col in cat_cols:
        # 全体の平均（Smoothing用）
        global_mean = target.mean()
        
        # --- テストデータへの適用 ---
        # テストデータには、学習データ全体を使って計算した値をマッピング
        temp_df = pd.DataFrame({col: train_df[col].values, target_name: target.values})
        agg = temp_df.groupby(col)[target_name].agg(['count', 'mean'])
        
        counts = agg['count']
        means = agg['mean']
        # Smoothing: データ数が少ないカテゴリは全体平均に近づける
        smooth = (counts * means + 10 * global_mean) / (counts + 10)
        test_encoded[col] = test_encoded[col].map(smooth).fillna(global_mean)
        
        # --- 学習データへの適用 (Out-of-Fold) ---
        oof_col = np.zeros(len(train_df))
        
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            y_tr = target.iloc[tr_idx]
            
            mean_tr = y_tr.mean()
            
            # Fold内での集計用データフレーム作成
            temp_df_tr = pd.DataFrame({col: X_tr[col].values, target_name: y_tr.values})
            agg_tr = temp_df_tr.groupby(col)[target_name].agg(['count', 'mean'])
            
            counts_tr = agg_tr['count']
            means_tr = agg_tr['mean']
            smooth_tr = (counts_tr * means_tr + 10 * mean_tr) / (counts_tr + 10)
            
            # 検証データ部分にマッピング
            oof_col[val_idx] = X_val[col].map(smooth_tr).fillna(mean_tr)
            
        train_encoded[col] = oof_col
        
    return train_encoded, test_encoded

# カテゴリ変数の抽出
cat_cols = X_train.select_dtypes(include='object').columns.tolist()

print(f"Performing Target Encoding on {len(cat_cols)} categorical features...")
X_train_te, X_test_te = target_encoding(X_train, X_test, y, cat_cols, n_splits=10)

# カラム名のリネーム（LightGBM対応）
X_train_te = X_train_te.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_test_te = X_test_te.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


# --- 4. Optunaによるハイパーパラメータ探索 ---
print("\nStarting Optuna Hyperparameter Tuning...")

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    dtrain = lgb.Dataset(X_train_te, label=y)
    
    cv_results = lgb.cv(
        params,
        dtrain,
        folds=kf,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        return_cvbooster=False
    )
    
    rmse_mean = cv_results['valid rmse-mean'][-1]
    return rmse_mean

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20) 

print(f"Best RMSE: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")


# --- 5. アンサンブルモデルの作成 (Seed Averaging) ---
best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt'
})

n_models = 6 
oof_preds_matrix = np.zeros((ntrain, n_models))
test_preds_matrix = np.zeros((ntest, n_models))

print(f"\nTraining {n_models} models with different seeds for Ensemble...")

kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)

for i in range(n_models):
    current_params = best_params.copy()
    current_params['random_state'] = 42 + i
    current_params['seed'] = 42 + i
    
    oof_preds = np.zeros(ntrain)
    test_preds_fold = [] 
    
    for tr_idx, val_idx in kf_outer.split(X_train_te, y):
        X_tr, X_val = X_train_te.iloc[tr_idx], X_train_te.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**current_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds_fold.append(model.predict(X_test_te))
    
    test_preds_avg = np.mean(test_preds_fold, axis=0)
    
    oof_preds_matrix[:, i] = oof_preds
    test_preds_matrix[:, i] = test_preds_avg
    
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"Model {i} (Seed {42+i}) CV RMSE: {rmse:.4f}")


# --- 6. アンサンブルの重み最適化 (Optuna) ---
print("\nOptimizing Ensemble Weights with Optuna...")

def weight_objective(trial):
    weights = []
    for i in range(n_models):
        weights.append(trial.suggest_float(f'w{i}', 0.0, 1.0))
    
    weights = np.array(weights)
    if weights.sum() == 0: return 100.0
    weights /= weights.sum()
    
    ensemble_pred = np.zeros(ntrain)
    for i in range(n_models):
        ensemble_pred += weights[i] * oof_preds_matrix[:, i]
        
    rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
    return rmse

weight_study = optuna.create_study(direction='minimize')
weight_study.optimize(weight_objective, n_trials=50)

best_weights = []
for i in range(n_models):
    best_weights.append(weight_study.best_params[f'w{i}'])
best_weights = np.array(best_weights)
best_weights /= best_weights.sum()

print(f"Best Ensemble RMSE: {weight_study.best_value:.4f}")
print(f"Optimized Weights: {best_weights}")


# --- 7. 最終提出ファイルの作成 ---
final_test_pred = np.zeros(ntest)
for i in range(n_models):
    final_test_pred += best_weights[i] * test_preds_matrix[:, i]

# 対数変換を元に戻す (np.expm1)
final_pred = np.expm1(final_test_pred)

sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = final_pred
sub.to_csv('./house_prices_data/submission.csv', index=False)

print("\nSubmission saved successfully!")