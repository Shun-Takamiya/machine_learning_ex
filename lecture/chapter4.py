import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
# ペンギンデータセットの読み込み
penguin = load_penguins()

# 使用する説明変数の指定
usecols = [
    'species',
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
    'year'
]
df1 = penguin[usecols].copy()
print(df1.head())

# 欠損値のあるデータを削除
df2=df1.dropna()
print(df2.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 目的変数を数値に変換
species_le = LabelEncoder()
df3=df2.copy()
df3['species'] = species_le.fit_transform(df2['species'])

# 目的変数
y=df3["species"]

# 説明変数
X=df3.drop("species", axis=1)

# 学習データとテストデータを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)


from sklearn.ensemble import RandomForestClassifier

# 弱学習機（決定木）の個数を100個, 木の最大の深さを3に設定
model = RandomForestClassifier(n_estimators=100,max_depth=3)

# 学習
model.fit(X_train, y_train)


from sklearn import metrics
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


import lightgbm as lgb

params = {
    "objective": "multiclass",
    "metrics": "multi_error",
    "max_depth":3,
    "num_class": 3,
}

train_set = lgb.Dataset(X_train, y_train)
model2 = lgb.train(params, train_set)

y_pred = model2.predict(X_test)


# テストデータで予測
y_pred = model2.predict(X_test, num_iteration=model2.best_iteration)
y_pred_max = np.argmax(y_pred, axis=1)

# Accuracy の計算
accuracy = sum(y_test == y_pred_max) / len(y_test)
print('accuracy:', accuracy)