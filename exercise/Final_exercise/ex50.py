"""
    Chapter 最終問題 50.      Kaggleのデータ分析コンテストに挑戦する.
                            ・提出はデータ分析に使用したコード(.py)と予測結果の出力(.txt)の二つを提出せよ．
                            ・予測精度の高さとコードの内容を総合的に評価する．
                            Titanic データセットのデータ分析に挑戦せよ．
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# 交差検証用のライブラリを追加
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. データの読み込み ---
train_data = pd.read_csv("./titanic_data/train.csv")
test_data = pd.read_csv("./titanic_data/test.csv")

# --- データの可視化 ---
print("\n--- データの基本情報 ---\n")
print(train_data.info())
print("\n--- 欠損値の数 ---\n")
print(train_data.isnull().sum())

# グラフのスタイル設定
plt.style.use('ggplot')
sns.set_palette('pastel')

# (1) カテゴリ変数と生存数の関係
# 性別，客室等級（Pclass），乗船港（Embarked）ごとに生存率に違いがあるかを確認．
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(x='Sex', hue='Survived', data=train_data, ax=axes[0])
axes[0].set_title('Survival by Sex')
sns.countplot(x='Pclass', hue='Survived', data=train_data, ax=axes[1])
axes[1].set_title('Survival by Pclass')
sns.countplot(x='Embarked', hue='Survived', data=train_data, ax=axes[2])
axes[2].set_title('Survival by Embarked')
plt.tight_layout()
plt.show()

# (2) 家族関係の変数と生存数の関係
# 兄弟・配偶者（SibSp）や親・子（Parch）の数によって生存率が変わるか確認．
# 例えば「一人旅」か「大家族」かで，避難のしやすさが違う可能性がある．
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x='SibSp', hue='Survived', data=train_data, ax=axes[0])
axes[0].set_title('Survival by SibSp')
sns.countplot(x='Parch', hue='Survived', data=train_data, ax=axes[1])
axes[1].set_title('Survival by Parch')
plt.tight_layout()
plt.show()

# (3) 数値変数と生存の関係
# 年齢や運賃の分布を確認．
# 「子供の生存率が高い」「運賃が高い（富裕層）方が生存率が高い」などの傾向を見る．
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(data=train_data, x='Age', hue='Survived', kde=True, element="step", ax=axes[0])
axes[0].set_title('Age Distribution')
sns.histplot(data=train_data, x='Fare', hue='Survived', kde=True, element="step", ax=axes[1])
axes[1].set_title('Fare Distribution')
axes[1].set_xlim(0, 300) # 外れ値で見にくくなるのを防ぐため表示範囲を制限．
plt.tight_layout()
plt.show()

# (4) 相関行列
# 各変数同士の相関関係を確認する．強い相関がある変数は，多重共線性（Multicollinearity）の原因になる可能性がある．
plt.figure(figsize=(10, 8))
numeric_data = train_data.select_dtypes(include=[np.number]) 
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# --- 前処理の開始 ---
print("\n--- 前処理と学習を開始します ---\n")

# 学習データとテストデータを結合して，一括で前処理を行う．
# 別々に処理すると，カテゴリの基準などがズレるリスクがあるため．
train_data['is_train'] = 1
test_data['is_train'] = 0
test_data['Survived'] = np.nan # テストデータの目的変数は空にしておく
all_data = pd.concat([train_data, test_data], ignore_index=True)


# --- 2. 特徴量エンジニアリング（精度の要） ---

# ---------------------------------------------------------
# 1. 欠損値の補完
# ---------------------------------------------------------

# Age（年齢）の補完:
# 単純な平均ではなく，「性別」と「客室等級（Pclass）」ごとの中央値を使って補完．
# 理由：男性より女性，下級クラスより上級クラスの方が年齢層が高いなどの傾向を反映させ，より正確な値を推定するため．
all_data['Age'] = all_data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

# Embarked（乗船港）の補完:
# 欠損はごくわずかであるため，最も頻出する値（最頻値）である 'S' で補完する．
all_data['Embarked'] = all_data['Embarked'].fillna('S')

# Fare（運賃）の補完:
# テストデータに1件だけ欠損がある．
# 理由：運賃は客室等級（Pclass）に大きく依存するため，同一Pclassの中央値で補完するのが妥当である．
all_data['Fare'] = all_data.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.median()))

# Cabin（客室番号）の補完:
# 欠損が非常に多いが，これは「客室情報が記録されていない（下級客室など）」こと自体に意味がある可能性がある．
# そのため，欠損を示す値として 'N' を代入し，一つのカテゴリとして扱う．
all_data['Cabin'] = all_data['Cabin'].fillna('N')


# ---------------------------------------------------------
# 2. 新しい特徴量の生成
# ---------------------------------------------------------

# (1) NameからTitle（敬称）とSurname（姓）を抽出
# 名前そのものよりも，「Mr」「Mrs」などの敬称が年齢や社会的地位，性別を強く表す．
# 生存率に直結する重要な情報．
all_data['Title'] = all_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
all_data['Surname'] = all_data['Name'].str.extract(r'([A-Za-z]+),', expand=False)

# Titleのマッピング（整理）:
# 既婚女性（Mrs）は救助優先度が高い可能性があるため，まずは既婚フラグを作成．
all_data['IsMarried'] = 0
all_data.loc[all_data['Title'] == 'Mrs', 'IsMarried'] = 1

# 稀な敬称（Dr, Revなど）を 'Other' にまとめ，表記ゆれ（Mlleなど）を修正．
# 男性（Mr）と男児（Master）は生存率が大きく異なるため，区別して残す．
# 女性の敬称（Miss, Mrsなど）は，既婚フラグを作った上で 'Ms' に統一し，次元数を削減．
title_map = {
    'Mr': 'Mr',
    'Miss': 'Ms', 'Mrs': 'Ms', 'Mlle': 'Ms', 'Ms': 'Ms', 'Mme': 'Ms',
    'Master': 'Master',
    'Don': 'Other', 'Rev': 'Other', 'Dr': 'Other', 'Mme': 'Other', 'Ms': 'Other',
    'Major': 'Other', 'Lady': 'Other', 'Sir': 'Other', 'Mlle': 'Other', 'Col': 'Other',
    'Capt': 'Other', 'Countess': 'Other', 'Jonkheer': 'Other', 'Dona': 'Other'
}
all_data['Title'] = all_data['Title'].map(title_map).fillna('Other')


# (2) FamilySize（同乗した家族の人数）
# 兄弟・配偶者(SibSp) + 親・子(Parch) + 本人(1) で計算．
# 理由：大家族は避難が遅れる，あるいは単身者は後回しにされるなど，家族構成が生存に影響するため．
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1


# (3) TicketCount（同一チケットを持つ人数）
# 同一のチケット番号を持つ乗客は，家族や友人のグループである可能性が高い．
# 理由：FamilySizeだけでは分からない，チケットを共有するグループの規模を特徴量にするため．
ticket_count = all_data['Ticket'].value_counts()
all_data['TicketCount'] = all_data['Ticket'].map(ticket_count)


# (4) CabinGroup（客室エリアのグルーピング）
# 客室番号の先頭文字は船内のデッキ（階層）を表しており，避難経路や等級と関係がある．
all_data['CabinInit'] = all_data['Cabin'].str[0]
# 理由：個別のデッキごとにすると細かすぎるため，Pclassとの相関や位置関係を踏まえて
# 'ABC'（上層・1等），'DE'（中層），'FGT'（下層），'N'（不明）の4つに大別する．
all_data['CabinGroup'] = 'N'
all_data.loc[all_data['CabinInit'].isin(['A', 'B', 'C']), 'CabinGroup'] = 'ABC'
all_data.loc[all_data['CabinInit'].isin(['D', 'E']), 'CabinGroup'] = 'DE'
all_data.loc[all_data['CabinInit'].isin(['F', 'G', 'T']), 'CabinGroup'] = 'FGT'


# (5) FareBin（運賃の区分化）
# 運賃には外れ値（極端に高い値）が含まれるため，そのまま使うとモデルが不安定になることがある．
# 理由：Quantile（分位数）を使って10等分し，数値の大きさではなく「どの価格帯か」というカテゴリ変数にする（Label Encoding）．
all_data['FareBin'] = pd.qcut(all_data['Fare'], 10, labels=False)


# (6) AgeBin（年齢の区分化）
# 年齢と死亡率は単純な線形関係ではない（子供と老人が助かりやすいなど）．
# 理由：10歳刻みなどのビン（区間）に分けることで，特定の年齢層の傾向をモデルが捉えやすくする．
all_data['AgeBin'] = pd.cut(all_data['Age'].astype(int), 10, labels=False)


# (7) Family/Group Survival Rate（グループの生存率）
# タイタニック号の事故では，「家族が助かれば自分も助かる」「家族が亡くなれば自分も亡くなる」という強い相関がある．
# 理由：姓（Surname）やチケット番号が同じグループを探し，自分以外のメンバーの生存情報を特徴量として利用する．
# これにより，精度の大幅な向上が期待できる．
all_data['FamilySurvival'] = 0.5 # 初期値は0.5（情報なし）とする

# 姓と運賃が同じなら家族とみなす
for grp, grp_df in all_data.groupby(['Surname', 'Fare']):
    if len(grp_df) > 1:
        # グループ内に生存情報があるか確認（学習データに含まれるメンバーの情報を使う）
        if grp_df['Survived'].notnull().sum() > 0:
            smax = grp_df['Survived'].max()
            smin = grp_df['Survived'].min()
            pass_ids = grp_df['PassengerId']
            
            # 誰か一人でも助かっていれば '1'，全員亡くなっていれば '0' を設定
            if smax == 1.0:
                all_data.loc[all_data['PassengerId'].isin(pass_ids), 'FamilySurvival'] = 1
            elif smin == 0.0:
                all_data.loc[all_data['PassengerId'].isin(pass_ids), 'FamilySurvival'] = 0

# チケット番号が同じならグループとみなす（家族以外も含む）
for grp, grp_df in all_data.groupby('Ticket'):
    if len(grp_df) > 1:
        if grp_df['Survived'].notnull().sum() > 0:
            smax = grp_df['Survived'].max()
            smin = grp_df['Survived'].min()
            pass_ids = grp_df['PassengerId']
            
            if smax == 1.0:
                all_data.loc[all_data['PassengerId'].isin(pass_ids), 'FamilySurvival'] = 1
            elif smin == 0.0:
                all_data.loc[all_data['PassengerId'].isin(pass_ids), 'FamilySurvival'] = 0


# ---------------------------------------------------------
# 3. 数値化（Encoding）と特徴量選択
# ---------------------------------------------------------

# カテゴリ変数を機械学習モデルが扱える数値（0/1のフラグ）に変換する（One-Hot Encoding）．
# 対象：Title, Pclass, Sex, Embarked, CabinGroup
cat_features = ['Title', 'Pclass', 'Sex', 'Embarked', 'CabinGroup']
all_data_encoded = pd.get_dummies(all_data, columns=cat_features)

# 学習に不要な列（元の文字列データやIDなど）を削除し，使用する特徴量を決定する．
target_cols = [col for col in all_data_encoded.columns if col not in 
               ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Surname', 'CabinInit', 'Age', 'Fare', 'SibSp', 'Parch', 'is_train']]

print(f"採用特徴量 ({len(target_cols)}個):\n")
print(target_cols)

# 学習用データとテスト用データを再び分離．
X = all_data_encoded[all_data_encoded['is_train'] == 1][target_cols].copy()
y = train_data["Survived"]
X_test = all_data_encoded[all_data_encoded['is_train'] == 0][target_cols].copy()


# --- 4. モデルの学習と検証 (Validation) ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 前処理の追加: 標準化 (Standardization) ---
# ロジスティック回帰やSVM，KNNなどは数値のスケール（大きさ）に敏感なため，
# 平均0，分散1になるようにデータを変換する（決定木系モデルには影響が少ないが悪影響もない）．
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_scaled = scaler.fit_transform(X)       # 本番学習用
X_test_scaled = scaler.transform(X_test) # 本番予測用

# --- 複数のモデルを定義 (過学習対策済みのパラメータ) ---
models = {
    # C=0.1: 正則化を強めて，複雑すぎる境界線を引かないようにする．
    "Logistic Regression": LogisticRegression(C=0.1, max_iter=1000, random_state=1),
    
    # C=1.0: デフォルトより少し控えめにしてマージンを確保．probability=Trueは確率を出すため．
    "Support Vector Machine (SVM)": SVC(C=1.0, kernel='rbf', probability=True, random_state=1),
    
    # n_neighbors=10: 近傍点を増やして，ノイズに左右されにくい滑らかな境界にする．
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=10),
    
    # max_depth=6: 木を浅くする（以前は8）．
    # min_samples_leaf=4: 葉ノードのデータ数を増やす（以前は2）．これらにより個別の暗記を防ぐ．
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=10, min_samples_leaf=4, random_state=1),
    
    # learning_rate=0.05: 学習率を下げて，ゆっくり学習させることで汎化性能を高める．
    # subsample=0.8: データの一部だけを使って木を作る（確率的勾配ブースティング）．
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=1)
}

best_model = None
best_cv_accuracy = 0.0
best_model_name = ""

print("\n--- モデル比較結果 (過学習チェック) ---")
# 5分割の交差検証を行う設定（層化抽出：生存/死亡の割合を維持して分割）
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for name, model in models.items():
    # 1. 通常の学習と検証（train_test_splitの結果）
    model.fit(X_train_scaled, y_train)
    
    # 訓練データ自身に対するスコア（Training Accuracy）
    # これが 1.0 (100%) に近すぎると過学習の疑いがある．
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    
    # 検証データに対するスコア（Validation Accuracy）
    val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    
    # 2. 交差検証（Cross Validation）による安定したスコア
    # データを5通りに分けて検証した平均スコア．より信頼性が高い．
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    cv_acc = cv_scores.mean()
    
    print(f"\n[{name}]")
    print(f"  Train Acc : {train_acc:.4f} (学習データへの当てはまり)")
    print(f"  Val Acc   : {val_acc:.4f} (検証データでのスコア)")
    print(f"  CV Mean   : {cv_acc:.4f} (交差検証の平均スコア)")
    
    # 過学習チェック: TrainとValの差を確認
    gap = train_acc - val_acc
    print(f"  Gap       : {gap:.4f}", end="")
    if gap > 0.1:
        print("  <-- 過学習の可能性大 (差が10%以上)")
    elif gap > 0.05:
        print("  <-- 注意: 少し過学習気味 (差が5%以上)")
    else:
        print("  (良好)")

    # 交差検証の平均スコアでベストモデルを決定
    if cv_acc > best_cv_accuracy:
        best_cv_accuracy = cv_acc
        best_model = model
        best_model_name = name

print("\n------------------------------------------------------")
print(f"最も安定して精度が高かったモデル: {best_model_name} (CV Mean: {best_cv_accuracy:.4f})")
print("------------------------------------------------------")

# 特徴量重要度の表示（決定木ベースのモデルの場合のみ可能）
if hasattr(best_model, 'feature_importances_'):
    feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(f"\n--- {best_model_name} の特徴量重要度 (Top 10) ---\n")
    print(feature_importances.head(10))


# --- 5. 本番用の予測とファイル出力 ---
# 最も成績の良かったモデルを使って，全学習データで再学習し，提出用ファイルを作成する．
print(f"\n--- {best_model_name} で本番用モデルを作成し，提出ファイルを出力します ---")

best_model.fit(X_scaled, y) 
test_predictions = best_model.predict(X_test_scaled)

# 提出用ファイル（csv）を作成する．
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})
output.to_csv('./titanic_data/submission.csv', index=False)
print("\nYour submission was successfully saved!")