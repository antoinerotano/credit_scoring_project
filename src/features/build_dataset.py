#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %% [code]
# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('/Users/arotanodari/Desktop/COURS/OCR 7 - projet MLOps/Projet+Mise+en+prod+-+home-credit-default-risk/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('/Users/arotanodari/Desktop/COURS/OCR 7 - projet MLOps/Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df= pd.concat([df, test_df]).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('/Users/arotanodari/Desktop/COURS/OCR 7 - projet MLOps/Projet+Mise+en+prod+-+home-credit-default-risk/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('/Users/arotanodari/Desktop/COURS/OCR 7 - projet MLOps/Projet+Mise+en+prod+-+home-credit-default-risk/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('/Users/arotanodari/Desktop/COURS/OCR 7 - projet MLOps/Projet+Mise+en+prod+-+home-credit-default-risk/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('/Users/arotanodari/Desktop/COURS/OCR 7 - projet MLOps/Projet+Mise+en+prod+-+home-credit-default-risk/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('/Users/arotanodari/Desktop/COURS/OCR 7 - projet MLOps/Projet+Mise+en+prod+-+home-credit-default-risk/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('/Users/arotanodari/Desktop/COURS/OCR 7 - projet MLOps/Projet+Mise+en+prod+-+home-credit-default-risk/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    
    return df


# In[2]:


df = main()


# In[3]:


# ================================================================
#  RÉ-ÉQUILIBRAGE  +  VISU HISTOGRAMMES
# ================================================================
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

print("\n>>>  Ré-équilibrage SMOTE + UnderSample")
train_df = df[df["TARGET"].notnull()].copy()
test_df  = df[df["TARGET"].isnull()].copy()

FEATS = [c for c in train_df.columns
         if c not in ["TARGET","SK_ID_CURR","SK_ID_BUREAU","SK_ID_PREV","index"]]

# ------------------------------------------------------------------
# 1) pipeline SMOTE + UnderSample
# ------------------------------------------------------------------
X = train_df[FEATS].replace([np.inf, -np.inf], np.nan)
y = train_df["TARGET"]

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("smote",   SMOTE(sampling_strategy=0.10, random_state=42)),
    ("under",   RandomUnderSampler(sampling_strategy=0.50, random_state=42))
])

X_res, y_res = pipe.fit_resample(X, y)

print("Avant :", Counter(y))
print("Après :", Counter(y_res))

# ------------------------------------------------------------------
# 2) affichage histogrammes (bar-plots) avant / après
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.countplot(x=y,    ax=axes[0], palette="pastel")
axes[0].set_title("TARGET – avant ré-échantillonnage")
axes[0].set_xlabel("Classe"); axes[0].set_ylabel("Nombre d'échantillons")

sns.countplot(x=y_res, ax=axes[1], palette="pastel")
axes[1].set_title("TARGET – après SMOTE + UnderSample")
axes[1].set_xlabel("Classe"); axes[1].set_ylabel("Nombre d'échantillons")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3) on reconstitue le dataframe complet (train équilibré + test)
# ------------------------------------------------------------------
train_df_res            = pd.DataFrame(X_res, columns=FEATS)
train_df_res["TARGET"]  = y_res.values
df_balanced             = pd.concat([train_df_res, test_df],
                                    axis=0, sort=False).reset_index(drop=True)

print("Nouveau df équilibré :", df_balanced.shape,
      "|  part TARGET=1 ->", round(train_df_res["TARGET"].mean(), 3))


# In[4]:


# %% [code] -------------------------------------------------------------------
# 1️⃣  MLflow : initialisation
# ---------------------------------------------------------------------------
import mlflow, mlflow.sklearn
MLFLOW_DIR = "/Users/arotanodari/Desktop/COURS/OCR 7 - projet MLOps/mlruns"   # <-- adaptez si besoin
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("home-credit-default")
print("✅  MLflow initialisé →", mlflow.get_tracking_uri())


# In[ ]:


# %% [code] -------------------------------------------------------------------
# 2️⃣  Score métier +  FE & data de travail
# ---------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, make_scorer
import numpy as np

def cost_metric(y_true, y_pred_proba, thr=0.5, fn_w=10, fp_w=1):
    # 1) Supporte proba shape (n,) ou (n, 2)
    if y_pred_proba.ndim == 2:          # matrice sklearn : on garde p(classe=1)
        y_pred_proba = y_pred_proba[:, 1]

    # 2) Seuillage : on transforme la proba en classe binaire
    y_pred = (y_pred_proba >= thr).astype(int)

    # 3) Matrice de confusion → TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 4) Calcul du coût (poids appliqués), puis signe -
    return -(fn_w * fn + fp_w * fp)

cost_scorer = make_scorer(cost_metric, needs_proba=True)

# -------------------------------------------------------------
#  on part du dataframe équilibré créé plus haut :
#     ↳  train_df_res  = X_res + y_res   (TARGET équilibré)
# -------------------------------------------------------------
FEATS     = [c for c in train_df_res.columns if c != "TARGET"]
X_train   = train_df_res[FEATS]
y_train   = train_df_res["TARGET"]

print(f"Shape train équilibré : {X_train.shape}")


# In[ ]:


# %% --------------------------------------------------------------------------
# 3️⃣  GridSearch + tracking MLflow
# --------------------------------------------------------------------------
from sklearn.pipeline          import Pipeline as SkPipeline      # (alias pour éviter le conflit imblearn.Pipeline)
from sklearn.preprocessing      import StandardScaler
from lightgbm                   import LGBMClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.model_selection    import StratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier   
from sklearn.metrics            import confusion_matrix
import numpy as np, os, json, tempfile, joblib, mlflow.sklearn

# -----------------------------------------------------------------------------
# Définition des deux pipelines + grilles d'hyper-paramètres
# -----------------------------------------------------------------------------
pipe_lgbm = SkPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", LGBMClassifier(
        objective     = "binary",
        n_estimators  = 500,
        random_state  = 42,
        n_jobs        = -1,
        verbose       = -1
    ))
])

grid_lgbm = {
    "clf__num_leaves"   : [31, 63],
    "clf__learning_rate": [0.05, 0.02],
    "clf__max_depth"    : [-1, 8]
}

pipe_logreg = SkPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler" , StandardScaler(with_mean=False)),   # évite l'erreur sur grosses matrices
    ("clf"    , LogisticRegression(max_iter=1000, solver="lbfgs"))
])

grid_logreg = {
    "clf__C": [0.1, 1.0, 10.0]
}

pipe_dummy = SkPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", DummyClassifier(strategy="most_frequent"))  # ou "stratified"
])

grid_dummy = {}

MODELS = {
    "LightGBM"          : (pipe_lgbm , grid_lgbm ),
    "LogisticRegression": (pipe_logreg, grid_logreg),
    "DummyClassifier"   : (pipe_dummy , grid_dummy)   
}

# -----------------------------------------------------------------------------
# Cross-validation commune
# -----------------------------------------------------------------------------
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_models = {}

# -----------------------------------------------------------------------------#
# Run parent – pour regrouper les deux sous-runs
# -----------------------------------------------------------------------------#
with mlflow.start_run(run_name="Model_Training") as parent_run:
    print("MLflow parent run :", parent_run.info.run_id)
    
    for name, (pipe, grid) in MODELS.items():
        print(f"\n🔍  GridSearch pour {name}")
        gs = GridSearchCV(
            estimator = pipe,
            param_grid= grid,
            scoring   = {"AUC": "roc_auc", "Cost": cost_scorer},
            refit     = "Cost",
            cv        = CV,
            n_jobs    = -1,
            verbose   = 1
        )
        
        # ----------- sous-run MLflow ----------------------------------------
        with mlflow.start_run(run_name=name, nested=True) as child_run:
            gs.fit(X_train, y_train)
            
            # ➡️  Params
            best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
            mlflow.log_params(best_params)
            
            # ➡️  Métriques CV
            best_auc  = gs.cv_results_["mean_test_AUC"  ][gs.best_index_]
            best_cost = gs.cv_results_["mean_test_Cost" ][gs.best_index_]
            mlflow.log_metric("AUC_cv" ,  best_auc)
            mlflow.log_metric("Cost_cv",  best_cost)
            
            # ➡️  Confusion matrix complète (train équilibré) au seuil 0.5
            y_proba = gs.best_estimator_.predict_proba(X_train)[:, 1]
            cm      = confusion_matrix(y_train, (y_proba >= 0.5).astype(int))
            cm_path = f"confusion_{name}.npy"
            np.save(cm_path, cm)
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")
            os.remove(cm_path)
            
            # ➡️  Sauvegarde du modèle
            mlflow.sklearn.log_model(gs.best_estimator_, "model")
            
            print(f"   → meilleur Cost = {best_cost:.0f} | AUC = {best_auc:.4f}  (run_id : {child_run.info.run_id})")
            best_models[name] = gs.best_estimator_

print("\n✅  GridSearch terminés – modèles et métriques disponibles dans MLflow!")

# -----------------------------------------------------------------------------
# 4️⃣  (optionnel) aperçu rapide des runs depuis le notebook
# -----------------------------------------------------------------------------
from pprint import pprint
runs = mlflow.search_runs(experiment_names=["home-credit-default"],
                          output_format="list")
print(f"\nNombre total de runs enregistrés : {len(runs)}")
pprint([(r.info.run_name, r.info.run_id) for r in runs][-4:])

