# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
/kaggle/input/ieee-fraud-detection/train_identity.csv 
/kaggle/input/ieee-fraud-detection/test_identity.csv 
/kaggle/input/ieee-fraud-detection/test_transaction.csv 
/kaggle/input/ieee-fraud-detection/sample_submission.csv 
/kaggle/input/ieee-fraud-detection/train_transaction.csv
"""
# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import os, gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# path
path_train_transaction = "../input/ieee-fraud-detection/train_transaction.csv"
path_train_id = "../input/ieee-fraud-detection/train_identity.csv"
path_test_transaction = "../input/ieee-fraud-detection/test_transaction.csv"
path_test_id = "../input/ieee-fraud-detection/test_identity.csv"
path_sample_submission = '../input/ieee-fraud-detection/sample_submission.csv'
path_submission = 'sub_xgb_95.csv'

BUILD95 = False
BUILD96 = True

# cols with strings
str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5',
            'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
            'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

# fisrt 53 columns
cols = ['TransactionID', 'TransactionDT', 'TransactionAmt',
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
        'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
        'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
        'M5', 'M6', 'M7', 'M8', 'M9']

# V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
# https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
v = [1, 3, 4, 6, 8, 11]
v += [13, 14, 17, 20, 23, 26, 27, 30]
v += [36, 37, 40, 41, 44, 47, 48]
v += [54, 56, 59, 62, 65, 67, 68, 70]
v += [76, 78, 80, 82, 86, 88, 89, 91]

# v += [96, 98, 99, 104] #relates to groups, no NAN
v += [107, 108, 111, 115, 117, 120, 121, 123]  # maybe group, no NAN
v += [124, 127, 129, 130, 136]  # relates to groups, no NAN

# LOTS OF NAN BELOW
v += [138, 139, 142, 147, 156, 162]  # b1
v += [165, 160, 166]  # b1
v += [178, 176, 173, 182]  # b2
v += [187, 203, 205, 207, 215]  # b2
v += [169, 171, 175, 180, 185, 188, 198, 210, 209]  # b2
v += [218, 223, 224, 226, 228, 229, 235]  # b3
v += [240, 258, 257, 253, 252, 260, 261]  # b3
v += [264, 266, 267, 274, 277]  # b3
v += [220, 221, 234, 238, 250, 271]  # b3

v += [294, 284, 285, 286, 291, 297]  # relates to grous, no NAN
v += [303, 305, 307, 309, 310, 320]  # relates to groups, no NAN
v += [281, 283, 289, 296, 301, 314]  # relates to groups, no NAN
# v += [332, 325, 335, 338] # b4 lots NAN

cols += ['V' + str(x) for x in v]
dtypes = {}
for c in cols + ['id_0' + str(x) for x in range(1, 10)] + ['id_' + str(x) for x in range(10, 34)]:
    dtypes[c] = 'float32'
for c in str_type:
    dtypes[c] = 'category'

# load data and merge
print("load data...")
X_train = pd.read_csv(path_train_transaction, index_col="TransactionID", dtype=dtypes, usecols=cols + ["isFraud"])
train_id = pd.read_csv(path_train_id, index_col="TransactionID", dtype=dtypes)
X_train = X_train.merge(train_id, how="left", left_index=True, right_index=True)

X_test = pd.read_csv(path_test_transaction, index_col="TransactionID", dtype=dtypes, usecols=cols)
test_id = pd.read_csv(path_test_id, index_col="TransactionID", dtype=dtypes)
X_test = X_test.merge(test_id, how="left", left_index=True, right_index=True)

# target
y_train = X_train["isFraud"]
del train_id, test_id, X_train["isFraud"]

print("X_train shape:{}, X_test shape:{}".format(X_train.shape, X_test.shape))

# transform D feature "time delta" as "time point"
for i in range(1, 16):
    if i in [1, 2, 3, 5, 9]:
        continue
    X_train["D" + str(i)] = X_train["D" + str(i)] - X_train["TransactionDT"] / np.float32(60 * 60 * 24)
    X_test["D" + str(i)] = X_test["D" + str(i)] - X_test["TransactionDT"] / np.float32(60 * 60 * 24)


# encoding function

# frequency encode
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col], df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col + "FE"
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype("float32")
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype("float32")
        print(col)


# label encode
def encode_LE(col, train=X_train, test=X_test, verbose=True):
    df_comb = pd.concat([train[col], test[col]], axis=0)
    df_comb, _ = pd.factorize(df_comb[col])
    nm = col
    if df_comb.max() > 32000:
        train[nm] = df_comb[0: len(train)].astype("float32")
        test[nm] = df_comb[len(train):].astype("float32")
    else:
        train[nm] = df_comb[0: len(train)].astype("float16")
        test[nm] = df_comb[len(train):].astype("float16")
    del df_comb
    gc.collect()
    if verbose:
        print(col)


def encode_AG(main_columns, uids, aggregations=["mean"], df_train=X_train, df_test=X_test, fillna=True, usena=False):
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_column = main_column + "_" + col + "_" + agg_type
                temp_df = pd.concat([df_train[[col, main_column]], df_test[[col, main_column]]])
                if usena:
                    temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan

                #求每个uid下，该col的均值或标准差
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_column})
                #将uid设成index
                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_column].to_dict()
                #temp_df是一个映射字典
                df_train[new_column] = df_train[col].map(temp_df).astype("float32")
                df_test[new_column] = df_test[col].map(temp_df).astype("float32")
                if fillna:
                    df_train[new_column].fillna(-1, inplace=True)
                    df_test[new_column].fillna(-1, inplace=True)
                print(new_column)


# COMBINE FEATURES交叉特征
def encode_CB(col1, col2, df1=X_train, df2=X_test):
    nm = col1 + '_' + col2
    df1[nm] = df1[col1].astype(str) + '_' + df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str) + '_' + df2[col2].astype(str)
    encode_LE(nm, verbose=False)
    print(nm, ', ', end='')


# GROUP AGGREGATION NUNIQUE
def encode_AG2(main_columns, uids, train_df=X_train, test_df=X_test):
    for main_column in main_columns:
        for col in uids:
            comb = pd.concat([train_df[[col] + [main_column]], test_df[[col] + [main_column]]], axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col + '_' + main_column + '_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col + '_' + main_column + '_ct'] = test_df[col].map(mp).astype('float32')
            print(col + '_' + main_column + '_ct, ', end='')


print("encode cols...")
# TRANSACTION AMT CENTS
X_train['cents'] = (X_train['TransactionAmt'] - np.floor(X_train['TransactionAmt'])).astype('float32')
X_test['cents'] = (X_test['TransactionAmt'] - np.floor(X_test['TransactionAmt'])).astype('float32')
print('cents, ', end='')

# FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
encode_FE(X_train, X_test, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])
# COMBINE COLUMNS CARD1+ADDR1, CARD1+ADDR1+P_EMAILDOMAIN
encode_CB('card1', 'addr1')
encode_CB('card1_addr1', 'P_emaildomain')
# FREQUENCY ENOCDE
encode_FE(X_train, X_test, ['card1_addr1', 'card1_addr1_P_emaildomain'])
# GROUP AGGREGATE
encode_AG(['TransactionAmt', 'D9', 'D11'], ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], ['mean', 'std'],
          usena=False)
for col in str_type:
    encode_LE(col, X_train, X_test)
"""
Feature Selection - Time Consistency
We added 28 new feature above. We have already removed 219 V Columns from correlation analysis done here. 
So we currently have 242 features now. We will now check each of our 242 for "time consistency". 
We will build 242 models. Each model will be trained on the first month of the training data and will only use one feature. 
We will then predict the last month of the training data. We want both training AUC and validation AUC to be above AUC = 0.5.
 It turns out that 19 features fail this test so we will remove them. 
 Additionally we will remove 7 D columns that are mostly NAN. More techniques for feature selection are listed here
"""
cols = list(X_train.columns)
cols.remove('TransactionDT')
for c in ['D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']:
    cols.remove(c)

# FAILED TIME CONSISTENCY TEST
for c in ['C3', 'M5', 'id_08', 'id_33']:
    cols.remove(c)
for c in ['card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34']:
    cols.remove(c)
for c in ['id_' + str(x) for x in range(22, 28)]:
    cols.remove(c)

print('NOW USING THE FOLLOWING', len(cols), 'FEATURES.')
# CHRIS - TRAIN 75% PREDICT 25%
idxT = X_train.index[:3 * len(X_train) // 4]
idxV = X_train.index[3 * len(X_train) // 4:]
print(X_train.info())
# X_train = X_train.convert_objects(convert_numeric=True)
# X_test = X_test.convert_objects(convert_numeric=True)

for col in str_type:
    print(col)
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)
print("after transform:")
print(X_train.info())

# fillna
for col in cols:
    X_train[col].fillna(-1, inplace=True)
    X_test[col].fillna(-1, inplace=True)

import xgboost as xgb

print("XGBoost version:", xgb.__version__)
"""

if BUILD95:
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))
    clf = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        missing=-1,
        eval_metric='auc',
        # USE CPU
        #nthread=4,
        #tree_method='hist'
        # USE GPU
        tree_method='gpu_hist'
    )
    h = clf.fit(X_train.loc[idxT,cols], y_train[idxT],
        eval_set=[(X_train.loc[idxV,cols],y_train[idxV])],
                verbose=100, early_stopping_rounds=200)

    oof[idxV] = clf.predict_proba(X_train[cols].iloc[idxV])[:, 1]
    preds = clf.predict_proba(X_test[cols])[:, 1]
    del h, clf
    x = gc.collect()
print('#' * 20)
print('XGB95 OOF CV=', roc_auc_score(y_train, oof))

if BUILD95:
    sample_submission = pd.read_csv(path_sample_submission)
    sample_submission.isFraud = preds
    sample_submission.to_csv(path_submission, index=False)
"""

import datetime

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
X_train['DT_M'] = X_train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
X_train['DT_M'] = (X_train['DT_M'].dt.year - 2017) * 12 + X_train['DT_M'].dt.month

X_test['DT_M'] = X_test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
X_test['DT_M'] = (X_test['DT_M'].dt.year - 2017) * 12 + X_test['DT_M'].dt.month

print("training...")
if BUILD95:
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    skf = GroupKFold(n_splits=6)
    for i, (idxT, idxV) in enumerate(skf.split(X_train, y_train, groups=X_train['DT_M'])):
        month = X_train.iloc[idxV]['DT_M'].iloc[0]
        print('Fold', i, 'withholding month', month)
        print(' rows of train =', len(idxT), 'rows of holdout =', len(idxV))
        clf = xgb.XGBClassifier(
            n_estimators=5000,
            max_depth=12,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.4,
            missing=-1,
            eval_metric='auc',
            # USE CPU
            # nthread=4,
            # tree_method='hist'
            # USE GPU
            tree_method='gpu_hist'
        )
        h = clf.fit(X_train[cols].iloc[idxT], y_train.iloc[idxT],
                    eval_set=[(X_train[cols].iloc[idxV], y_train.iloc[idxV])],
                    verbose=100, early_stopping_rounds=200)

        oof[idxV] += clf.predict_proba(X_train[cols].iloc[idxV])[:, 1]
        preds += clf.predict_proba(X_test[cols])[:, 1] / skf.n_splits
        del h, clf
        x = gc.collect()
    print('#' * 20)
    print('XGB95 OOF CV=', roc_auc_score(y_train, oof))

if BUILD95:
    sample_submission = pd.read_csv(path_sample_submission)
    sample_submission.isFraud = preds
    sample_submission.to_csv(path_submission, index=False)

X_train['day'] = X_train.TransactionDT / (24 * 60 * 60)
X_train['uid'] = X_train.card1_addr1.astype(str) + '_' + np.floor(X_train.day - X_train.D1).astype(str)

X_test['day'] = X_test.TransactionDT / (24 * 60 * 60)
X_test['uid'] = X_test.card1_addr1.astype(str) + '_' + np.floor(X_test.day - X_test.D1).astype(str)

# FREQUENCY ENCODE UID
encode_FE(X_train, X_test, ['uid'])
# AGGREGATE
encode_AG(['TransactionAmt', 'D4', 'D9', 'D10', 'D15'], ['uid'], ['mean', 'std'], fillna=True, usena=True)
# AGGREGATE
encode_AG(['C' + str(x) for x in range(1, 15) if x != 3], ['uid'], ['mean'], X_train, X_test, fillna=True, usena=True)
# AGGREGATE
encode_AG(['M' + str(x) for x in range(1, 10)], ['uid'], ['mean'], fillna=True, usena=True)
# AGGREGATE
encode_AG2(['P_emaildomain', 'dist1', 'DT_M', 'id_02', 'cents'], ['uid'], train_df=X_train, test_df=X_test)
# AGGREGATE
encode_AG(['C14'], ['uid'], ['std'], X_train, X_test, fillna=True, usena=True)
# AGGREGATE
encode_AG2(['C13', 'V314'], ['uid'], train_df=X_train, test_df=X_test)
# AGGREATE
encode_AG2(['V127', 'V136', 'V309', 'V307', 'V320'], ['uid'], train_df=X_train, test_df=X_test)
# NEW FEATURE
X_train['outsider15'] = (np.abs(X_train.D1 - X_train.D15) > 3).astype('int8')
X_test['outsider15'] = (np.abs(X_test.D1 - X_test.D15) > 3).astype('int8')
print('outsider15')

cols = list(X_train.columns)
cols.remove('TransactionDT')
for c in ['D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']:
    if c in cols:
        cols.remove(c)
for c in ['oof', 'DT_M', 'day', 'uid']:
    if c in cols:
        cols.remove(c)

# FAILED TIME CONSISTENCY TEST
for c in ['C3', 'M5', 'id_08', 'id_33']:
    if c in cols:
        cols.remove(c)
for c in ['card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34']:
    if c in cols:
        cols.remove(c)
for c in ['id_' + str(x) for x in range(22, 28)]:
    if c in cols:
        cols.remove(c)
print('NOW USING THE FOLLOWING', len(cols), 'FEATURES.')
print(np.array(cols))

if BUILD96:

    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    skf = GroupKFold(n_splits=6)
    for i, (idxT, idxV) in enumerate(skf.split(X_train, y_train, groups=X_train['DT_M'])):
        month = X_train.iloc[idxV]['DT_M'].iloc[0]
        print('Fold', i, 'withholding month', month)
        print(' rows of train =', len(idxT), 'rows of holdout =', len(idxV))
        clf = xgb.XGBClassifier(
            n_estimators=5000,
            max_depth=12,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.4,
            missing=-1,
            eval_metric='auc',
            # USE CPU
            # nthread=4,
            # tree_method='hist'
            # USE GPU
            tree_method='gpu_hist'
        )
        h = clf.fit(X_train[cols].iloc[idxT], y_train.iloc[idxT],
                    eval_set=[(X_train[cols].iloc[idxV], y_train.iloc[idxV])],
                    verbose=100, early_stopping_rounds=200)

        oof[idxV] += clf.predict_proba(X_train[cols].iloc[idxV])[:, 1]
        preds += clf.predict_proba(X_test[cols])[:, 1] / skf.n_splits
        del h, clf
        x = gc.collect()
    print('#' * 20)
    print('XGB96 OOF CV=', roc_auc_score(y_train, oof))

if BUILD96:
    sample_submission = pd.read_csv(path_sample_submission)
    sample_submission.isFraud = preds
    sample_submission.to_csv(path_submission, index=False)

