#1 Parameters:

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

#2.read train and test files

# Set constants
USE_DATA_LEAK = 'Y'
RAND_VAL = 44
num_folds = 7
n_est = 5000

# Load data
file_path = 'C:\\Users\\Rubhon\\Videos\\project\\DATASET\\abc.csv'
df = pd.read_csv(file_path)

# Split into train and test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RAND_VAL)

# Make a copy of the test set
df_test_ov = df_test.copy()

# Display first five rows of the test set
print(df_test.head())

#3.scaling

scale_cols = ['Age','CreditScore', 'Balance','EstimatedSalary']
###
for c in scale_cols:
    min_value = df_train[c].min()
    max_value = df_train[c].max()
    df_train[c+"_scaled"] = (df_train[c] - min_value) / (max_value - min_value)
    df_test[c+"_scaled"] = (df_test[c] - min_value) / (max_value - min_value)

#4.TF-IDF Vectorization for Surname

# Ensure no missing values and correct types
for col in ['CustomerId', 'Surname', 'Geography', 'Gender']:
    df_train[col] = df_train[col].fillna("missing").astype(str)
    df_test[col] = df_test[col].fillna("missing").astype(str)

df_train['EstimatedSalary'] = pd.to_numeric(df_train['EstimatedSalary'], errors='coerce').fillna(0)
df_test['EstimatedSalary'] = pd.to_numeric(df_test['EstimatedSalary'], errors='coerce').fillna(0)

# Combine features into a single column
df_train['Sur_Geo_Gend_Sal'] = (
    df_train['CustomerId'] + 
    df_train['Surname'] + 
    df_train['Geography'] + 
    df_train['Gender'] + 
    np.round(df_train['EstimatedSalary']).astype('str')
)
df_test['Sur_Geo_Gend_Sal'] = (
    df_test['CustomerId'] + 
    df_test['Surname'] + 
    df_test['Geography'] + 
    df_test['Gender'] + 
    np.round(df_test['EstimatedSalary']).astype('str')
)

# Ensure no missing or empty strings
df_train['Sur_Geo_Gend_Sal'] = df_train['Sur_Geo_Gend_Sal'].fillna("missing")
df_test['Sur_Geo_Gend_Sal'] = df_test['Sur_Geo_Gend_Sal'].fillna("missing")

# Apply TF-IDF and SVD transformations
def get_vectors(df_train, df_test, col_name):
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors_train = vectorizer.fit_transform(df_train[col_name])
    vectors_test = vectorizer.transform(df_test[col_name])
    
    svd = TruncatedSVD(n_components=3)
    x_pca_train = svd.fit_transform(vectors_train)
    x_pca_test = svd.transform(vectors_test)

    tfidf_df_train = pd.DataFrame(x_pca_train, columns=[f"{col_name}_tfidf_{i}" for i in range(3)], index=df_train.index)
    tfidf_df_test = pd.DataFrame(x_pca_test, columns=[f"{col_name}_tfidf_{i}" for i in range(3)], index=df_test.index)

    df_train = pd.concat([df_train, tfidf_df_train], axis=1)
    df_test = pd.concat([df_test, tfidf_df_test], axis=1)

    print(f"TF-IDF and SVD transformation completed for column: {col_name}")
    return df_train, df_test

df_train, df_test = get_vectors(df_train, df_test, 'Sur_Geo_Gend_Sal')

#5.feature engineering
def getFeats(df):
    # Validate required columns
    required_columns = ['Age', 'Surname', 'Tenure', 'NumOfProducts', 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Ensure numeric and string columns have valid values
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0)
    df['Surname'] = df['Surname'].fillna("missing")
    df['Tenure'] = df['Tenure'].fillna(0)
    df['NumOfProducts'] = df['NumOfProducts'].replace(0, np.nan).fillna(1)

    # Create new features
    df['IsSenior'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)
    df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
    df['Products_Per_Tenure'] = df['Tenure'] / df['NumOfProducts'].replace(0, np.nan).fillna(0)
    df['len_SurName'] = df['Surname'].apply(lambda x: len(x))
    df['AgeCat'] = np.round(df['Age'] / 20).astype('int').astype('category')

    # One-hot encode categorical columns
    cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'AgeCat']
    df = pd.get_dummies(df, columns=cat_cols)

    return df

# Apply feature engineering
df_train = getFeats(df_train)
df_test = getFeats(df_test)

# Define scale_cols and ensure they exist
scale_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
feat_cols = df_train.columns.drop(['id', 'CustomerId', 'Surname', 'Exited', 'Sur_Geo_Gend_Sal'])
feat_cols = feat_cols.drop(feat_cols.intersection(scale_cols))

# Display feature columns
print(feat_cols)

# Create feature and target variables
X = df_train[feat_cols]
if 'Exited' in df_train.columns:
    y = df_train['Exited']
else:
    raise KeyError("Target variable 'Exited' is missing in the training data.")

print(X.head())
print(y.head())

# Define n_est
n_est = 1000

# LightGBM parameters
lgbParams = {'n_estimators': n_est,
             'max_depth': 22, 
             'learning_rate': 0.025,
             'min_child_weight': 3.43,
             'min_child_samples': 216, 
             'subsample': 0.782,
             'subsample_freq': 4, 
             'colsample_bytree': 0.29, 
             'num_leaves': 21}

# Train LightGBM model
LGB = lgb.LGBMClassifier(**lgbParams)
LGB.fit(X, y)

# Plot feature importance
lgb.plot_importance(LGB, importance_type="gain", figsize=(12, 10), max_num_features=12, title="LightGBM Feature Importance (Gain)")
plt.show()


#7.training
folds = StratifiedKFold(n_splits=num_folds, random_state=RAND_VAL, shuffle=True)
test_preds = np.zeros((num_folds, len(df_test)))  # Initialize with zeros (shape: (folds, test_samples))
auc_vals = []

# Ensure feat_cols is correctly defined (it should contain feature columns only)
feat_cols = [col for col in X.columns if col != 'Exited']

# Cross-validation loop
# Cross-validation loop
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    
    # Split the data for training and validation
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
    # Train the model
    LGB = lgb.LGBMClassifier(**lgbParams)
    LGB.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        #early_stopping_rounds=100,  # Ensure it's supported by your LightGBM version
        #verbose=200
    )
    
    # Predict and evaluate the model for validation set
    y_pred_val = LGB.predict_proba(X_val[feat_cols])[:, 1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    print(f"AUC for fold {n_fold}: {auc_val}")
    auc_vals.append(auc_val)
    
    # Store the test predictions for the current fold
    #y_pred_test = LGB.predict_proba(df_test[feat_cols])[:, 1]
    #test_preds[n_fold, :] = y_pred_test
    #print("----------------")

#8.evaluation

print("Mean AUC: ",np.mean(auc_vals))

#9.Prediction and Submission

y_pred = test_preds.mean(axis=0)

#10.Override from Original Dataset

df_orig=pd.read_csv('C:\\Users\\Rubhon\\Videos\\project\\DATASET\\abc.csv')
join_cols=list(df_orig.columns.drop(['RowNumber','Exited']))
df_orig.rename(columns={'Exited':'Exited_Orig'},inplace=True)
df_orig['Exited_Orig']=df_orig['Exited_Orig'].map({0:1,1:0})
df_test_ov=df_test_ov.merge(df_orig,on=join_cols,how='left')[['id','Exited_Orig']].fillna(-1)
####
df_sub = df_test_ov[['id','Exited_Orig']]

if USE_DATA_LEAK=='Y':
    df_sub['Exited'] = np.where(df_sub.Exited_Orig==-1,y_pred,df_sub.Exited_Orig)
else:
    df_sub['Exited'] = y_pred
    
df_sub.drop('Exited_Orig',axis=1,inplace=True)
df_sub.head()

df_sub.to_csv("abc.csv",index=False)

df_sub.hist(column='Exited', bins=20, range=[0,1],figsize=(12,6))
plt.show()