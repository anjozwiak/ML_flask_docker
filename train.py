import pandas as pd
import numpy as np
from joblib import dump

df = pd.read_csv('Customertravel.csv')
from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.Target.values
y_val = df_val.Target.values
y_test = df_test.Target.values
del df_train['Target']
del df_val['Target']
del df_test['Target']
df_full_train = df_full_train.reset_index(drop=True)
numerical = ['Age', 'ServicesOpted']
from sklearn.metrics import mutual_info_score
categorical = ['FrequentFlyer', 'AnnualIncomeClass',
       'AccountSyncedToSocialMedia', 'BookedHotelOrNot']
def mutual_info_churn_score(series):
    return mutual_info_score(series,df_full_train.Target)
mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)
from sklearn.feature_extraction import DictVectorizer
train_dicts = df_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.fit_transform(val_dicts)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_train)[:,1]
import pickle
output_file = f'model.bin'
f_out = open(output_file, 'wb') 
pickle.dump((dv, model), f_out)
f_out.close()




