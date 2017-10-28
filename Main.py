import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

store = pd.read_csv('store.csv')
train = pd.read_csv('train.csv',dtype={"StateHoliday": str})
test = pd.read_csv('test.csv')

"""
Functions that help initialise data and include new features to the data set to be trained
"""
def initialise_data(train, store):
    ## removed 0 sales because they are not used in grading
    train = train[train.Sales != 0]
    ## fill the N.A.N values
    store = store.fillna(0)
    ## combine all features together
    df = train.merge(store, on='Store')
    # Get labels and remove from dataframe
    labels = df.values[:,3]
    labels = np.array([labels]).T
    df = df.drop('Sales', axis=1)
    return (df, labels)

## converting dates into year, month, day and additional feature week of the year
def date_convert(df):
    df['Date']  = pd.to_datetime(df['Date'], errors='coerce')
    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.weekofyear
    return df

## adjust and standardise the mappings for all the categorical variables.
def mapping_encoding(df):
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    mappings_month = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    ## replace with values so they can be one hot encoded
    df.StoreType.replace(mappings, inplace=True)
    df.Assortment.replace(mappings, inplace=True)
    df.StateHoliday.replace(mappings, inplace=True)
    df.PromoInterval.replace(mappings, inplace=True)
    
    df['StateHoliday'] = LabelEncoder().fit_transform(df['StateHoliday'])
    df['Assortment']   = LabelEncoder().fit_transform(df['Assortment'])
    df['StoreType']    = LabelEncoder().fit_transform(df['StoreType'])
    return df

## check for rows which stores are closed
def get_closed_stores_index(df):
    return df.ix[test['Open']==0].index

"""
Build Features in the order defined by feature_builders array
"""
df, labels = initialise_data(train, store)

feature_builders = [date_convert, mapping_encoding]

for i in range(len(feature_builders)):
    df = feature_builders[i](df)
    
closed_index = get_closed_stores_index(test)

"""
Functions Wrappers that return models
"""

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
mask = np.ones(3, dtype=bool)
mask[0:2] = False
print(mask)
print(arr[mask])

from sklearn.svm import SVC
import math

def rmspe(pred, labels):
    return np.sqrt(np.mean((pred/labels-1) ** 2))

"""
Cross Validation Code
"""
def cv(df, labels, model):
    df = df.values
    num_rows = df.shape[0]
    K = 10
    cv_score = 0
    for i in range(K):
        # Get validation array
        start_val = math.floor(i/K * num_rows)
        end_val = math.floor((i+1)/K * num_rows)
        
        if K==10:
            end_val = num_rows
        print(start_val, end_val)
        
        df_val = df[start_val:end_val,:]
        labels_val = df[start_val:end_val,:]
        
        # Get training array by deleting rows for validation
        mask = np.ones(num_rows, dtype=bool)
        mask[start_val:end_val+1] = False
        df_train = df[mask]
        labels_train = labels[mask]
        print(mask)
        print(mask.shape)
        print(df.shape)
        print(labels.shape)
        
        fitted = model.fit(df_train, labels_train)
        pred = fitted.predict(df_val)
        
        cv_score += rmspe(pred, labels_val)

md = SVC()

print(cv(df, labels, md))

store1 = df.loc[df.Store == 1]
#store1.CompetitionOpenSinceYear
#plt.plot(store1.CompetitionOpenSinceMonth)
#plt.plot(store1.Sales)
plt.show()
s = test["Store"]
d = test["Date"]
test['Id'] = list(zip(test.Store, test.Date))
test.head()

"""
Submission Code
"""

## Sort the values back to original; Pandas merge function will jumble up the rows
test = test.sort_values(['Date','DayOfWeek','Store'], ascending=[0,0,1])

## Parameters for XGBoost
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.2,
          "max_depth": 10,
          "subsample": 0.8,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 3244
          }

num_boost_round = 300

## Train a XGBoost model
X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=False)

## Local Validation
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))

## Local RMSPE
print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)


result = pd.DataFrame({'Id': np.array(range(1,len(test_probs)+1)) , 'Sales': np.expm1(test_probs)})

## Set closed stores to have 0 sales
for i in closed_index:
    result.ix[i,'Sales'] = 0

## Make a Submission
result.to_csv("fake.csv", index=False)

print("DONE")
