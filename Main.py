import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split

## importing the models that will be trialed and tested against validation dataset
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

store = pd.read_csv('store.csv')
train = pd.read_csv('train.csv',dtype={"StateHoliday": str})
test = pd.read_csv('test.csv',dtype={"StateHoliday": str})

"""
Functions that help initialise and preprocess the data
"""
def initialise_train_data(train, store):
    ## label our dataframe with id
    train['Id'] = range(1,len(train)+1)
    ## fill the N.A.N values
    store = store.fillna(0)
    ## combine all features together
    df = train.merge(store, on='Store')
    df = df.sort_values(['Id'], ascending=[1])
    df = df.drop(['Id'], axis=1)
    return df

def initialise_test_data(test, store):
    ## label our dataframe with id
    test['Id'] = range(1,len(test)+1)
    ## fill the N.A.N values
    store = store.fillna(0)
    ## combine all features together
    df = test.merge(store, on='Store')
    df = df.sort_values(['Id'], ascending=[1])
    return df

"""
Functions that include preprocess existing features to the data set to be trained
"""
def set_store_close_and_sales_data(df):
    df['StoreClosedNextDay'] = df['Open']
    df['StoreClosedNextDay'] = df.StoreClosedNextDay.shift(-1)
    df['StoreClosedNextDay'] = -1 * (df['StoreClosedNextDay'] - 1)
    df['StoreClosedNextDay'][df.shape[0] - 1] = 0
    df = df.fillna(0)
    df['StoreClosedNextDay'] = df.StoreClosedNextDay.astype(int)

    if 'Sales' in df.columns:
        df = df[df.Sales != 0]
    return df

## check for rows which stores are closed
def get_closed_stores_index(df):
    return df.ix[df['Open']==0].index

## converting dates into year, month, day and additional feature week of the year
def date_convert(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.weekofyear
    return df

## adjust and standardise the mappings for all the categorical variables.
def mapping_encoding(df):
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    
    ## replace with values so they can be one hot encoded
    df.StoreType.replace(mappings, inplace=True)
    df.Assortment.replace(mappings, inplace=True)
    df.StateHoliday.replace(mappings, inplace=True)
    df.PromoInterval.replace(mappings, inplace=True)
    
    df['StateHoliday'] = LabelEncoder().fit_transform(df['StateHoliday'])
    df['Assortment'] = LabelEncoder().fit_transform(df['Assortment'])
    df['StoreType'] = LabelEncoder().fit_transform(df['StoreType'])
    return df

def store_features(df):
    mappings_month = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
    				7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['monthStr'] = df.Month.map(mappings_month)
    df.loc[df.PromoInterval == 0, 'PromoInterval'] = ''
    df['IsPromoMonth'] = 0
    for interval in df.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                df.loc[(df.monthStr == month) & (df.PromoInterval == interval),
                	'IsPromoMonth'] = 1
    return df

"""
Functions that engineer new features to from the train data set and map to train & test data set
"""
## return a mapping the avearge sales for each store ID 
def avg_store_sales(train):
    dic = {}
    for store in train.Store.unique():
        ratio = sum(train.loc[(train['Store'] == store)].Sales)/ len(train.loc[(train['Store'] == store)].Sales)
        dic[store] = ratio
    return dic

def map_avg_store_sales(df,mapping):
    for key,value in mapping.items():
        df.ix[(df.Store == key),'AvgStoreSales'] = value
    return df

## return a mapping the avearge sales for each store ID for each day of the week
def avg_store_sales_by_week(train):
    dic = {}
    for store in train.Store.unique():
        ratio_list = []
        for day in range(1,8):
            n = len(train.loc[(train['Store'] == store) & (train['DayOfWeek'] == day)].Sales)
            if (n!=0):
                ratio = sum(train.loc[(train['Store'] == store) & (train['DayOfWeek'] == day)].Sales)/ n
            else:
                ratio = 0
            ratio_list.append(ratio)
        dic[store] = ratio_list
    return dic

def map_avg_store_sales_by_week(df,mapping):
    for key,value in mapping.items():
        for i in range(len(value)):
            df.ix[(df.Store == key) & (df.DayOfWeek == (i+1)),'AvgStoreSalesbyWeek'] = value[i]
    return df

## return a mapping for the average customers for each store ID for each month     
def store_month_customers(train):
    dic = {}
    for store in train.Store.unique():
        ratio_list = []
        for month in range(1,13):
            n = len(train.loc[(train['Store'] == store) & (train['Month'] == month)].Customers)
            if (n!=0):
                ratio = sum(train.loc[(train['Store'] == store) & (train['Month'] == month)].Customers)/ n
            else:
                ratio = 0
            ratio_list.append(ratio)
        dic[store] = ratio_list
    return dic

def map_store_month_customers(df,mapping):
    for key,value in mapping.items():
        for i in range(len(value)):
            df.ix[(df.Store == key) & (df.Month == (i+1)),'AvgMonthlyCustomer'] = value[i]
    return df

"""
Select the finalized features that will be used for modeling
"""
def select_features(df, feature_type):
    train_features = ['Day','AvgStoreSales','AvgStoreSalesbyWeek','AvgMonthlyCustomer','Promo','Promo2','Year','Assortment','StoreType','StateHoliday','SchoolHoliday','CompetitionDistance']
    test_features = ['Id','Day','AvgStoreSales','AvgStoreSalesbyWeek','AvgMonthlyCustomer','Promo','Promo2','Year','Assortment','StoreType','StateHoliday','SchoolHoliday','CompetitionDistance']
    if feature_type == 'train':
        return df[train_features]
    return df[test_features]

"""
Build Features in the order defined by feature_builders array
"""
closed_index = get_closed_stores_index(test)

train = initialise_train_data(train, store)
test = initialise_test_data(test, store)

feature_builders = [[set_store_close_and_sales_data], [date_convert], [mapping_encoding], [store_features], \
                    [store_month_customers,map_store_month_customers], \
                    [avg_store_sales_by_week,map_avg_store_sales_by_week], \
                    [avg_store_sales,map_avg_store_sales]]

for i in range(len(feature_builders)):
    if (len(feature_builders[i]) == 1):
        train = feature_builders[i][0](train)
        test = feature_builders[i][0](test)
    else:
        mapping = feature_builders[i][0](train)
        train = feature_builders[i][1](train,mapping)
        test = feature_builders[i][1](test,mapping)

#train.to_csv("cached_train.csv", index=False)
#test.to_csv("cached_test.csv", index=False)
#train = pd.read_csv('cached_train.csv')
#test = pd.read_csv('cached_test.csv')

labels = train.Sales
train = select_features(train, 'train')
test = select_features(test, 'test')

"""
Functions Wrappers that return models
"""

def rmspe(pred, labels):
    return np.float32(np.sqrt(np.mean((pred/labels-1) ** 2)))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

"""
Hyper Parameter Optimisation

Choosing hyper parameters for models
"""
params_rf = {'n_estimators':  [10, 100, 500, 1000],
             'max_features': ['auto', 'sqrt', 'log2'],
             'max_depth': [8,9,10,11,12]
            }

params_xgb = {'n_estimators': [100, 300, 500, 1000],
              'learning_rate': [0.1, 0.2, 0.5],
              'max_depth': [8, 9, 10, 12],
              "subsample": [0.8],
              "colsample_bytree": [0.7],
              "seed": [3244],
           }

def optimise_hyper_parameters(train, labels, models, k, scoring_fn):
    clfs = []
    cv_sets = TimeSeriesSplit(n_splits=k).split(train)
    for idx, model_params in enumerate(models):
        params, model = model_params
        clf = GridSearchCV(model, params, scoring=scoring_fn)
        fitted = clf.fit(train, labels.ravel())
        clfs.append(fitted)
    return clfs


#models = [(params_rf, RandomForestRegressor()), (params_xgb, xgb.XGBRegressor())]
#optimal_models = optimise_hyper_parameters(train, np.log1p(labels), models, 5, make_scorer(rmspe))

#for optimal_model in optimal_models:
#    print(optimal_model.best_params_)


"""
Model Selection, Ensembling and Local Validation Result

We set parameters here again so that we do not have to rerun the above cell.
Hyper parameter tuning is time costly.
"""

n = round(len(train)*0.012)
X_train = train[n:] 
X_valid = train[:n]
y_train = labels[n:]
y_valid = labels[:n]
y_train = np.log1p(np.array(y_train, dtype=np.int32))
y_valid = np.log1p(np.array(y_valid, dtype=np.int32))

## XGBoost Results using optimal parameters selected above
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.3,
          "max_depth": 8,
          "subsample": 0.8,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 3244,
          "n_estimators": 1000
          }

num_boost_round = 300
dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_valid, y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

xg_model = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=False)

## Local Validation
pred_y = xg_model.predict(xgb.DMatrix(X_valid))
error = rmspe(np.expm1(y_valid), np.expm1(pred_y))
print('XGB RMSPE: {:.6f}'.format(error))

## Random Forest Results by using optimal parameters selected above
rf_model = RandomForestRegressor(n_estimators=10, max_depth=8, max_features='log2')
rf_model.fit(X_train,y_train)
pred_y = rf_model.predict(X_valid)
error = rmspe(np.expm1(y_valid), np.expm1(pred_y))
print('RF RMSPE: {:.6f}'.format(error))

## SVR Results by using arbitrary parameters. Hyper Parameter Selection takes too long


# We would choose the best model from reported validation result.

"""
Submission Code
"""
# Preferably we can have a final model that is not just xgb. Or using the CV code above

#full_matrix = xgb.DMatrix(train, np.log1p(labels))
#final_model = xgb.train(params, full_matrix, num_boost_round, evals=watchlist, \
#  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=False) 

dtest = xgb.DMatrix(test.drop('Id', axis=1))
output = final_model.predict(dtest)

result = pd.DataFrame({'Id': test.Id,'Sales': np.expm1(output)})

## Set closed stores to have 0 sales
for i in closed_index:
    result.ix[result.Id == (i+1), 'Sales'] = 0

## Make a Submission
result.to_csv("The Learning Machine.csv", index=False)

print("DONE")
