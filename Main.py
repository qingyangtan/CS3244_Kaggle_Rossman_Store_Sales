import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split

## importing the models that will be trialed and tested against validation dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

store = pd.read_csv('store.csv')
train = pd.read_csv('train.csv',dtype={"StateHoliday": str})
test = pd.read_csv('test.csv',dtype={"StateHoliday": str})

"""
Functions that help initialise data and include new features to the data set to be trained
"""
def initialise_train_data(train, store):
    ## removed 0 sales because they are not used in grading
    train = train[train.Sales != 0]
    ## fill the N.A.N values
    store = store.fillna(0)
    ## combine all features together
    df = train.merge(store, on='Store')
    ## Get labels and remove from dataframe
    labels = df.values[:,3]
    labels = np.array([labels]).T
    #df = df.drop('Sales', axis=1)
    return (df, labels)

def initialise_test_data(test, store):
    ## label our dataframe with id
    test['Id'] = range(1,len(test)+1)
    ## fill the N.A.N values
    store = store.fillna(0)
    ## combine all features together
    df = test.merge(store, on='Store')
    df = df.sort_values(['Id'], ascending=[1])
    return df

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

def remove_intermediate_features(df):
    df = df.drop('monthStr', 1)
    df = df.drop('PromoInterval', 1)
    df = df.drop('Date', 1)
    ## should we remove December data points, apparently it's an outlier (due to it
	## being a holiday and they tend to spend more)
	# df = df[df.Month != 12]
    return df

## check for rows which stores are closed
def get_closed_stores_index(df):
    return df.ix[test['Open']==0].index

"""
Build Features in the order defined by feature_builders array
"""
closed_index = get_closed_stores_index(test)

train, labels = initialise_train_data(train, store)
test = initialise_test_data(test, store)

feature_builders = [date_convert, mapping_encoding, store_features, remove_intermediate_features]

for i in range(len(feature_builders)):
    train = feature_builders[i](train)
    test = feature_builders[i](test)

"""
Functions Wrappers that return models
"""

def rmspe(pred, labels):
    return np.float32(np.sqrt(np.mean((pred/labels-1) ** 2)))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

# """
# Cross Validation Code
# """
# params_svr = {'kernel': ['rbf', 'poly'], 
# 			 'degree': range(5), 
# 			 'C': np.logspace(-4, 1, 5),
# 			 'epsilon': np.logspace(-4, 0, 5),
# 			 'gamma': np.logspace(-3, 2, 5)
# 			 }

# params_rf = {'n_estimators': np.logspace(0, 4, 5),
# 			'max_features': ['auto', 'sqrt', 'log2'],
# 			'max_depth': range(9)
# 			}

# params_xgb = {'n_estimators': np.logspace(0, 4, 5),
# 			'learning_rate': np.logspace(-4, 0, 5),
# 			'max_depth': range(9),
# 			'objective':['reg:linear'],
# 			"subsample": [0.8],
#           	"colsample_bytree": [0.7]
# 			}

# #Subset of params just to make code faster to test
# # params_svr = {
# #     'kernel': ['rbf', 'poly'],
# #     'degree': range(3)
# # }

# # params_rf = {
# #     'n_estimators': np.logspace(0,3,5),
# #     'max_depth': range(5),
# # }

# # params_xgb = {
# #     'n_estimators': np.logspace(0,3,5),
# #     'max_depth': range(0,9,2),
# #     'objective': ['reg:linear'],
# # }


# def cv(df, labels, models, n, k, scoring_fn):
#     clfs = []
#     cv_sets = TimeSeriesSplit(n_splits=k).split(df)
#     counter = 0
#     for idx, model_params in enumerate(models):
#     	if idx == 2:
#     		df = xgb.DMatrix(df, labels)
#     		cv_sets = TimeSeriesSplit(n_splits=k).split(df)
#     	params, model = model_params
#         counter += 1
#         print(counter)
#         clf = RandomizedSearchCV(model, params, scoring=scoring_fn, cv=cv_sets)
#         fitted = clf.fit(df, labels.ravel())
#         clfs.append(fitted)
#     return clfs


# models = [(params_svr, SVR()), (params_rf, RandomForestRegressor()),\
# 		(params_xgb, xgb.XGBClassifier())]
# optimal_models = cv(df[,:50000], labels[,:50000], models, 5, 10,\
# 				make_scorer(rmspe))
# print(optimal_models)


# ##############
# ### TO-DOs ###
# ##############
# # fitted models used on test sets to obtain ensembled average and compare results
# # of single & ensembled models (train on test data - extracted from our train data)


# """
# """
# store1 = df.loc[df.Store == 1]
# #store1.CompetitionOpenSinceYear
# #plt.plot(store1.CompetitionOpenSinceMonth)
# #plt.plot(store1.Sales)
# plt.show()
# s = test["Store"]
# d = test["Date"]
# test['Id'] = list(zip(test.Store, test.Date))
# test.head()

"""
Submission Code
"""
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
dtrain = xgb.DMatrix(X_train.drop('Sales', axis=1), y_train)
dvalid = xgb.DMatrix(X_valid.drop('Sales', axis=1), y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=False)

## Local Validation
pred_y = gbm.predict(xgb.DMatrix(X_valid.drop('Sales', axis=1)))
error = rmspe(X_valid.Sales.values, np.expm1(pred_y))

## Local RMSPE
print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the Test set")
dtest = xgb.DMatrix(test.drop('Id', axis=1))
output = gbm.predict(dtest)

result = pd.DataFrame({'Id': test.Id,'Sales': np.expm1(output)})

## Set closed stores to have 0 sales
for i in closed_index:
    result.ix[result.Id == (i+1), 'Sales'] = 0

## Make a Submission
result.to_csv("The Learning Machine.csv", index=False)

print("DONE")
