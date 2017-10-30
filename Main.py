import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

## importing the models that will be trialed and tested against validation dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

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
    mappings_month = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['monthStr'] = df.Month.map(mappings_month)
    df.loc[df.PromoInterval == 0, 'PromoInterval'] = ''
    df['IsPromoMonth'] = 0
    for interval in df.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                df.loc[(df.monthStr == month) & (df.PromoInterval == interval), 'IsPromoMonth'] = 1

    return df

def remove_intermediate_features(df):
    df = df.drop('monthStr', 1)
    df = df.drop('PromoInterval', 1)
    df = df.drop('Date', 1)
    return df

## check for rows which stores are closed
def get_closed_stores_index(df):
    return df.ix[test['Open']==0].index

"""
Build Features in the order defined by feature_builders array
"""
df, labels = initialise_data(train, store)

feature_builders = [date_convert, mapping_encoding, store_features, remove_intermediate_features]

for i in range(len(feature_builders)):
    df = feature_builders[i](df)

closed_index = get_closed_stores_index(test)

"""
Functions Wrappers that return models
"""

def rmspe(model, X, labels):
    clf = model.fit(X, labels)
    pred = clf.predict(X)
    return np.float32(np.sqrt(np.mean((pred/labels-1) ** 2)))

"""
Cross Validation Code
"""
params_svr = {'kernel': ['rbf', 'poly'], 
			 'degree': range(5), 
			 'C': np.logspace(-4, 2, 10),
			 'epsilon': np.logspace(-4, 0, 10),
			 'gamma': np.logspace(-3, 2, 10)
			 }

params_rf = {'n_estimators': np.logspace(0, 4, 10),
			'max_features': ['auto', 'sqrt', 'log2'],
			'max_depth': range(9)
			}

params_xgb = {'n_estimators': np.logspace(0, 4, 10),
			'learning_rate': np.logspace(-4, 0, 10),
			'max_depth': range(9),
			'objective':['reg:linear'],
			"subsample": [0.8],
          	"colsample_bytree": [0.7]
			}

def cv(df, labels, models, n, k, scoring_fn):
    clfs = []
    cv_sets = TimeSeriesSplit(n_splits = k).split(df)
    counter = 0
    for params, model in models:
        counter += 1
        print(counter)
        clf = GridSearchCV(model, params, scoring=scoring_fn, cv=cv_sets)
        fitted = clf.fit(df, labels.ravel())
        clfs.append(fitted)
    return clfs


models = [(params_svr, SVR()), (params_rf, RandomForestRegressor()), (params_xgb, xgb.XGBClassifier())]
optimal_models = cv(df, labels, models, 5, 10, rmspe)
print(optimal_models)
"""

"""
store1 = df.loc[df.Store == 1]
#store1.CompetitionOpenSinceYear
#plt.plot(store1.CompetitionOpenSinceMonth)
#plt.plot(store1.Sales)
plt.show()
s = test["Store"]
d = test["Date"]
test['Id'] = list(zip(test.Store, test.Date))
test.head()




# """
# Submission Code
# """

# ## Sort the values back to original; Pandas merge function will jumble up the rows
# test = test.sort_values(['Date','DayOfWeek','Store'], ascending=[0,0,1])

# ## Parameters for XGBoost
# params = {"objective": "reg:linear",
#           "booster" : "gbtree",
#           "eta": 0.2,
#           "max_depth": 10,
#           "subsample": 0.8,
#           "colsample_bytree": 0.7,
#           "silent": 1,
#           "seed": 3244
#           }

# num_boost_round = 300

# ## Train a XGBoost model
# X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
# y_train = np.log1p(X_train.Sales)
# y_valid = np.log1p(X_valid.Sales)
# dtrain = xgb.DMatrix(X_train[features], y_train)
# dvalid = xgb.DMatrix(X_valid[features], y_valid)

# watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
# gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
#   early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=False)

# ## Local Validation
# yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
# error = rmspe(X_valid.Sales.values, np.expm1(yhat))

# ## Local RMSPE
# print('RMSPE: {:.6f}'.format(error))

# print("Make predictions on the test set")
# dtest = xgb.DMatrix(test[features])
# test_probs = gbm.predict(dtest)


# result = pd.DataFrame({'Id': np.array(range(1,len(test_probs)+1)) , 'Sales': np.expm1(test_probs)})

# ## Set closed stores to have 0 sales
# for i in closed_index:
#     result.ix[i,'Sales'] = 0

# ## Make a Submission
# result.to_csv("fake.csv", index=False)




# print("DONE")
