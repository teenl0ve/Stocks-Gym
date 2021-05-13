import pandas as pd
import yfinance as yf

# Load the tickers in with pandas
sp500_tickers = pd.read_csv('sp500.csv', usecols=[1])

# Finding odd entries in tickers list, these tickers do not have data from the scrape and do not
# work with yfinance. They have a common trait of containing periods, which we will use to locate
bad_tickers = []
for ticker in sp500_tickers.Symbol:
    if '.' in ticker:
        bad_tickers.append(ticker)
        
# Exclude the bad tickers from our list
sp500_tickers = sp500_tickers[~sp500_tickers.Symbol.isin(bad_tickers)]['Symbol']

# Use yfinance to gather the pricing data, isolating the Closing prices
start_date = '2020-05-08'
end_date = '2020-10-29'
# yfinance likes the tickers formatted as a list
ticks = yf.Tickers(list(sp500_tickers))
sp500_close = ticks.history(start=start_date, end=end_date).Close
sp500_close.tail()


# Dropping AGN and ETFC
sp500_close.drop(columns=['AGN', 'ETFC'], inplace=True)

# Get log prices
log_close = np.log(sp500_close)
# Create a new dataframe to store our log returns
log_returns = pd.DataFrame(index=log_close.columns, columns=['log_return'])
# Populate new dataframe with log returns for each security
for col in log_close.columns:
    log_return = log_close[col].iloc[-1] - log_close[col].iloc[0]
    log_returns.loc[col, 'log_return'] = log_return
    
log_returns.head()



import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(seaborn-darkgrid)

sns.distplot(log_returns.log_return)
mean_return = log_returns.log_return.mean()
plt.title('Distribution of Log Returns in S&P 500 from May 8, 2020 - Oct 29, 2020')
plt.axvline(mean_return, ls=':', label='Index Mean: {}'.format(round(mean_return, 2)))
plt.legend()



to_drop = ['% Above Low', # Price related
           '% Below High', # Price related
           '52-Wk Range', # Price related
           '5yr High', # Price related
           '5yr Low', # Price related
           'Annual Dividend $', # Leave this out because % is more generalized
           'Ask', # Price related
           'Ask Size', # Not fundamental
           'Ask close', # Price related
           'B/A Ratio', # Not fundamental
           'B/A Size', # Not fundamental
           'Bid', # Price related
           'Bid Size', # Not fundamental
           'Bid close', # Price related
           'Change Since Close', # Price related
           'Closing Price', # Price related
           'Day Change $', # Price related
           'Day Change %', # Price related
           'Day High', # Price related
           'Day Low', # Price related
           'Dividend Pay Date', # Datetime
           'Ex-dividend', # Price related
           'Ex-dividend Date', # Datetime
           'Last (size)', # Not fundamental
           'Last (time)', # Datetime
           'Last Trade', # Price related
           'Next Earnings Announcement', # Datetime
           'Prev Close', # Price related
           'Price', # Price related
           "Today's Open", # Price related
           'Volume', # Not fundamental
           'Volume Past Day', # Not fundamental
           'cfra since', # Datetime
           'creditSuisse since', # Datetime
           'ford since', # Datetime
           'marketEdge opinion since', # Datetime
           'marketEdge since', # Datetime
           'newConstructs since', # Datetime
           'researchTeam since', # Datetime
           'theStreet since', # Datetime
           'Annual Dividend Yield' # Duplicate of Annual Dividend %
          ]

X = df.drop(columns=to_drop)
X.info()


analysts = ['cfra', 
            'creditSuisse', 
            'ford', 
            'marketEdge', 
            'Market Edge Opinion:', 
            'marketEdge opinion',
            'newConstructs',
            'researchTeam',
            'theStreet'
           ]
X = X.drop(columns=analysts)

to_drop = ['P/E Ratio (TTM, GAAP)',
           'PEG Ratio (TTM, GAAP)',
           'Price/Earnings (TTM, GAAP)',
          ]
X = X.drop(columns=to_drop)



X.drop(columns=['Market Cap'], inplace=True)

# Check for inf or -inf in data
np.inf in X.values or -np.inf in X.values

X = X.replace([np.inf, -np.inf], np.NaN)


[ticker for ticker in X.index if ticker not in log_close.columns]


X_new = X.drop(index=['AGN', 'ETFC'])


to_fix = ['Price/Book (MRQ)',
          'Price/Cash Flow (TTM)',
          'Price/Earnings (TTM)',
          'Price/Sales (TTM)'
         ]

# The closing prices are along the first row of sp500_close, so we need to isolate the
# row and transpose it, then divide it by each ratio.
for col in to_fix:
    X_new[col.replace('Price/','')] = sp500_close.iloc[0].T.loc[X2.index] / X2[col]

X_new = X_new.drop(columns=to_fix)

# Now a look at what we have
X_new.info()


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Now for some helper functions to make it easier to compare imputation methods
# Note that these are modified versions of the code found in the scikit learn docs
def get_scores_for_imputer(imputer, X_missing, y_missing, regressor, scoring, scale=True):
    if scale:
        steps = [('scaler', StandardScaler()), 
                 ('imputer', imputer), 
                 ('regressor', regressor)]
    else:
        steps = [('imputer', imputer), 
                 ('regressor', regressor)]
        
    estimator = Pipeline(steps=steps)
    impute_scores = cross_val_score(estimator, X_missing, y_missing,
                                    scoring=scoring,
                                    cv=N_SPLITS)
    return impute_scores

def get_impute_zero_score(X_missing, y_missing, regressor, scoring):

    imputer = SimpleImputer(missing_values=np.nan, add_indicator=True,
                            strategy='constant', fill_value=0)
    zero_impute_scores = get_scores_for_imputer(imputer, 
                                                X_missing, 
                                                y_missing, 
                                                regressor, 
                                                scoring)
    return zero_impute_scores.mean(), zero_impute_scores.std()

def get_impute_knn_score(X_missing, y_missing, regressor, scoring, n_neighbors=5):
    imputer = KNNImputer(missing_values=np.nan, 
                         add_indicator=True, 
                         n_neighbors=n_neighbors,
                         weights='distance')
    knn_impute_scores = get_scores_for_imputer(imputer, 
                                               X_missing, 
                                               y_missing, 
                                               regressor, 
                                               scoring)
    return knn_impute_scores.mean(), knn_impute_scores.std()

def get_impute_mean(X_missing, y_missing, regressor, scoring):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean",
                            add_indicator=True)
    mean_impute_scores = get_scores_for_imputer(imputer, 
                                                X_missing, 
                                                y_missing, 
                                                regressor, 
                                                scoring)
    return mean_impute_scores.mean(), mean_impute_scores.std()

def get_impute_iterative(X_missing, y_missing, regressor, scoring, estimator=None,
                         n_nearest_features=None, max_iter=10
                        ):
    sample_posterior = False
    if 'BayesianRidge' in str(estimator):
        sample_posterior = True
    elif estimator is None:
        n_nearest_features = 5
        sample_posterior = True
       
    imputer = IterativeImputer(estimator=estimator,
                               missing_values=np.nan, 
                               max_iter=max_iter,
                               add_indicator=True,
                               random_state=0, 
                               n_nearest_features=n_nearest_features,
                               sample_posterior=sample_posterior
                              )
    iterative_impute_scores = get_scores_for_imputer(imputer,
                                                     X_missing,
                                                     y_missing,
                                                     regressor,
                                                     scoring
                                                    )
    
    return iterative_impute_scores.mean(), iterative_impute_scores.std()
	
	
	def graph_imputer_scores(scores, stds, labels, scoring, regressor=None, iterative=False):
    n_bars = len(scores)
    xval = np.arange(n_bars)
    
    if regressor is not None:
        name = str(regressor).split('(')[0]
        title_string = ' with {}'.format(name)
    else:
        title_string = ''

    x_labels = labels
    colors = ['r', 'g', 'b', 'orange', 'black']

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(111)
    for j in xval:
        ax1.barh(j, scores[j], xerr=stds[j],
                 color=colors[j], alpha=0.6, align='center')
    if not iterative:
        ax1.set_title('Imputation Techniques Comparison{}'.format(title_string))
    else:
        ax1.set_title('Iterative Imputation Estimator Comparison{}'.format(title_string))
    ax1.set_xlim(left=np.min(scores*2) * 0.9,
                 right=np.max(scores*2) * 1.1)
    ax1.set_yticks(xval)
    ax1.set_xlabel(scoring)
    ax1.invert_yaxis()
    ax1.set_yticklabels(x_labels)

    return ax1

# Creating a wrapper function to keep code DRY
def compare_imputer_scores(X, y, regressor, scoring, iterative_estimators=None, max_iter=10):
    # Getting scores for these imputers
    zero_scores = get_impute_zero_score(X, y, regressor, scoring)
    mean_scores = get_impute_mean(X, y, regressor, scoring)
    knn_scores = get_impute_knn_score(X, y, regressor, scoring)
    iter_scores = get_impute_iterative(X, y, regressor, scoring, n_nearest_features=5)
    
    scores = [zero_scores[0], mean_scores[0], knn_scores[0], iter_scores[0]]
    stds = [zero_scores[1], mean_scores[1], knn_scores[1], iter_scores[1]]
    
    # Graphing the scores of the imputation methods above
    labels = ['Zero imputation',
              'Mean Imputation',
              'KNN Imputation',
              'Iterative Imputation']
    
    # Now to run through the estimators for IterativeImputer
    results = []

    if iterative_estimators:
        labels2 = []
        for estimator in iterative_estimators:
            name = str(estimator).split('(')[0]
            print('Imputing with IterativeImputer using {} estimator'.format(name))
            labels2.append(name)
            
            n_nearest_features = None
            if 'BayesianRidge' in str(estimator):
                n_nearest_features = 5
                
            results.append(get_impute_iterative(X,
                                                y,
                                                regressor,
                                                scoring,
                                                estimator,
                                                n_nearest_features=n_nearest_features,
                                                max_iter=max_iter
                                              )
                         )
            
        scores2 = [results[i][0] for i in range(len(results))]
        stds2 = [results[i][1] for i in range(len(results))]
    
    ax1 = graph_imputer_scores(scores, stds, labels, scoring, regressor)
    
    if iterative_estimators:
        ax2 = graph_imputer_scores(scores2, stds2, labels2, scoring, regressor, iterative=True)
        return ax1, ax2
        
    return ax1
	
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor()

# Load some estimators to use with IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

# Must set this global variable for the number of folds used in  cv scoring imputers
N_SPLITS = 5

# Creating a list of these estimators with some set params
estimators = [BayesianRidge(),
              DecisionTreeRegressor(max_features='log2', random_state=0),
              ExtraTreesRegressor(n_estimators=20, max_features='sqrt', random_state=0),
              KNeighborsRegressor(n_neighbors=30)
             ]

# Pass our data, regressor, and estimators into wrapper function to get plots
ax1, ax2 = compare_imputer_scores(X_new, 
                                  y,
                                  regressor, 
                                  scoring='r2', 
                                  iterative_estimators=estimators)
								  
								  
								  
								 
scaler = StandardScaler()

# Adding an indicator to the imputer makes feature importance evaluation very difficult,
# but since we are not going to be worry about that for this task, we can use it
# Remember that we did not remove multicollinearity, so checking feature importances would
# be misleading anyways
imputer = KNNImputer(weights='distance', add_indicator=True)

# Previous testing showed 'huber' was the superior loss function
regressor = GradientBoostingRegressor(loss='huber', random_state=0)

# Make the pipeline
steps = [('scaler', scaler),
         ('imputer', imputer),
         ('regressor', regressor)
        ]
estimator = Pipeline(steps=steps)

# Make the grid of hyperparameters to be tested
param_grid = {'imputer__n_neighbors': [5, 10, 20, 30],
              #'regressor__loss': ['ls', 'lad', 'huber'],
              'regressor__learning_rate': [.01, .1, .5],
              'regressor__n_estimators': [300, 500, 1000],
              'regressor__subsample': [0.7, 0.8, 1.0],
              'regressor__max_depth': [2, 3, 5]
             }

# Instantiate grid search and fit to data. verbose=2 will give you a report on progress,
# which is great when fitting many models. n_jobs=-1 will use all of the computer's CPU cores
grid_search = GridSearchCV(estimator, 
                           param_grid=param_grid, 
                           scoring='r2',
                           cv=5, 
                           n_jobs=-1,
                           verbose=2)
grid_fit = grid_search.fit(X_new, y)


print('Best Model: \n r-squared:', grid_fit.best_score_)


preds = grid_fit.best_estimator_.predict(X_new)
resids = y - preds
sns.distplot(resids)
plt.title('Distribution of Residuals for Asset Pricing Model')
plt.xlabel('Residuals');


targets = log_returns.copy()
targets.loc[targets.log_return > 0, 'class1'] = 1
targets.loc[targets.log_return <= 0, 'class1'] = 0
targets.loc[targets.log_return - targets.log_return.mean() > 0, 'class2'] = 1
targets.loc[targets.log_return - targets.log_return.mean() <= 0, 'class2'] = 0
targets.head()


to_drop = ['% Above Low',
           '% Below High',
           'Annual Dividend $', # Leave this out because % is more generalized
           'Annual Dividend Yield', # This is the same as Annual Dividend %
           'Ask close', # Too many missing values, closing price good enough
           'Bid close', # same as above
           'Change Since Close', # same as above
           'Day Change $', # % is more generalized
           'Dividend Pay Date',
           'Ex-dividend',
           'Ex-dividend Date',
           'Last (time)',
           'Last Trade',
           'Next Earnings Announcement',
           'Price',
           'cfra since',
           'creditSuisse since',
           'ford since',
           'marketEdge opinion since',
           'marketEdge since',
           'newConstructs since',
           'researchTeam since',
           'theStreet since'
          ]

X = df.drop(columns=to_drop)
X.info()



X = X.drop(columns=['B/A Size',
                    'P/E Ratio (TTM, GAAP)',
                    'Market Edge Opinion:',
                    'marketEdge opinion',
                    'Volume Past Day'
                   ])

X[['52-Wk Range', 'Market Cap', 'creditSuisse', 'researchTeam', 'theStreet']].head()



X['52-Wk Range'] = X['52-Wk Range'].map(lambda x: x.strip('[]').replace("'",'').replace(' ',''))

# Now apply lambda functions to split the values and turn them into floats in two separate columns
X['52-Wk Low'] = X['52-Wk Range'].map(lambda x: float(x.replace(',','').split('-')[0]))
X['52-Wk High'] = X['52-Wk Range'].map(lambda x: float(x.replace(',','').split('-')[1]))

# Check the results to make sure it has worked
X[['52-Wk Low', '52-Wk High']].head()


X.drop('52-Wk Range', axis=1, inplace=True)

# Isolate the last character in the column and get unique values
X['Market Cap'].map(lambda x: x[-1]).unique()


# Building function to map to column
def fix_market_cap(x):
    if x[-1] == 'M':
        x = x.strip('M')
        x = float(x) * 1e6
    elif x[-1] == 'B':
        x = x.strip('B')
        x = float(x) * 1e9
    elif x[-1] == 'T':
        x = x.strip('T')
        x = float(x) * 1e12
    else:
        raise ValueError('Invalid input')
        
    return x
  
# Generate new column by mapping the function to old column
X['marketCap'] = X['Market Cap'].map(fix_market_cap)

# Test to make sure this has worked
X['marketCap'].head()

# Dropping old Market Cap column
X.drop(columns=['Market Cap'], inplace=True)

# Finding tickers not present in log_returns
[colname for colname in X.index if colname not in log_returns.index]

# Dropping tickers that were acquired during time period
X_new = X.drop(index=['AGN', 'ETFC'])

# Replace the infs with nans for consistent missing data format
X_new = X_new.replace([np.inf, -np.inf], np.NaN)


# get_dummies will automatically find remaining object dtype features
X_new = pd.get_dummies(X_new, drop_first=True)

# A look at our new data frame
X_new.info()


# Establish the continuous target variable
y = targets.log_return

# Import and instantiate the random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state=0)

# Generate list of estimators to test in IterativeImputer
estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='auto', random_state=123),
    ExtraTreesRegressor(n_estimators=10, max_features='sqrt', random_state=123),
    KNeighborsRegressor(n_neighbors=15)
]

# Run our imputation method comparison function
ax1, ax2 = compare_imputer_scores(X_new, y, regressor, 'r2', estimators)



from xgboost import XGBClassifier

clf = XGBClassifier()
y = targets.class1

estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=123),
    ExtraTreesRegressor(n_estimators=10, max_features='sqrt', random_state=123),
    KNeighborsRegressor(n_neighbors=15)
]

ax1, ax2 = compare_imputer_scores(X_reduced, y, clf, 'roc_auc', estimators)


# Establish the target variable and generate a training and holdout set.
y = targets.class1
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Create estimator pipeline to be optimized
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean', fill_value=0)
clf = XGBClassifier(random_state=0, n_jobs=-1)
steps = [('scaler', scaler),
         ('imputer', imputer),
         ('clf', clf)
        ]
estimator = Pipeline(steps=steps)

# Make parameter grid to search
param_grid = {'clf__n_estimators': [100, 300, 500, 1000],
              'clf__max_depth': [2, 3, 5, 7],
              'clf__learning_rate': [.001, .01, .1, .5],
              #'clf__booster': ['gbtree', 'gblinear', 'dart'],
              #'clf__reg_alpha': [0, 1],
              'clf__reg_lambda': [0.25, .5, 0.75, 1],
              'clf__subsample': [.8, 1],
              'clf__colsample_bytree': [.6, .8, 1],
              'clf__colsample_bylevel': [.6, .8, 1]
             }

# Make grid search and fit
grid_search = GridSearchCV(estimator, 
                           param_grid=param_grid, 
                           scoring='roc_auc', 
                           cv=5,
                           n_jobs=-2,
                           verbose=2
                          )
gridfit_class1_xgb = grid_search.fit(X_train, y_train)


print('Best CV roc_auc score:', gridfit_class1_xgb.best_score_)
gridfit_class1_xgb.best_params_


# Import classifier, instantiate, and set target variable
from xgboost import XGBRFClassifier
clf = XGBRFClassifier(random_state=0, n_jobs=-1)
y = targets.class2

# Establish estimators to be used with IterativeImputer
estimators = [BayesianRidge(),
              DecisionTreeRegressor(max_features='sqrt', random_state=123),
              ExtraTreesRegressor(max_features='sqrt', n_estimators=10, random_state=123),
              KNeighborsRegressor(n_neighbors=15)]

# Use our wrapper function to compare imputation methods
ax1, ax2 = compare_imputer_scores(X_reduced, y, clf, 'roc_auc', estimators)




# Generate training and holdout (testing) set
y = targets.class2
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Create pipeline with scaler, imputer, and classifier
scaler = StandardScaler()
imputer = SimpleImputer()
clf = XGBRFClassifier(n_jobs=-1, random_state=0)

steps = [('scaler', scaler),
         ('imputer', imputer),
         ('clf', clf)
        ]
estimator = Pipeline(steps=steps)

# Create parameter grid for Grid Search
param_grid = {'clf__max_depth': [3, 5, 7],
              'clf__learning_rate': [.001, .01, .1, .5],
              'clf__n_estimators': [100, 300, 500],
              'clf__subsample': [0.8, 1.0],
              'clf__colsample_bytree': [0.6, 0.8, 1.0],
              'clf__colsample_bylevel': [0.8, 1.0],
              'clf__colsample_bynode': [0.8, 1.0],
              'clf__reg_lambda': [.25, .5, .75, 1]
             }

# Create Grid Search and fit to train data
grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid,
                           scoring='roc_auc',
                           n_jobs=-2,
                           cv=5,
                           verbose=2
                          )
gridfit_class2_xgbrf = grid_search.fit(X_train, y_train)



# Let's see what the best score was
print('Best CV training score:', gridfit_class2_xgbrf.best_score_)

# Now a look at the best parameters from the grid search
gridfit_class2_xgbrf.best_params_