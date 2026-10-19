"""
In this assignment you will learn to program a nested cross-validation in order to
calculate a non-biased score of grid_search.best_estimator_
The alternative, which is the score provided by grid_search.best_score_, is biased, is too optimistic.
To calculate a non-biased score of grid_search.best_estimator_ 
you will use a nested cross-validation = cross_val_score(grid_search, ...)
In the end, you will continue to use grid_search.best_estimator_ 
(you will not replace that estimator) but
you will use nested cross-validation to obtain a non-biased metric of its performance, which
is a metric of the performace of the grid_search optimization procedure as a whole.
To do this assignment you must read:
https://github.com/amueller/introduction_to_ml_with_python/blob/main/05-model-evaluation-and-improvement.ipynb 
https://archive.is/zLiGA
specifically: 05-model-evaluation-and-improvement.ipynb, "Nested cross-validation"
This pipeline does 2 selections: models, fetaures and regularization  parameters.


the output file of your code should be as in outputfile.txt
See INSTRUCTIONS below
"""



import warnings
warnings.simplefilter('ignore')

import functions as ff
import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import sys

np.random.seed(1) #to fix the results
rs = 2

# Correct annualization for ~8 bars per trading day (H3)
ANNUALIZATION = 252 * 8  # 2016

file_path = 'outputfile.txt'
sys.stdout = open(file_path, "w")

#we define this ourselves to correct the division by zero error
def single_autocorr(series, lag):
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')


df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)

#save the open for white reality check
openp = df['<OPEN>'].copy() #for the case we want to enter trades at the open

#buld window features:
for n in list(range(1,21)):
    name = 'ret' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n) #for trading with open

#buld window volatility features:
for n in list(range(1,5)):
    name = 'ret_squared' + str(n)
    df[name] = df["<OPEN>"].pct_change(periods=n) #for trading with open
    df[name] = df[name]*df[name]

#new window features
df['autocorr1']=df['ret1'].rolling(50).apply(lambda s:single_autocorr(s, lag=1)).fillna(0)
lg=3
df['vratio']=df['<OPEN>'].rolling(100*lg).apply(lambda s: ff.vratio(np.log(s.values), lag=lg, cor='hom')[0]).fillna(0)

#build date-time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values

#build target assuming we know today's open
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade immediately after the open


#transform the target
df['retFut1_categ'] = np.where((df['retFut1'] > 0), 1, 0)

#Since we are trading right after the open,
#we only know yesterday's  high low close volume spread etc.
df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

#select the features (by dropping)
cols_to_drop = ["<OPEN>","<HIGH>","<LOW>","<CLOSE>","<TICKVOL>","<VOL>","<SPREAD>"]  #optional
df_filtered = df.drop(cols_to_drop, axis=1)

#distribute the df data into X inputs and y target
X = df_filtered.drop(['retFut1', 'retFut1_categ'], axis=1)
y = df_filtered[['retFut1_categ']]

#select the samples
x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]

df_train = df_filtered.iloc[0:10000]
df_test = df_filtered.iloc[10000:12000]


##########################################################################################################################

#set up the grid search and fit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
import phik
from phik.report import plot_correlation_matrix
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif #ANOVA F-value between label/feature for classification tasks. (linear)
from sklearn.feature_selection import mutual_info_classif #Mutual information for a discrete target. (non linear)
from scipy.special import ndtr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import detrendPrice
import WhiteRealityCheckFor1
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import matthews_corrcoef


#myscorer = None #use default accuracy score
#myscorer = "balanced_accuracy"
myscorer = "matthews_corrcoef" #phi-k is a version of this


percentile = 95
#selector = SelectPercentile(score_func= phi_k_select, percentile=percentile)
#selector = SelectPercentile(score_func= f_classif, percentile=percentile)
selector = SelectPercentile(score_func= mutual_info_classif, percentile=percentile)

numeric_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()),
    ('selector', selector)])
categorical_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

print(x_train.dtypes)
numeric_features_ix = x_train.select_dtypes(include=['float64']).columns
categorical_features_ix = x_train.select_dtypes(include=['integer']).columns

#Note: transformer 3-element tuples can be: ('name', function or pipeline, column_number_list or column_index)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_sub_pipeline, numeric_features_ix),
        ('cat', categorical_sub_pipeline, categorical_features_ix)], remainder='passthrough')


default_classifier = LogisticRegression(max_iter=1000, solver='liblinear')

pipe = Pipeline(steps=[('preprocessor', preprocessor),('classifier', default_classifier)])

c_rs = np.logspace(3, -4, num=20, endpoint = True)
#penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)
p_rs= ["l1", "l2"]

"""
INSTRUCTIONS

Replace ##### with the required code.

This pipeline is selecting the best hyper parameters, the best input features and
the best classifier from two potential candidates: 
a LogisticRegression (instantiated with (max_iter=1000, solver='liblinear')) and 
a LinearSVD (instantiated with (max_iter=10000, loss='squared_hinge', dual=False))
Both classifiers obtain the list of regularization candidates from the same list, c_rs
Both classifiers obtain the list of penalty candidates from the same list, p_rs

"""

#two separate dicts (one per candidate classifier) so RandomizedSearchCV never has to sample an
#invalid (penalty, loss, dual) combination for LinearSVC by mixing it with LogisticRegression's grid
param_grid = [
    {'classifier': [LogisticRegression(max_iter=1000, solver='liblinear')],
     'classifier__C': c_rs,
     'classifier__penalty': p_rs},
    {'classifier': [LinearSVC(max_iter=10000, loss='squared_hinge', dual=False)],
     'classifier__C': c_rs,
     'classifier__penalty': p_rs},
]

#when using smoother, use TimesSeriesSplit
#split = 5
split = TimeSeriesSplit(n_splits=5, max_train_size=2000) #fixed size window
#split = TimeSeriesSplit(n_splits=5)

grid_search = RandomizedSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True, random_state=rs)
#grid_search = GridSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)

grid_search.fit(x_train, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters : {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score : {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_logistic_svc_reg.csv")

"""
INSTRUCTIONS

Replace ##### with the required code.
Use as reference:
https://github.com/amueller/introduction_to_ml_with_python/blob/main/05-model-evaluation-and-improvement.ipynb 
https://archive.is/zLiGA
specifically: 05-model-evaluation-and-improvement.ipynb, "Nested cross-validation"

Note that the pipeline in this homework script is more complex than the pipeline in the github notebook:
In the github notebook the pipeline just selects the best regularization hyper-parameter for the model.
In this homework script, the pipeline selects the best model, the best input-features and the best hyper parameter for the model.

"""

##########################################################################################################################

#Nested cross-validation (see 05-model-evaluation-and-improvement.ipynb, "Nested cross-validation")
#grid_search.best_score_ is optimistic: the same inner CV folds are used both to pick the
#hyperparameters and to report the score. Nested CV wraps the whole search (grid_search) inside
#an outer split, so no outer test fold ever leaks into hyperparameter selection.
outer_split = TimeSeriesSplit(n_splits=5, max_train_size=2000) #fixed size window, same shape as the inner split

nested_scores = cross_val_score(grid_search, x_train, y_train.values.ravel(), cv=outer_split, scoring=myscorer)

print("Nested CV scores : {}".format(nested_scores))
print("Nested CV mean score : {:.4f} (std {:.4f})".format(nested_scores.mean(), nested_scores.std()))
print("Non-nested best cross-validation score (optimistic) : {:.4f}".format(grid_search.best_score_))

#Per-classifier nested CV: nested_scores above mixes whichever classifier the inner search
#happened to pick per outer fold, so a "LinearSVC beat LogisticRegression" read of best_params_
#could just be outer-fold noise. Rerun nested CV once per classifier, each restricted to only its
#own param dict, so the two classifiers can be compared on equal footing.
per_classifier_grids = {
    'LogisticRegression': param_grid[0],
    'LinearSVC': param_grid[1],
}

for clf_name, clf_param_grid in per_classifier_grids.items():
    clf_search = RandomizedSearchCV(pipe, [clf_param_grid], cv=split, scoring=myscorer, random_state=rs)
    clf_nested_scores = cross_val_score(clf_search, x_train, y_train.values.ravel(), cv=outer_split, scoring=myscorer)
    print("{} nested CV scores : {}".format(clf_name, clf_nested_scores))
    print("{} nested CV mean score : {:.4f} (std {:.4f})".format(clf_name, clf_nested_scores.mean(), clf_nested_scores.std()))

#final_model = grid_search.fit(x_train,y_train).best_estimator_  #final_model = best_model above

#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
#positions = np.where(best_model.predict(x_train)> 0,1,-1 )
positions = np.where(grid_search.predict(x_train)> 0,1,-1 ) #POSITIONS

dailyRet = pd.Series(positions).fillna(0).values * df_train.retFut1 #for trading right after the open

dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1


plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated LogisticRegression on currency: train set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("TrainCumulative"))


cagr = (1 + cumret[-1]) ** (ANNUALIZATION / len(cumret)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret)
ratio = (ANNUALIZATION ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
print (('In-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))

# Test set
# Make "predictions" on test set (out-of-sample)

#positions2 = np.where(best_model.predict(x_test)> 0,1,-1 )
positions2 = np.where(grid_search.predict(x_test)> 0,1,-1 ) #POSITIONS


dailyRet2 = pd.Series(positions2).fillna(0).values * df_test.retFut1 #for trading right after the open
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
title = 'Cross-validated LogisticRegression on currency: test set'
plt.title(title)
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("TestCumulative"))

#metrics
accuracy_score = accuracy_score(y_test.values.ravel(), grid_search.predict(x_test))

#If this figure does not plot correctly select the lines and press F9 again
arr1 = y_test.values.ravel()
arr2 = grid_search.predict(x_test)
dfc = pd.DataFrame({'y_true': arr1, 'y_pred': arr2})
phi_k_corr = dfc.phik_matrix(interval_cols=[]).iloc[1,0]
significance_overview = dfc.significance_matrix(interval_cols=[])
phi_k_sig  = dfc.significance_matrix(interval_cols=[]).iloc[1,0]
phi_k_p_val = 1 - ndtr(phi_k_sig) 
plot_correlation_matrix(significance_overview.fillna(0).values, 
                        x_labels=significance_overview.columns, 
                        y_labels=significance_overview.index, 
                        vmin=-5, vmax=5, title="Significance of the coefficients", 
                        usetex=False, fontsize_factor=1.5, figsize=(7, 5))
plt.tight_layout()
#plt.show()
plt.savefig(r'Results\%s.png' %("PhikSignificance"))

cagr = (1 + cumret2[-1]) ** (ANNUALIZATION / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (ANNUALIZATION ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  phi_k_corr={:0.6} phi_k_p_val={:0.6}  accuracy_score={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, phi_k_corr, phi_k_p_val, accuracy_score))


#plot the residuals
true_y = y_test.values.ravel()
pred_y = grid_search.predict(x_test)
residuals = np.subtract(true_y, pred_y)

from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
axes[0].set_title('Residual Distribution')
axes[0].legend()
plot_acf(residuals, lags=10, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout();
#plt.show()
plt.savefig(r'Results\%s.png' %("ResidualDistribution"))
plt.close("all")

#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb['lb_pvalue'].values)

#Detrending Prices and Returns and white reality check
detrended_open = detrendPrice.detrendPrice(openp[10000:12000])
detrended_retFut1 = detrended_open.pct_change(periods=1).shift(-1).fillna(0)
detrended_syst_rets = detrended_retFut1 * pd.Series(positions2).fillna(0)
WhiteRealityCheckFor1.bootstrap(detrended_syst_rets)
plt.show()

#recover the actual post-transform feature names (post-imputer/scaler/SelectPercentile/one-hot)
#straight from the fitted pipeline, instead of fabricating generic numeric_features_N/dummies_N labels
column_names = best_model.named_steps['preprocessor'].get_feature_names_out()

##plot the coefficients
importance = pd.DataFrame(zip(best_model[1].coef_.ravel().tolist(), column_names))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
plt.tight_layout()
#plt.show()
plt.savefig(r'Results\%s.png' %("Coefficients"), bbox_inches='tight')