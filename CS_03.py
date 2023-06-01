#!/usr/bin/env python
# coding: utf-8

# # Case-Study Title: Using Classification algorithms in Financial markets (Stock Market Prediction)
# ###### Data Analysis methodology: CRISP-DM
# ###### Dataset: S&P-500 (The Standard and Poor's 500) Index Timeseries daily data from 2019 to 2023
# ###### Case Goal: Create an automatic financial trading algorithm for S&P-500 index (Algorithmic Trading)

# # Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import mplfinance as fplt
import yfinance as yf
import ta
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow import keras


# In[2]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None)


# # Read Data directly from Yahoo Finance API

# In[3]:


data = yf.download(['^GSPC'], start = '2019-01-01', end = '2023-12-31')  # data resolution is daily


# In[4]:


data.shape  # 1109 records, 6 variables


# # Business Understanding
#  * know business process and issues
#  * know the context of the problem
#  * know the order of numbers in the business

# # Data Understanding
# ## Data Inspection (Data Understanding from Free Perspective)
# ### Dataset variables definition

# In[5]:


data.columns


# * **Index**:        the Timestamp of record (Date)
# * **Open**:         the price of asset at market opening (Start of day)
# * **High**:         the maximum price of asset in a day
# * **Low**:          the minimum price of asset in a day
# * **Close**:        the price of asset at market closing (End of day)
# * **Adj Close**:
# * **Volume**:       the trading volume of asset in a day

# In[6]:


type(data)


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.info()


# In[10]:


# Do we have any NA in our Variables?
data.isna().sum()

# We have no MV problem in this dataset


# In[11]:


# Check for abnormality in data
data.describe(include='all')


# ### Data Visualization of Financial Data

# In[12]:


data.index  # index of our dataset is Timestamp (Date)


# In[13]:


data.loc['2019-01-02']  # retrieve data for day '2019-01-02'


# In[14]:


data.loc['2019', 'Close'].plot()  # plot 'Close' data for year '2019'


# In[15]:


data.loc['2019-06', 'Close'].plot()  # plot 'Close' data for month 'June 2019'


# In[16]:


# Candlestick plot
fplt.plot(data.loc['2019-06'],
          type = 'candle',
          style = 'classic',
          volume = True,
          show_nontrading = True,
          figratio = (10, 6),
          title = 'S&P-500, June 2019')


# # Data PreProcessing
# ## Prepare some Technical Analysis Indicators to use as Features in our ML models

# Simple Moving Average (SMA)

# In[17]:


sma5 = ta.trend.sma_indicator(data['Close'], 5)  # Moving Average 5
sma20 = ta.trend.sma_indicator(data['Close'], 20)  # Moving Average 30


# In[18]:


sma20.head(20)  # first 19 data is NaN (Missing Value)


# In[19]:


sma20.tail()


# Exponential Moving Average (EMA)

# In[20]:


ema5 = ta.trend.ema_indicator(data['Close'], 5)
ema20 = ta.trend.ema_indicator(data['Close'], 20)


# In[21]:


ema20.head(20)


# In[22]:


ema20.tail()


# Relative Strength Index (RSI)

# In[23]:


rsi3 = ta.momentum.rsi(data['Open'], 3)


# In[24]:


rsi3.head()


# In[25]:


rsi3.tail()


# In[26]:


data['sma5'] = sma5
data['sma20'] = sma20
data['ema5'] = ema5
data['ema20'] = ema20
data['rsi3'] = rsi3


# In[27]:


data.head()


# Calculate Daily Return = (today's Close price - yesterday's Close price) / yesterday's Close price

# In[28]:


data['d_return'] = 0


# In[29]:


for i in range(1, data.shape[0]):
    data.iloc[i, 11] = data.iloc[i, 3] / data.iloc[i - 1, 3] - 1
    
data.head()


# Calculate Volume Change

# In[30]:


data['volume_change'] = 0

for i in range(1, data.shape[0]):
    if data.iloc[i - 1, 5] != 0:
        data.iloc[i, 12] = data.iloc[i, 5] / data.iloc[i - 1, 5] - 1
    else:
        data.iloc[i, 12] = 1
    
data.head()


# In[31]:


data.tail()


# Plot Daily Return

# In[32]:


data['d_return'].plot()  # very noisy around 0

# do some Timeseries Analysis here: is this a White Noise? is this a Random Walk model?


# In[33]:


# Plot Histogram of Daily Return
sns.histplot(data['d_return'], 
             stat = 'probability', 
             kde = True, 
             alpha = 0.7, 
             color = 'green',
             bins = 50)

# usually, distribution of 'Return' in Finance Markets is t-student


# In[34]:


data['d_return'].describe()


# In[35]:


data.tail()


# Add Lags: probably, some few past days can be a good predictor for today!

# In[36]:


data['sma5_lag1'] = data['sma5'].shift(1)
data['sma20_lag1'] = data['sma20'].shift(1)
data['ema5_lag1'] = data['ema5'].shift(1)
data['ema20_lag1'] = data['ema20'].shift(1)
data['rsi3_lag1'] = data['rsi3'].shift(1)
data['h_lag1'] = data['High'].shift(1)
data['l_lag1'] = data['Low'].shift(1)

# Daily Return Lag: add lag 1 to 5
data['r_lag1'] = data['d_return'].shift(1)
data['r_lag2'] = data['d_return'].shift(2)
data['r_lag3'] = data['d_return'].shift(3)
data['r_lag4'] = data['d_return'].shift(4)
data['r_lag5'] = data['d_return'].shift(5)

# Volume Change Lag: add lag 1 to 5
data['v_lag1'] = data['volume_change'].shift(1)
data['v_lag2'] = data['volume_change'].shift(2)
data['v_lag3'] = data['volume_change'].shift(3)
data['v_lag4'] = data['volume_change'].shift(4)
data['v_lag5'] = data['volume_change'].shift(5)

data.head()


# In[37]:


data.tail()


# In[38]:


# add market trend (label our data based-on Daily Return)
conditions = [data['d_return'] <= 0, data['d_return'] > 0]
values = [0, 1]  # directions of Market
data['trend'] = np.select(conditions, values)
data.head(20)


# > We want to predict **'trend'** column: Predict Market Direction

# In[39]:


# Remove first 20 rows (rows with NaN and 0.0000) from dataset
data.drop(index = data.index[range(20)], inplace = True)
data.head()  # Prepared dataset


# In[40]:


data.isna().sum()  # check NA


# ## Correlation Analysis

# In[41]:


# correlation table between 'd_return' and continuous variables
corr_table = round(data[['d_return',
                         'sma5_lag1',
                         'sma20_lag1',
                         'ema5_lag1',
                         'ema20_lag1',
                         'rsi3_lag1',
                         'h_lag1',
                         'l_lag1',
                         'r_lag1', 'r_lag2', 'r_lag3', 'r_lag4', 'r_lag5',
                         'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5'
                        ]].corr(method = 'pearson'), 2)
corr_table


# > **'r_lag1'** and **'r_lag2'** have good correlation

# In[42]:


sns.heatmap(corr_table, annot = False)


# In[43]:


# Scatter Plot (between 'd_return' and other continuous variables 2 by 2)
var_ind = list(range(13,30))
plot = plt.figure(figsize = (26, 20))
plot.subplots_adjust(hspace = 0.5, wspace = 0.5)
for i in range(1, 18):
    a = plot.add_subplot(6, 3, i)
    a.scatter(x = data.iloc[:, var_ind[i - 1]],
              y = data.iloc[:, 11],
              alpha = 0.5)
    a.title.set_text('d_return vs. ' + data.columns[var_ind[i - 1]])


# ## Divide Dataset into Train and Test and Real

# In[44]:


train = data.loc['2019':'2020']
train


# In[45]:


test = data.loc['2021']
test


# In[46]:


real = data.loc['2022':'2023']
real


# # Modeling
# ## Model 1: Logistic Regression

# In[47]:


# Define the features set X (features-matrix)
X_train = train[['sma5_lag1',
                 'sma20_lag1',
                 'ema5_lag1',
                 'ema20_lag1',
                 'rsi3_lag1',
                 'h_lag1',
                 'l_lag1',
                 'r_lag1', 'r_lag2', 'r_lag3', 'r_lag4', 'r_lag5',
                 'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5'
                        ]]
X_train = sm.add_constant(X_train)  # adding a constant column (a column of 1)

# Define response variable (response-matrix)
y_train = train['trend']


# In[48]:


X_train.head()


# In[49]:


y_train.head()


# In[50]:


model_lr = sm.Logit(y_train, X_train).fit()
print(model_lr.summary())


# Prediction on train

# In[51]:


y_prob_train = model_lr.predict(X_train)
y_prob_train  # probabilities


# In[52]:


y_pred_train = [1 if _ > 0.48 else 0 for _ in y_prob_train]  # compare the probabilities with Threshold 48%
y_pred_train


# In[53]:


# Accuracy (how percent True prediction?)
from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_pred_train) * 100


# In[54]:


# Confusion Matrix for train dataset
from sklearn.metrics import confusion_matrix
confusion_matrix_lr = confusion_matrix(y_train, y_pred_train)
print(confusion_matrix_lr)

#                prediction_0  prediction_1
# observation_0      35            169
# observation_1      17            264


# Prediction on test

# In[55]:


# Define the features set X (features-matrix)
X_test = test[['sma5_lag1',
                 'sma20_lag1',
                 'ema5_lag1',
                 'ema20_lag1',
                 'rsi3_lag1',
                 'h_lag1',
                 'l_lag1',
                 'r_lag1', 'r_lag2', 'r_lag3', 'r_lag4', 'r_lag5',
                 'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5'
                        ]]
X_test = sm.add_constant(X_test)  # adding a constant column (a column of 1)
X_test.head()


# In[56]:


# Define response variable (response-matrix)
y_test = test['trend']
y_test.head()


# In[57]:


y_prob_test = model_lr.predict(X_test)
y_prob_test


# In[58]:


y_pred_test = pd.Series([1 if _ > 0.48 else 0 for _ in y_prob_test], index = y_prob_test.index)
y_pred_test


# In[59]:


# Accuracy (how percent True prediction?)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_test) * 100


# In[60]:


# Confusion Matrix for test dataset
from sklearn.metrics import confusion_matrix
confusion_matrix_lr = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix_lr)

#                prediction_0  prediction_1
# observation_0      42            67
# observation_1      52            91


# Model Evaluation (based-on Confusion Matrix)

# In[61]:


# Positive Predictive Value: if model says the market is Bullish, how percent really will be Bullish?
lr_ppv = 91 / (91 + 67)

lr_ppv * 100


# In[62]:


# Negative Predictive Value: if model says the market is Bearish, how percent really will be Bearish?
lr_npv = 42 / (42 + 52)

lr_npv * 100


# ## Model 2: Random Forest

# In[63]:


# Define the features set X (features-matrix)
X_train = train[['sma5_lag1',
                 'sma20_lag1',
                 'ema5_lag1',
                 'ema20_lag1',
                 'rsi3_lag1',
                 'h_lag1',
                 'l_lag1',
                 'r_lag1', 'r_lag2', 'r_lag3', 'r_lag4', 'r_lag5',
                 'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5'
                        ]]
X_train.head()  # we have not a constant column


# In[64]:


y_train.head()


# use k-fold Cross-Validation for tunning model's hyper-parameters

# In[65]:


# create hyper-parameters grid
import itertools
max_features = [2, 4, 8, 16]
max_depth = [1, 3, 5]
min_samples_leaf = [5, 10, 15]
ccp_alpha = [0.001, 0.01, 0.1]
grid = list(itertools.product(max_features, max_depth, min_samples_leaf, ccp_alpha))
grid = pd.DataFrame(data = grid, index = range(1, 109), columns = ['max_features', 'max_depth', 'min_samples_leaf', 'ccp_alpha'])
grid


# In[66]:


# 10-fold Cross-Validation approach

k = 10  # create 10 folds
np.random.seed(123)
folds = np.random.randint(low = 1, high = k + 1, size = X_train.shape[0])
folds


# In[67]:


cv_accr = pd.DataFrame(index = range(1, k + 1), columns = range(1, 109))
cv_accr


# In[68]:


from sklearn.ensemble import RandomForestClassifier
for i in range(1, grid.shape[0] + 1):
    for j in range(1, k + 1):
        cls_rf = RandomForestClassifier(max_features = grid.loc[i, 'max_features'],
                                        max_depth = grid.loc[i, 'max_depth'],
                                        min_samples_leaf = grid.loc[i, 'min_samples_leaf'],
                                        ccp_alpha = grid.loc[i, 'ccp_alpha'],
                                        n_estimators = 500)
        rf_res = cls_rf.fit(X_train.iloc[folds != j, :], y_train[folds != j])
        pred = pd.Series(rf_res.predict(X_train.iloc[folds == j, :]), index = y_train.index[folds == j])
        cv_accr.iloc[j - 1, i - 1] = accuracy_score(y_train[folds == j], pred) * 100
        
cv_accr  # Cross-Validation Accuracy for 108 different RandomForest models


# In[69]:


cv_accr.mean(axis = 0)


# In[70]:


cv_accr.mean(axis = 0).max()


# In[71]:


cv_accr.mean(axis = 0).argmax() + 1


# In[72]:


# the Best model's hyper-parameters
grid.iloc[cv_accr.mean(axis = 0).argmax()]


# In[73]:


model_rf = RandomForestClassifier(max_features = 16,
                                  max_depth = 3,
                                  min_samples_leaf = 10,
                                  ccp_alpha = 0.01,
                                  random_state = 123,
                                  n_estimators = 1000
                                 ).fit(X_train, y_train)


# Importance of Variables: the percentage of increasing MSE if we remove the variable from the Tree

# In[74]:


importance = pd.DataFrame({'Importance': model_rf.feature_importances_ * 100}, index = X_train.columns)
importance.sort_values(by = 'Importance', axis = 0, ascending = True).plot(kind = 'barh', color = 'r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# Prediction on test

# In[75]:


X_test = test[['sma5_lag1',
                 'sma20_lag1',
                 'ema5_lag1',
                 'ema20_lag1',
                 'rsi3_lag1',
                 'h_lag1',
                 'l_lag1',
                 'r_lag1', 'r_lag2', 'r_lag3', 'r_lag4', 'r_lag5',
                 'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5'
                        ]]
X_test.head()


# In[76]:


y_pred_rf = pd.Series(model_rf.predict(X_test), index = y_test.index)
y_pred_rf


# In[77]:


# Accuracy (how percent True prediction?)
accuracy_score(y_test, y_pred_rf) * 100


# In[78]:


# Confusion Matrix for test dataset
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(confusion_matrix_rf)

#                prediction_0  prediction_1
# observation_0      18            91
# observation_1      20            123


# Model Evaluation (based-on Confusion Matrix)

# In[79]:


# Positive Predictive Value: if model says the market is Bullish, how percent really will be Bullish?
rf_ppv = 123 / (123 + 91)

rf_ppv * 100


# In[80]:


# Negative Predictive Value: if model says the market is Bearish, how percent really will be Bearish?
rf_npv = 18 / (18 + 20)

rf_npv * 100


# ## Model 3: Naive Bayes Classifier

# In[81]:


from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB().fit(X_train, y_train)


# Prediction on test

# In[82]:


y_pred_nb = pd.Series(model_nb.predict(X_test), index = y_test.index)
y_pred_nb


# In[83]:


# Accuracy (how percent True prediction?)
accuracy_score(y_test, y_pred_nb) * 100


# In[84]:


# Confusion Matrix for test dataset
confusion_matrix_nb = confusion_matrix(y_test, y_pred_nb)
print(confusion_matrix_nb)

#                prediction_0  prediction_1
# observation_0      108            1
# observation_1      139            4


# Model Evaluation (based-on Confusion Matrix)

# In[85]:


# Positive Predictive Value: if model says the market is Bullish, how percent really will be Bullish?
nb_ppv = 4 / (4 + 1)

nb_ppv * 100


# In[86]:


# Negative Predictive Value: if model says the market is Bearish, how percent really will be Bearish?
nb_npv = 108 / (108 + 139)

nb_npv * 100


# ## Model 4: Linear Discriminant Analysis (LDA)

# In[87]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model_lda = LinearDiscriminantAnalysis().fit(X_train, y_train)


# Prediction on test

# In[88]:


y_pred_lda = pd.Series(model_lda.predict(X_test), index = y_test.index)
y_pred_lda


# In[89]:


# Accuracy (how percent True prediction?)
accuracy_score(y_test, y_pred_lda) * 100


# In[90]:


# Confusion Matrix for test dataset
confusion_matrix_lda = confusion_matrix(y_test, y_pred_lda)
print(confusion_matrix_lda)

#                prediction_0  prediction_1
# observation_0      50            59
# observation_1      66            77


# Model Evaluation (based-on Confusion Matrix)

# In[91]:


# Positive Predictive Value: if model says the market is Bullish, how percent really will be Bullish?
lda_ppv = 77 / (77 + 59)

lda_ppv * 100


# In[92]:


# Negative Predictive Value: if model says the market is Bearish, how percent really will be Bearish?
lda_npv = 50 / (50 + 66)

lda_npv * 100


# ## Model 5: Support Vector Machines (SVM)

# use k-fold Cross-Validation for tunning model's hyper-parameters

# In[93]:


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
degree_grid = [2, 3, 4]
acc_scores = []
for d in degree_grid:
    svc = SVC(kernel = 'poly', degree = d, C = 1E6)
    scores = cross_val_score(svc, X_train, y_train, cv = 5, scoring = 'accuracy')
    acc_scores.append(scores.mean())
print(acc_scores)


# In[94]:


# Plot Cross-Validation results for SVM
plt.plot(degree_grid, acc_scores)
plt.xticks(degree_grid)
plt.xlabel('Degree')
plt.ylabel('Cross-Validated Accuracy')


# In[95]:


model_svc = SVC(kernel = 'poly', degree = 2, C = 1E6).fit(X_train, y_train)


# Prediction on test

# In[96]:


y_pred_svc = model_svc.predict(X_test)
y_pred_svc


# In[97]:


# Accuracy (how percent True prediction?)
accuracy_score(y_test, y_pred_svc) * 100


# In[98]:


# Confusion Matrix for test dataset
confusion_matrix_svm = confusion_matrix(y_test, y_pred_svc)
print(confusion_matrix_svm)

#                prediction_0  prediction_1
# observation_0      22            87
# observation_1      24            119


# Model Evaluation (based-on Confusion Matrix)

# In[99]:


# Positive Predictive Value: if model says the market is Bullish, how percent really will be Bullish?
svm_ppv = 119 / (119 + 87)

svm_ppv * 100


# In[100]:


# Negative Predictive Value: if model says the market is Bearish, how percent really will be Bearish?
svm_npv = 22 / (22 + 24)

svm_npv * 100


# ## Model 6: Artificial Neural Networks (ANN)

# In[417]:


# Define the features set X (features-matrix)
X_train = train[['sma5_lag1',
                 'sma20_lag1',
                 'ema5_lag1',
                 'ema20_lag1',
                 'rsi3_lag1',
                 'h_lag1',
                 'l_lag1',
                 'r_lag1', 'r_lag2', 'r_lag3', 'r_lag4', 'r_lag5',
                 'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5'
                        ]]

# Define response variable (response-matrix)
y_train = train['trend']


# In[418]:


X_train.head()


# In[419]:


y_train.head()


# In[420]:


# Min-Max Normalization to scale the train data
from sklearn.preprocessing import MinMaxScaler
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_train_scaled.head()


# In[421]:


X_train_scaled.describe()

# 'min' of all columns should be 0
# 'max' of all columns should be 1


# In[422]:


y_train.shape  # this is not appropriate format for ANN


# In[423]:


# One-Hot Encoding y_train classes
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes = 2)


# In[424]:


y_train[0]


# In[425]:


y_train.shape


# In[426]:


X_train_scaled.shape


# In[427]:


# Define the model architecture
model_ann = keras.Sequential()  # define general structure of model
model_ann.add(keras.layers.Dense(8, input_dim = 17, activation = 'relu'))  # input layer
model_ann.add(keras.layers.Dense(4, activation = 'relu'))  # hidden layer
model_ann.add(keras.layers.Dense(2, activation = 'softmax'))  # output layer
model_ann.summary()  # structure of model


# In[428]:


# Configure the model
model_ann.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[429]:


# Fit the model on train data
model_ann.fit(X_train_scaled, y_train, epochs = 200)


# Prediction on test

# In[430]:


# Define the features set X (features-matrix)
X_test = test[['sma5_lag1',
                 'sma20_lag1',
                 'ema5_lag1',
                 'ema20_lag1',
                 'rsi3_lag1',
                 'h_lag1',
                 'l_lag1',
                 'r_lag1', 'r_lag2', 'r_lag3', 'r_lag4', 'r_lag5',
                 'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5'
                        ]]

# Define response variable (response-matrix)
y_test = test['trend']


# In[431]:


X_test.head()


# In[432]:


y_test.head()


# In[433]:


# Min-Max Normalization to scale the test data
X_test_scaled = MinMaxScaler().fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)
X_test_scaled.head()


# In[434]:


y_prob_test = model.predict(X_test_scaled)
y_prob_test


# In[435]:


y_pred_test = pd.Series([np.argmax(row) for row in y_prob_test], index = test.index)
y_pred_test


# In[436]:


# Accuracy (how percent True prediction?)
accuracy_score(y_test, y_pred_test) * 100


# In[438]:


# Confusion Matrix for test dataset
confusion_matrix_ann = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix_ann)

#                prediction_0  prediction_1
# observation_0      17            92
# observation_1      18            125


# Model Evaluation (based-on Confusion Matrix)

# In[439]:


# Positive Predictive Value: if model says the market is Bullish, how percent really will be Bullish?
ann_ppv = 125 / (125 + 92)

ann_ppv * 100


# In[440]:


# Negative Predictive Value: if model says the market is Bearish, how percent really will be Bearish?
ann_npv = 17 / (17 + 18)

ann_npv * 100


# ## Summary of different models results on test data (Prediction Accuracy)
# 
# > Logistic Regression: 52.77, PPV = 57.59, NPV = 44.68
# 
# > Random Forest: 55.95, PPV = 57.47, NPV = 47.36
# 
# > Naive Bayes Classifier: 44.44, **PPV = 80.00**, NPV = 43.72
# 
# > Linear Discriminant Analysis: 50.39, PPV = 56.61, NPV = 43.10
# 
# > Support Vector Machines: 55.95, PPV = 57.76, NPV = 47.82
# 
# > **Artificial Neural Networks**: **56.34**, PPV = 57.60, **NPV = 48.57**

# # Strategy Implementation

# In[441]:


# Define the features set X (features-matrix)
X_real = real[['sma5_lag1',
                 'sma20_lag1',
                 'ema5_lag1',
                 'ema20_lag1',
                 'rsi3_lag1',
                 'h_lag1',
                 'l_lag1',
                 'r_lag1', 'r_lag2', 'r_lag3', 'r_lag4', 'r_lag5',
                 'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5'
                        ]]
X_real = sm.add_constant(X_real)  # adding a constant column (a column of 1)
X_real.head()


# In[442]:


# Min-Max Normalization to scale the test data
X_real_scaled = MinMaxScaler().fit_transform(X_real.iloc[:, 1:])
X_real_scaled = pd.DataFrame(X_real_scaled, columns = X_real.columns[1:], index = X_real.index)
X_real_scaled.head()


# In[443]:


# Define response variable (response-matrix)
y_real = real['trend']
y_real.head()


# Prediction on real

# In[444]:


y_prob_real_lr = model_lr.predict(X_real)
y_pred_real_lr = pd.Series([1 if _ > 0.48 else 0 for _ in y_prob_real_lr], index = y_real.index)
y_pred_real_rf = pd.Series(model_rf.predict(X_real.iloc[:, 1:]), index = y_real.index)
y_pred_real_nb = pd.Series(model_nb.predict(X_real.iloc[:, 1:]), index = y_real.index)
y_pred_real_lda = pd.Series(model_lda.predict(X_real.iloc[:, 1:]), index = y_real.index)
y_pred_real_svm = pd.Series(model_svc.predict(X_real.iloc[:, 1:]), index = y_real.index)
y_prob_real_ann = model_ann.predict(X_real_scaled)
y_pred_real_ann = pd.Series([np.argmax(row) for row in y_prob_real_ann], index = y_real.index)

y_pred_lr_por = np.array([lr_ppv if _ == 1 else -lr_npv for _ in y_pred_real_lr])
y_pred_rf_por = np.array([rf_ppv if _ == 1 else -rf_npv for _ in y_pred_real_rf])
y_pred_nb_por = np.array([nb_ppv if _ == 1 else -nb_npv for _ in y_pred_real_nb])
y_pred_lda_por = np.array([lda_ppv if _ == 1 else -lda_npv for _ in y_pred_real_lda])
y_pred_svm_por = np.array([svm_ppv if _ == 1 else -svm_npv for _ in y_pred_real_svm])
y_pred_ann_por = np.array([ann_ppv if _ == 1 else -ann_npv for _ in y_pred_real_ann])
y_pred_real_por = y_pred_lr_por + y_pred_rf_por + y_pred_nb_por + y_pred_lda_por + y_pred_svm_por + y_pred_ann_por
y_pred_real_pred = pd.Series([1 if _ > 0 else 0 for _ in y_pred_real_por], index = y_real.index)


# In[445]:


# Accuracy (how percent True prediction?)
print('Logistic Regression: {:.2f}'.format(accuracy_score(y_real, y_pred_real_lr) * 100),
      'Random Forest: {:.2f}'.format(accuracy_score(y_real, y_pred_real_rf) * 100),
      'Naive Bayes: {:.2f}'.format(accuracy_score(y_real, y_pred_real_nb) * 100),
      'LDA: {:.2f}'.format(accuracy_score(y_real, y_pred_real_lda) * 100),
      'SVM: {:.2f}'.format(accuracy_score(y_real, y_pred_real_svm) * 100),
      'ANN: {:.2f}'.format(accuracy_score(y_real, y_pred_real_ann) * 100),
      'Voting: {:.2f}'.format(accuracy_score(y_real, y_pred_real_pred) * 100),
      sep = '\n')


# # Simulate Real Trading

# In[446]:


real['pred'] = y_pred_real_pred  # model prediction
real.head()


# In[447]:


real['balance'] = 0  # Balance over time (shows result of our trading in each day)
real.head()


# In[448]:


# Initial Deposit in begining of 2022: $1000
real.loc['2022-01-03','balance'] = 1000
real.head()


# In[449]:


# Trade simulation (the algorithm decides to Buy or Sell based-on model prediction)
for i in range(1, real.shape[0]):
    if real.iloc[i, 31] == 1:
        real.iloc[i, 32] = real.iloc[i - 1, 32] * real.iloc[i, 3] / real.iloc[i, 0]
    
    if real.iloc[i, 31] == 0:
        real.iloc[i, 32] = real.iloc[i - 1, 32] * real.iloc[i, 0] / real.iloc[i, 3]
        
real.head()


# In[450]:


# Plot 'balance' during time
real['balance'].plot()
plt.axhline(1000, color = 'red', linewidth = 2, linestyle = '--')

