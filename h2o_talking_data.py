# Last amended: 30th Oct, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\talkingdata
# Ref: https://www.kaggle.com/nanomathias/h2o-distributed-random-forest-starter
#      https://www.kdnuggets.com/2020/01/h2o-framework-machine-learning.html
# Data source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection
# Objectives:
#           i)   Ad-tracking fraud challenge
#           ii)  Learning to use bigdata
#           iii) Learn to work with h2o
#           iv)  Feature engineering
#                of categorical variables
#                using aggregation.
#

# 1.0 Call libraries
%reset -f
import numpy as np
import pandas as pd
import random
import os,time

# 1.1 Change folder to where data is:
os.chdir("C:\\Users\\Administrator\\OneDrive\\Documents\\talkingdata")

# 1.2 Start h2o
import h2o
h2o.init()

# 2.0 Read a fraction of data
tr_f = "train.csv.zip"
total_lines = 184903891
read_lines = 300000    # Reduce it if less RAM


# 2.1 Read randomly 'p' fraction of files
#     Ref: https://stackoverflow.com/a/48589768

p = read_lines/total_lines  # fraction of lines to read

# 2.1.1 How to pick up random rows from hard-disk
#       without first loading the complete file in RAM
#       Toss a coin:
#           At each row, toss a biased-coin: 60%->Head, 40%->tail
#           If tail comes, select the row else not.
#           Toss a coin: random.random()
#           Head occurs if value > 0.6 else it is tail
#
#       We do not toss the coin for header row. Keep the header
start = time.time()
train = pd.read_csv(
                tr_f,
                header=0,   # First row is header-row
                            # 'and' operator returns True if both values are True
                            #  random.random() returns values between (0,1)
                            #  No of rows skipped will be around 60% of total
                skiprows=lambda i: (i >0 ) and (np.random.random() > p)    # (i>0) implies skip first header row
                )
end = time.time()
(end-start)/60      # 4 minutes

# 2.1.2
train.head()

# 2.1.3
train.dtypes    # ip int64,app int64,device int64,os int64,channel int64,
                # click_time object,attributed_time object,is_attributed int64

# 2.1.4 Let us see distribution of NULLS
train.isnull().sum()      # 299084 nulls in attributed_time

# 2.1.5
train.shape    # (299847, 8)

# 2.2 Transform pandas DataFrame to H2OFrame
train_h2o_types = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'time', 'time', 'numeric']
train_h2o = h2o.H2OFrame(train,column_types = train_h2o_types )

# 2.3 Check
type(train_h2o)    # h2o.frame.H2OFrame

# 2.3.1
train_h2o.types    # {'ip': 'int','app': 'int','device': 'int','os': 'int',
                   # 'channel': 'int','click_time': 'time','attributed_time': 'time',
                   # 'is_attributed': 'int'}


# 2.4 Explore data:
#     Note that NaN in pandas are replaced
#     by empty spaces.

train_h2o.head()




train_h2o.shape    # (299847, 8)

# 2.4.1 Check NAs
#       (Note: Without return_frame=True, one
#       gets total results.)
train_h2o.isna().sum(axis = 0, return_frame=True)

# 2.4.2 Check if data is balanced?
train_h2o['is_attributed'].sum()    # 732.0
train_h2o['is_attributed'].mean()   # 0.002433599632965301

# 3.0 We need to change some
#     columns to factor types

category_cols = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']

# 3.1 The columns to be used during training
train_h2o.columns   # Existing columns

# 3.1.1 New columns are: 'day', 'hour', 'qty1', 'qty2','qty3'
#       'qty1', 'qty2','qty3' are aggregated counts

X1 = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour', 'qty1', 'qty2', 'qty3']
X2 = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour', 'qty1', 'qty2']
X3 = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour', 'qty1']

y = 'is_attributed'

# 4.0 Convert some column-types to factors
for c in category_cols:
    if c in train.columns:
        train_h2o[c] = train_h2o[c].asfactor()

# 4.1 Categorical columns become of type 'enum':
train_h2o.types      #  {'ip': 'enum','app': 'enum','device': 'enum',
                 #   'os': 'enum','channel': 'enum','click_time': 'time',
                 #   'attributed_time': 'time','is_attributed': 'enum'}


# 4.2 Extract time data:
train_h2o['day']  = train_h2o['click_time'].day()
train_h2o['hour'] = train_h2o['click_time'].hour()

# 4.3 Let us now count how many times
#     the combination of values in
#     ['ip','day','hour'] occur together
#     Here are the steps:
# 4.3.1 (Not sure if the 'grouped' formation occurs)
grouped = train_h2o[['ip','day','hour']].group_by(by=['ip','day','hour'])
grouped.count()
grouped.count().get_frame()  # A new column by name of nrow is created

# 4.3.2 (Better avoid 'grouped')
#       Aggregate categories
dx1 = train_h2o[['ip','day','hour']].group_by(by=['ip','day','hour']).count().get_frame()
dx2 = train_h2o[['ip','day']].group_by(by=['ip','day']).count().get_frame()
dx3 = train_h2o[['ip']].group_by(by=['ip']).count().get_frame()

# 4.3.3
dx1.shape      # (242488, 4)  (587680, 4)
dx2.shape      # (103459, 3)
dx3.shape      # (55477, 2)
dx1.head()
dx1.sort(by = 'nrow', ascending = False).head()

# 4.4
dx1.columns
dx2.columns
dx3.columns
train_h2o.columns # ['ip','app','device','os','channel','click_time',
              #  'attributed_time','is_attributed','day','hour']


# 5.0 Merge this aggregated data with
#     main dataset. That is merge: train and dx

df_h2o = train_h2o.merge(                          # train is designated as 'x':
                     dx1,
                     all_x = True,              # So all of train. Left join
                     by_x=['ip','day','hour'],  # Key-columns from left-hand side
                     by_y=['ip','day','hour']   # Key-columns from right-hand side
                    )

# 5.1 Rename column 'nrow'
df_h2o = df_h2o.rename({'nrow': 'qty1'})

# 5.2 Similarly more merger
df_h2o = df_h2o.merge(
                     dx2,
                     all_x = True,
                     by_x=['ip','day'],
                     by_y=['ip','day']
                    )
# 5.2.1
df_h2o = df_h2o.rename({'nrow': 'qty2'})

# 5.3 More merger
df_h2o = df_h2o.merge(
                     dx3,
                     all_x = True,
                     by_x=['ip'],
                     by_y=['ip']
                    )
# 5.3.1
df_h2o = df_h2o.rename({'nrow': 'qty3'})

# 5.4 Normalize columns: qty1, qty2, qty3
df_h2o['qty1'] = df_h2o['qty1']/df_h2o.nrow
df_h2o['qty2'] = df_h2o['qty2']/df_h2o.nrow
df_h2o['qty3'] = df_h2o['qty3']/df_h2o.nrow


# 5.4.1
df_h2o.shape     # (299705, 13)  (899323, 12)
df_h2o.columns   # ['ip','hour','day','app','device','os',
             #  'channel','click_time','attributed_time',
             # 'is_attributed','qty1']

# 5.4.2 Just have a look at resulting DataFrame
df_h2o.sort(by = 'qty1', ascending= False).head()


#######################
#   MODEL TRAINING    #
#######################

# 6.0 Create classifier, balance classes
clf = h2o.estimators.random_forest.H2ORandomForestEstimator(
                                                            ntrees = 200,     # Default 50
                                                            balance_classes=True,
                                                            nfolds = 3
                                                            )


# 6.1 Create classifier, do not balance classes
clf1 = h2o.estimators.random_forest.H2ORandomForestEstimator(
                                                             ntrees = 200,
                                                             balance_classes=False,
                                                             nfolds = 3
                                                             )

# 6.2 Want to know more about H2ORandomForestEstimator?
#     Here is help
help(clf)


# 7.0 Train classifier: Ist
start = time.time()
clf.train(
          X1, y,      # Can try with X2 or X3 also
          training_frame = df_h2o
          )
end = time.time()
(end-start)/60

# 7.1 Train classifier: IInd
start = time.time()
clf1.train(
           X1, y,
           training_frame = df_h2o
           )
end= time.time()
(end-start)/60

# 7.2 Check model performance. Ist model:
clf.model_performance()
clf.cross_validation_metrics_summary

# 7.3 Model performance: IInd model
clf1.model_performance()
clf1.cross_validation_metrics_summary

##################################################
