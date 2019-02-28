
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_rows = 10
# pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 600
from tqdm import tqdm
import gc

from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.decomposition import PCA

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential, Model

from keras.layers import Input, Embedding, Dense, Activation, Dropout, Flatten

from keras import regularizers 

import keras

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')


# # 辅助函数的构造方法

# In[2]:


def init():
    np.random.seed = 0   
init()

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def smape2D(y_true, y_pred):
    return smape(np.ravel(y_true), np.ravel(y_pred))
    
def smape_mask(y_true, y_pred, threshold):
    denominator = (np.abs(y_true) + np.abs(y_pred)) 
    diff = np.abs(y_true - y_pred) 
    diff[denominator == 0] = 0.0
    
    return diff <= (threshold / 2.0) * denominator


# # 原始数据读入与分析

# In[3]:


max_size = 181 # number of days in 2015 with 3 days before end

offset = 1/2

train_all = pd.read_csv("../input/train_2.csv")
train_all.head()


# In[4]:


print(f"the data shape is:{train_all.shape}!!")
print(f"the start time is:{train_all.columns[0]}, and the end time is:{train_all.columns[-1]}!!")
print(f"the time range day is:{(pd.to_datetime(train_all.columns[-1]) - pd.to_datetime(train_all.columns[1])).days}!!")


# # 训练数据的缺失率的计算方法

# In[5]:


train_all.count(axis=1)


# In[6]:


raito_train = train_all.set_index('Page')
raito_train['loss_ratio'] = raito_train.count(axis=1) / raito_train.shape[1]


# In[7]:


raito_train.loss_ratio.plot.hist()
raito_train.loss_ratio.describe()


# # 基本的label周期性分析

# In[8]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import pandas  as pd
import numpy as np
from sqlalchemy import create_engine

import plotly.offline as offline
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns; sns.set(color_codes=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt

from sklearn import  linear_model

get_ipython().run_line_magic('matplotlib', 'inline')
offline.init_notebook_mode(connected=False)

import os
os.listdir('./')


# In[9]:


raito_train.drop('loss_ratio', axis=1, inplace=True)


# In[13]:


index_sample = np.random.choice(raito_train.index.tolist())
plot_data = raito_train[raito_train.index == index_sample]

trace0 = go.Scatter(x=plot_data.columns.values.tolist(),
                    y=plot_data.iloc[:,:].values.ravel().tolist(),
                    name='dispatch',
                    mode='lines+markers'
)

layout = go.Layout(
                    xaxis=dict(title='date'),
                    yaxis=dict(title='values')
)
data = go.Data([trace0])
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, show_link=False)
plot_data


# # 训练数据的基本处理方法

# In[14]:


all_page = train_all.Page.copy()
train_key = train_all[['Page']].copy()
train_all = train_all.iloc[:,1:] * offset 
train_all.head()


# In[15]:


def get_date_index(date, train_all=train_all):
    for idx, c in enumerate(train_all.columns):
        if date == c:
            break
    if idx == len(train_all.columns):
        return None
    return idx


# In[16]:


get_date_index('2016-09-10')


# # train and test数据的切分方法

# # 其中trains为7段窗口为181的训练数据
# ## 而train_all为训练集从后向前的181窗口的数据
# ## 而test为时间窗口为63的7段窗口的数据
# ## 而test_all为需要预测的未来2个月的数据

# In[17]:


trains = []
tests = []
train_end = get_date_index('2016-09-10') + 1
test_start = get_date_index('2016-09-13')

for i in range(-3,4):
    train = train_all.iloc[ : , (train_end - max_size + i) : train_end + i].copy().astype('float32')
    test = train_all.iloc[:, test_start + i : (63 + test_start) + i].copy().astype('float32')
    train = train.iloc[:,::-1].copy().astype('float32')
    trains.append(train)
    tests.append(test)

train_all = train_all.iloc[:,-(max_size):].astype('float32')
train_all = train_all.iloc[:,::-1].copy().astype('float32')

test_3_date = tests[3].columns


# In[18]:


train_all.head()


# In[19]:


train_all.shape


# In[20]:


trains[0]


# In[21]:


tests[0]


# # 网页信息的提取

# In[22]:


train_key


# In[23]:


all_page


# In[24]:


data = [page.split('_') for page in tqdm(train_key.Page)]

access = ['_'.join(page[-2:]) for page in data]

site = [page[-3] for page in data]

page = ['_'.join(page[:-3]) for page in data]
page[:2]

train_key['PageTitle'] = page
train_key['Site'] = site
train_key['AccessAgent'] = access
train_key.head()


# # 对数化训练数据

# In[25]:


train_norms = [np.log1p(train).astype('float32') for train in trains]
train_norms[3].head()


# In[26]:


train_all_norm = np.log1p(train_all).astype('float32')
train_all_norm.head()


# # 对数据进行周期化操作

# In[27]:


tests[3]


# In[28]:


for i,test in enumerate(tests):
    first_day = i-2 # 2016-09-13 is a Tuesday
    test_columns_date = list(test.columns)
    test_columns_code = ['w%d_d%d' % (i // 7, (first_day + i) % 7) for i in range(63)]
    test.columns = test_columns_code

tests[3].head()


# In[29]:


for test in tests:
    test.fillna(0, inplace=True)

    test['Page'] = all_page
    test.sort_values(by='Page', inplace=True)
    test.reset_index(drop=True, inplace=True)


# In[30]:


tests = [test.merge(train_key, how='left', on='Page', copy=False) for test in tests]

tests[3].head()


# # 预测数据格式化

# In[31]:


test_all_id = pd.read_csv('../input/key_2.csv')

test_all_id['Date'] = [page[-10:] for page in tqdm(test_all_id.Page)]
test_all_id['Page'] = [page[:-11] for page in tqdm(test_all_id.Page)]
test_all_id.head()


# In[32]:


print(f"the test_id shape is:{test_all_id.shape}!!!")


# In[33]:


test_all = test_all_id.drop('Id', axis=1)
test_all['Visits_true'] = np.NaN

test_all.Visits_true = test_all.Visits_true * offset
test_all = test_all.pivot(index='Page', columns='Date', values='Visits_true').astype('float32').reset_index()

test_all['2017-11-14'] = np.NaN
test_all.sort_values(by='Page', inplace=True)
test_all.reset_index(drop=True, inplace=True)

test_all.head()


# In[34]:


test_all.shape


# In[35]:


test_all_columns_date = list(test_all.columns[1:])
first_day = 2 # 2017-13-09 is a Wednesday
test_all_columns_code = ['w%d_d%d' % (i // 7, (first_day + i) % 7) for i in range(63)]
cols = ['Page']
cols.extend(test_all_columns_code) 
test_all.columns = cols
test_all.head()


# In[36]:


test_all.shape


# In[37]:


test_all = test_all.merge(train_key, how='left', on='Page')
test_all.head()


# In[38]:


y_cols = test.columns[:63]
y_cols


# In[39]:


for test in tests:
    test.reset_index(inplace=True)
test_all = test_all.reset_index()


# In[40]:


test_all.head()


# In[41]:


test


# In[42]:


test = pd.concat(tests[2:5], axis=0).reset_index(drop=True)
test.shape


# In[43]:


test_all.shape


# In[44]:


test


# In[45]:


test_all = test_all[test.columns].copy()
print(f"the test_all shape is :{test_all.shape}!!!!")
test_all.head()


# In[46]:


test_all.head()


# In[47]:


train_cols = ['d_%d' % i for i in range(train_norms[0].shape[1])]
len(train_cols)


# In[48]:


for train_norm in train_norms:
    train_norm.columns = train_cols
train_all_norm.columns = train_cols
train_norm = pd.concat(train_norms[2:5], axis=0).reset_index(drop=True)
train_norm.shape


# In[49]:


train_norm


# In[50]:


train_norm.iloc[:,::-1]


# In[51]:


train_norm


# # 时序shfit特征的转换与提取

# ## 提取shit为1和7的特征

# In[52]:


train_norm_diff = train_norm - train_norm.shift(-1, axis=1)
train_norm_diff.head()

train_all_norm_diff = train_all_norm - train_all_norm.shift(-1, axis=1)
train_all_norm_diff.head()

train_norm_diff7 = train_norm - train_norm.shift(-7, axis=1)
train_norm_diff7.head()

train_all_norm_diff7 = train_all_norm - train_all_norm.shift(-7, axis=1)
train_all_norm_diff7.head()


# ## 提取与滑动窗口为7的中位数的差值的特征

# In[53]:


train_norm = train_norm.iloc[:,::-1]
train_norm_diff7m = train_norm - train_norm.rolling(window=7, axis=1).median()
train_norm = train_norm.iloc[:,::-1]
train_norm_diff7m = train_norm_diff7m.iloc[:,::-1]
train_norm_diff7m.head()


# In[54]:


train_all_norm = train_all_norm.iloc[:,::-1]
train_all_norm_diff7m = train_all_norm - train_all_norm.rolling(window=7, axis=1).median()
train_all_norm = train_all_norm.iloc[:,::-1]
train_all_norm_diff7m = train_all_norm_diff7m.iloc[:,::-1]
train_all_norm_diff7m.head()


# In[55]:


sites = train_key.Site.unique()
sites


# In[75]:


accesses = train_key.AccessAgent.unique()
accesses


# ## 对网点信息进行编码操作

# In[56]:


test_site = pd.factorize(test.Site)[0]
test['Site_label'] = test_site
test_all['Site_label'] = test_site[:test_all.shape[0]]

test_access = pd.factorize(test.AccessAgent)[0]
test['Access_label'] = test_access
test_all['Access_label'] = test_access[:test_all.shape[0]]


# In[57]:


test.shape


# In[58]:


test_all.shape


# In[59]:


test0 = test.copy()
test_all0 = test_all.copy()


# In[60]:


print(f"the predict columns length is :{len(y_cols)}!!!!!")
y_cols


# In[61]:


y_norm_cols = [c+'_norm' for c in y_cols]
y_pred_cols = [c+'_pred' for c in y_cols]


# In[62]:


train_norm_diff7


# In[63]:


train_key


# In[64]:


train_norm


# In[65]:


train_norm.median(axis=1)


# In[66]:


test


# In[67]:


y_norm_cols


# In[68]:


y_cols


# ## 提取训练数据的特征，分别为各种周期的数据的中位数以及差值等

# In[69]:


# all visits is median
def add_median(test, train, train_diff, train_diff7, train_diff7m,
               train_key, periods, max_periods, first_train_weekday):
    train =  train.iloc[:,:7*max_periods]
    
    df = train_key[['Page']].copy()
    df['AllVisits'] = train.median(axis=1).fillna(0)
    test = test.merge(df, how='left', on='Page', copy=False)
    test.AllVisits = test.AllVisits.fillna(0).astype('float32')
    
    for site in sites:
        test[site] = (1 * (test.Site == site)).astype('float32')
    
    for access in accesses:
        test[access] = (1 * (test.AccessAgent == access)).astype('float32')

    for (w1, w2) in periods:
        
        df = train_key[['Page']].copy()
        c = 'median_%d_%d' % (w1, w2)
        cm = 'mean_%d_%d' % (w1, w2)
        cmax = 'max_%d_%d' % (w1, w2)
        cd = 'median_diff_%d_%d' % (w1, w2)
        cd7 = 'median_diff7_%d_%d' % (w1, w2)
        cd7m = 'median_diff7m_%d_%d' % (w1, w2)
        cd7mm = 'mean_diff7m_%d_%d' % (w1, w2)
        df[c] = train.iloc[:,7*w1:7*w2].median(axis=1, skipna=True) 
        df[cm] = train.iloc[:,7*w1:7*w2].mean(axis=1, skipna=True) 
        df[cmax] = train.iloc[:,7*w1:7*w2].max(axis=1, skipna=True) 
        df[cd] = train_diff.iloc[:,7*w1:7*w2].median(axis=1, skipna=True) 
        df[cd7] = train_diff7.iloc[:,7*w1:7*w2].median(axis=1, skipna=True) 
        df[cd7m] = train_diff7m.iloc[:,7*w1:7*w2].median(axis=1, skipna=True) 
        df[cd7mm] = train_diff7m.iloc[:,7*w1:7*w2].mean(axis=1, skipna=True) 
        test = test.merge(df, how='left', on='Page', copy=False)
        test[c] = (test[c] - test.AllVisits).fillna(0).astype('float32')
        test[cm] = (test[cm] - test.AllVisits).fillna(0).astype('float32')
        test[cmax] = (test[cmax] - test.AllVisits).fillna(0).astype('float32')
        test[cd] = (test[cd] ).fillna(0).astype('float32')
        test[cd7] = (test[cd7] ).fillna(0).astype('float32')
        test[cd7m] = (test[cd7m] ).fillna(0).astype('float32')
        test[cd7mm] = (test[cd7mm] ).fillna(0).astype('float32')

    for c_norm, c in zip(y_norm_cols, y_cols):
        test[c_norm] = (np.log1p(test[c]) - test.AllVisits).astype('float32')

    gc.collect()

    return test


# In[76]:


max_periods = 16
periods = [(0,1), (1,2), (2,3), (3,4), 
           (4,5), (5,6), (6,7), (7,8),
           (0,2), (2,4),(4,6),(6,8),
           (0,4),(4,8),(8,12),(12,16),
           (0,8), (8,16), (0,12), 
           (0,16), 
          ]


site_cols = list(sites)
access_cols = list(accesses)

test, test_all = test0.copy(), test_all0.copy()

for c in y_pred_cols:
    test[c] = np.NaN
    test_all[c] = np.NaN


# In[77]:


sites


# In[79]:


access_cols


# In[80]:


test_all


# In[81]:


test1 = add_median(test, train_norm, train_norm_diff, train_norm_diff7, train_norm_diff7m, 
                   train_key, periods, max_periods, 3)

test_all1 = add_median(test_all, train_all_norm, train_all_norm_diff, train_all_norm_diff7, train_all_norm_diff7m, 
                       train_key, periods, max_periods, 5)


# In[82]:


test1


# In[83]:


test_all1


# ## 列名的设计

# In[84]:


num_cols = (['median_%d_%d' % (w1,w2) for (w1,w2) in periods])
num_cols.extend(['mean_%d_%d' % (w1,w2) for (w1,w2) in periods])
num_cols.extend(['max_%d_%d' % (w1,w2) for (w1,w2) in periods])
num_cols.extend(['median_diff_%d_%d' % (w1,w2) for (w1,w2) in periods])
num_cols.extend(['median_diff7m_%d_%d' % (w1,w2) for (w1,w2) in periods])
num_cols.extend(['mean_diff7m_%d_%d' % (w1,w2) for (w1,w2) in periods])


# # 模型设计方法

# ## 定义loss函数

# In[85]:


import keras.backend as K

def smape_error(y_true, y_pred):
    return K.mean(K.clip(K.abs(y_pred - y_true),  0.0, 1.0), axis=-1)


# ## 模型的结构设计

# In[86]:


def get_model(input_dim, num_sites, num_accesses, output_dim):
    
    dropout = 0.5
    regularizer = 0.00004
    main_input = Input(shape=(input_dim,), dtype='float32', name='main_input')
    site_input = Input(shape=(num_sites,), dtype='float32', name='site_input')
    access_input = Input(shape=(num_accesses,), dtype='float32', name='access_input')
    
    
    x0 = keras.layers.concatenate([main_input, site_input, access_input])
    x = Dense(200, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x0)
    x = Dropout(dropout)(x)
    x = keras.layers.concatenate([x0, x])
    x = Dense(200, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
                           gamma_regularizer=regularizers.l2(regularizer)
                          )(x)
    x = Dropout(dropout)(x)
    x = Dense(100, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)

    x = Dense(200, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)
    x = Dense(output_dim, activation='linear', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)

    model =  Model(inputs=[main_input, site_input, access_input], outputs=[x])
    model.compile(loss=smape_error, optimizer='adam')
    return model


# In[87]:


print(test1.shape)
test1.head()


# In[88]:


print(test_all1.shape)
test_all1.head()


# ## 模型的数据准备阶段

# In[89]:


group = pd.factorize(test1.Page)[0]

n_bag = 20
kf = GroupKFold(n_bag)
batch_size=4096

#print('week:', week)
test2 = test1
test_all2 = test_all1
X, Xs, Xa, y = test2[num_cols].values, test2[site_cols].values, test2[access_cols].values, test2[y_norm_cols].values
X_all, Xs_all, Xa_all, y_all = test_all2[num_cols].values, test_all2[site_cols].values, test_all2[access_cols].values, test_all2[y_norm_cols].fillna(0).values

y_true = test2[y_cols]
y_all_true = test_all2[y_cols]

models = [get_model(len(num_cols), len(site_cols), len(access_cols), len(y_cols)) for bag in range(n_bag)]

print('offset:', offset)
print('batch size:', batch_size)


best_score = 100
best_all_score = 100

save_pred = 0
saved_pred_all = 0


# In[90]:


print(len(X.shape), len(Xs.shape), len(Xa.shape), len(y.shape))
print(len(X_all.shape), len(Xs_all.shape), len(Xa_all.shape), len(y_all.shape))


# In[91]:


print(len(y_norm_cols), len(num_cols))


# In[92]:


X


# In[93]:


y_true


# In[94]:


y_all_true


# In[95]:


models[0].summary()


# In[96]:


models[1].summary()


# In[97]:


print(X.shape, Xs.shape, Xa.shape, y.shape)
print(X_all.shape, Xs_all.shape, Xa_all.shape, y_all.shape)


# In[98]:


offset


# ## 分批训练阶段

# In[ ]:


for n_epoch in range(10, 201, 10):
    print('************** start %d epochs **************************' % n_epoch)

    y_pred0 = np.zeros((y.shape[0], y.shape[1]))
    y_all_pred0 = np.zeros((n_bag, y_all.shape[0], y_all.shape[1]))
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, group)):
        print('train fold', fold, end=' ')    
        model = models[fold]
        X_train, Xs_train, Xa_train, y_train = X[train_idx,:], Xs[train_idx,:], Xa[train_idx,:], y[train_idx,:]
        X_test, Xs_test, Xa_test, y_test = X[test_idx,:], Xs[test_idx,:], Xa[test_idx,:], y[test_idx,:]

        model.fit([ X_train, Xs_train, Xa_train],  y_train, 
                  epochs=10, batch_size=batch_size, verbose=0, shuffle=True, 
                  #validation_data=([X_test, Xs_test, Xa_test],  y_test)
                 )
        y_pred = model.predict([ X_test, Xs_test, Xa_test], batch_size=batch_size)
        y_all_pred = model.predict([X_all, Xs_all, Xa_all], batch_size=batch_size)

        y_pred0[test_idx,:] = y_pred
        y_all_pred0[fold,:,:]  = y_all_pred

        y_pred += test2.AllVisits.values[test_idx].reshape((-1,1))
        y_pred = np.expm1(y_pred)
        y_pred[y_pred < 0.5 * offset] = 0
        res = smape2D(test2[y_cols].values[test_idx, :], y_pred)
        y_pred = offset*((y_pred / offset).round())
        res_round = smape2D(test2[y_cols].values[test_idx, :], y_pred)

        y_all_pred += test_all2.AllVisits.values.reshape((-1,1))
        y_all_pred = np.expm1(y_all_pred)
        y_all_pred[y_all_pred < 0.5 * offset] = 0
        res_all = smape2D(test_all2[y_cols], y_all_pred)
        y_all_pred = offset*((y_all_pred / offset).round())
        res_all_round = smape2D(test_all2[y_cols], y_all_pred)
        print('smape train: %0.5f' % res, 'round: %0.5f' % res_round,
              '     smape LB: %0.5f' % res_all, 'round: %0.5f' % res_all_round)

    #y_pred0  = np.nanmedian(y_pred0, axis=0)
    y_all_pred0  = np.nanmedian(y_all_pred0, axis=0)

    y_pred0  += test2.AllVisits.values.reshape((-1,1))
    y_pred0 = np.expm1(y_pred0)
    y_pred0[y_pred0 < 0.5 * offset] = 0
    res = smape2D(y_true, y_pred0)
    print('smape train: %0.5f' % res, end=' ')
    y_pred0 = offset*((y_pred0 / offset).round())
    res_round = smape2D(y_true, y_pred0)
    print('round: %0.5f' % res_round)

    y_all_pred0 += test_all2.AllVisits.values.reshape((-1,1))
    y_all_pred0 = np.expm1(y_all_pred0)
    y_all_pred0[y_all_pred0 < 0.5 * offset] = 0
    #y_all_pred0 = y_all_pred0.round()
    res_all = smape2D(y_all_true, y_all_pred0)
    print('     smape LB: %0.5f' % res_all, end=' ')
    y_all_pred0 = offset*((y_all_pred0 / offset).round())
    res_all_round = smape2D(y_all_true, y_all_pred0)
    print('round: %0.5f' % res_all_round, end=' ')
    if res_round < best_score:
        print('saving')
        best_score = res_round
        best_all_score = res_all_round
        test.loc[:, y_pred_cols] = y_pred0
        test_all.loc[:, y_pred_cols] = y_all_pred0
    else:
        print()
    print('*************** end %d epochs **************************' % n_epoch)
print('best saved LB score:', best_all_score)


# ## 数据预测与结果保存

# In[ ]:


filename = 'keras_kf_12_stage2_sept_10'

test_all_columns_save = [c+'_pred' for c in test_all_columns_code]
test_all_columns_save.append('Page')
test_all_save = test_all[test_all_columns_save]

test_all_save.columns = test_all_columns_date+['Page']

test_all_save.to_csv('../data/%s_test_all_save.csv' % filename, index=False)

test_all_save_columns = test_all_columns_date[:-1]+['Page']
test_all_save = test_all_save[test_all_save_columns]

test_all_save = pd.melt(test_all_save, id_vars=['Page'], var_name='Date', value_name='Visits')

test_all_sub = test_all_id.merge(test_all_save, how='left', on=['Page','Date'])

test_all_sub.Visits = (test_all_sub.Visits / offset).round()
#print('%.5f' % smape(test_all_sub.Visits_true, test_all_sub.Visits))

test_all_sub_sorted = test_all_sub[['Id', 'Visits']].sort_values(by='Id')

test_all_sub_sorted[['Id', 'Visits']].to_csv('../submissions/%s_test_sorted.csv' % filename, index=False)

#print('%.5f' % smape(test_all_sub.Visits_true, test_all_sub.Visits))
test_all_sub[['Id', 'Visits']].to_csv('../submissions/%s_test.csv' % filename, index=False)

test_columns_save = [c+'_pred' for c in test_columns_code]
test_columns_save.append('Page')
test_save = test[test_columns_save]
test_save.shape

test3_save_columns = [c+'_pred' for c in tests[3].columns[1:-4]][:62]
test3_save_columns.append('Page')
test_save = test_save[test3_save_columns].reset_index(drop=True)
test_save.head()

test_save = test_save.iloc[145063:2*145063,:].reset_index(drop=True)

test3_save_columns = list(test_3_date)[:62]
test3_save_columns.append('Page')
test_save.columns = test3_save_columns
test_save.head()

test_save = pd.melt(test_save, id_vars=['Page'], var_name='Date', value_name='Visits')
test_save.Visits = (test_save.Visits / offset).round()

#print('%.5f' % smape(test_save.Visits_true, test_save.Visits))
test_save.to_csv('../submissions/%s_train.csv' % filename, index=False)

