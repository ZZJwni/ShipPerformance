#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import sys
sys.path.append('..')


# In[23]:


from model.utils import evaluate_model
from model.utils import plot_train_val_loss, plot_pred_test
from model.utils import split_train_test, standardize_train_test
from model.utils import feature_path, output_figpath

from model import NN
from model.NN import layer_param

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow import optimizers, losses, metrics
from tensorflow.keras import callbacks


# In[5]:


feature_filename = 'leg_top5_feat.csv'


# In[6]:


df = pd.read_csv(feature_path / feature_filename,
                 index_col=0, parse_dates=['utc'])
ship_ids = np.unique(df.index)

print(ship_ids)


# ###### 24881

# In[7]:


X_y = df.loc[24881, :].reset_index().drop('ship_id', axis=1)
X = X_y.drop(['foc_me', 'utc'], axis=1)
y = X_y['foc_me']
time = X_y['utc']
print('The number of features : ', X.shape[1])


# In[8]:


scaler = StandardScaler()
X_train_, y_train_, X_test_, y_test_ = split_train_test(X, y, .3)
X_train_val, y_train_val, X_test, y_test = standardize_train_test(scaler, X_train_, y_train_, X_test_, y_test_)
X_train, y_train, X_val, y_val = split_train_test(X_train_val, y_train_val, .25, shuffle=True)

print('The numbers of train data : ', X_train.shape[0])
print('The numbers of validate data : ', X_val.shape[0])
print('The numbers of test data : ', X_test.shape[0])


# In[9]:


# batch_size = 32
# epochs = 1000
# optimizer = optimizers.SGD(learning_rate=.001)
# loss = losses.mean_absolute_error
val_metrics = [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_absolute_percentage_error]
callbacks = [callbacks.EarlyStopping(patience=50)]


# In[10]:


model = NN(X.shape[1], 1, [layer_param(64, 'relu', .64)])


# In[11]:


model.build()


# In[12]:


model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, callbacks=callbacks)


# In[13]:


y_pred = model.predict(X_test, rescaler=scaler)


# In[20]:


y_pred


# In[14]:


model.delete()


# In[15]:


model.build()


# In[17]:


model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, callbacks=callbacks)


# In[18]:


plot_train_val_loss(model._history['loss'], model._history['val_loss'], '1')


# In[21]:


plot_pred_test(y_pred, y_test_, '2')


# In[29]:


def run(feature, label, log_path='1', **model_params):
    print(log_path)
    print(model_params)


# In[30]:


run(1, 2, ll=5)


# In[ ]:




