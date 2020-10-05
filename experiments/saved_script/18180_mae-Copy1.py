#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import sys
sys.path.append('..')


# In[3]:


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

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


# In[4]:


feature_filename = 'leg_top5_feat.csv'


# In[5]:


df = pd.read_csv(feature_path / feature_filename,
                 index_col=0, parse_dates=['utc'])
ship_ids = np.unique(df.index)

print(ship_ids)


# ###### 18180

# In[6]:


X_y = df.loc[18180, :].reset_index().drop('ship_id', axis=1)
X = X_y.drop(['foc_me', 'utc'], axis=1)
y = X_y['foc_me']
time = X_y['utc']
print('The number of features : ', X.shape[1])


# In[7]:


scaler = StandardScaler()
X_train_, y_train_, X_test_, y_test_ = split_train_test(X, y, .2)
X_train_val, y_train_val, X_test, y_test = standardize_train_test(scaler, X_train_, y_train_, X_test_, y_test_)
X_train, y_train, X_val, y_val = split_train_test(X_train_val, y_train_val, .2, shuffle=True)

print('The numbers of train data : ', X_train.shape[0])
print('The numbers of validate data : ', X_val.shape[0])
print('The numbers of test data : ', X_test.shape[0])


# In[8]:


batch_size = 32
epochs = 2000
optimizer = optimizers.SGD(learning_rate=.001)
loss = losses.mean_absolute_error
val_metrics = [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_absolute_percentage_error]
callback = [callbacks.EarlyStopping(patience=200)]


# In[9]:


model = NN(input_size=X.shape[1], output_size=1, layer_params=[
           layer_param(128, 'relu', .64), layer_param(64, 'relu', .64)],
           loss=loss, optimizer=optimizer, metrics=val_metrics)

model.build()


# In[10]:


model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, callbacks=callback, batch_size=batch_size, epochs=epochs)


# In[11]:


y_test_pred = model.predict(X_test, rescaler=scaler)
fig, ax = plt.subplots(figsize=(16, 8))
sns.scatterplot(x=y_test_, y=y_pred, ax=ax)
sns.lineplot(x=y_test_, y=y_test_, ax=ax, color='r')
_ = ax.set_title('y_test vs y_pred')


# In[ ]:


y_train_pred = model.predict(X_train_val, rescaler=scaler)
fig, ax = plt.subplots(figsize=(16, 8))
sns.scatterplot(x=y_train_, y=y_train_pred, ax=ax)
sns.lineplot(x=y_train_, y=y_train_, ax=ax, color='r')
_ = ax.set_title('y_train vs y_train_pred')


# In[ ]:


evaluate_model(y_test=y_test_, y_test_pred=y_test_pred,
               y_train=y_train_, y_train_pred=y_train_pred)


# In[27]:


model.delete()


# In[ ]:




