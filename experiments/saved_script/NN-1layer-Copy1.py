#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import namedtuple
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow import keras

import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
sns.set_style("whitegrid")


# In[8]:


fig, ax = plt.subplots(figsize=(16, 8))
_ = ax.set_title('eee')


# In[5]:





# In[3]:


type(StandardScaler())


# In[10]:


ROOT = Path('/home/sprinter/MyProject/ShipPerformance/')
feature_path = ROOT / 'data/processed'
feature_filename = 'leg_top5_feat.csv'


# In[3]:


df = pd.read_csv(feature_path / feature_filename,
                 index_col=0, parse_dates=['utc'])
ship_ids = np.unique(df.index)

print(ship_ids)


# In[ ]:


s


# In[4]:


def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: np.float, shuffle: bool = False):
    """split train and test.
    """
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle)
    return X_train, y_train, X_test, y_test


def standardize_train_test(X_train, y_train, X_test, y_test):
    """standardize train and test dataset.
    """
    scaler = StandardScaler()
    train_rescaled = scaler.fit_transform(
        np.concatenate((X_train, y_train[:, np.newaxis]), axis=1))
    test_rescaled = scaler.transform(np.concatenate(
        (X_test, y_test[:, np.newaxis]), axis=1))
    return train_rescaled[:, :-1], train_rescaled[:, -1], test_rescaled[:, :-1], test_rescaled[:, -1], scaler


layer_param = namedtuple(
    'layer_param', ('units', 'activation', 'dropout_rate'))


def build_model(input_size, output_size, *layer_params):
    """Build a dense nerual network model.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    for layer_param in layer_params:
        model.add(keras.layers.Dense(units=layer_param.units,
                                     activation=layer_param.activation))
        if layer_param.dropout_rate is not None:
            model.add(keras.layers.Dropout(rate=layer_param.dropout_rate))
    model.add(keras.layers.Dense(output_size))
    model.build(input_shape=(None, input_size))
    return model


def train_model(model,
                X_train, y_train,
                X_val, y_val,
                batch_size, epochs,
                optimizer, loss, metrics,
                callbacks=None
                ):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x=X_train, y=y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(X_val, y_val),
              shuffle=True, validation_freq=1, callbacks=callbacks)
    return model


def plot_train_val(history: Dict):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(history['loss'], label='train')
    ax.plot(history['val_loss'], label='val')
    fig.legend()


def evaluate_model(model,
                   X_train, y_train, X_test, y_test, X_val, y_val,
                   scaler):

    y_train_pred = scaler.inverse_transform(
        np.concatenate((X_train, model.predict(X_train)), axis=1))[:, -1]
    y_val_pred = scaler.inverse_transform(
        np.concatenate((X_val, model.predict(X_val)), axis=1))[:, -1]
    y_test_pred = scaler.inverse_transform(
        np.concatenate((X_test, model.predict(X_test)), axis=1))[:, -1]

    y_train = scaler.inverse_transform(
        np.concatenate((X_train, y_train[:, np.newaxis]), axis=1))[:, -1]
    y_val = scaler.inverse_transform(
        np.concatenate((X_val, y_val[:, np.newaxis]), axis=1))[:, -1]
    y_test = scaler.inverse_transform(
        np.concatenate((X_test, y_test[:, np.newaxis]), axis=1))[:, -1]

    print('Prediction on train, RMSE : {}, R^2 : {}, MAE : {}'.format(np.sqrt(
        mean_squared_error(y_train, y_train_pred)), r2_score(y_train, y_train_pred), mean_absolute_error(y_train, y_train_pred)))
    print('Prediction on validation, RMSE : {}, R^2 : {}, MAE : {}'.format(np.sqrt(
        mean_squared_error(y_val, y_val_pred)), r2_score(y_val, y_val_pred), mean_absolute_error(y_val, y_val_pred)))
    print('Prediction on test, RMSE : {}, R^2 : {}, MAE : {}'.format(np.sqrt(
        mean_squared_error(y_test, y_test_pred)), r2_score(y_test, y_test_pred), mean_absolute_error(y_test, y_test_pred)))

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.scatterplot(x=y_test, y=y_test_pred, ax=ax)
    sns.lineplot(x=y_test, y=y_test, ax=ax, color='r')
    ax.set_title('y_test vs y_pred')


# ###### 24881

# In[5]:


X_y = df.loc[24881, :].reset_index().drop('ship_id', axis=1)
X = X_y.drop(['foc_me', 'utc'], axis=1)
y = X_y['foc_me']
time = X_y['utc']
print('The number of features : ', X.shape[1])


# In[6]:


X_train_, y_train_, X_test_, y_test_ = split_train_test(X, y, .3)
X_train_val, y_train_val, X_test, y_test, scaler = standardize_train_test(X_train_, y_train_, X_test_, y_test_)
X_train, y_train, X_val, y_val = split_train_test(X_train_val, y_train_val, .25, shuffle=True)

print('The numbers of train data : ', X_train.shape[0])
print('The numbers of validate data : ', X_val.shape[0])
print('The numbers of test data : ', X_test.shape[0])


# In[48]:


batch_size = 32
epochs = 1000
optimizer = tf.optimizers.SGD(learning_rate=.001)
# loss = tf.losses.mean_squared_error
loss = tf.losses.mean_absolute_error
metrics = [tf.metrics.mean_absolute_error, tf.metrics.mean_squared_error, tf.metrics.mean_absolute_percentage_error]
callbacks = [keras.callbacks.EarlyStopping(patience=50)]


# In[49]:


model = build_model(X.shape[1], 1, layer_param(64, 'relu', .64))
model = train_model(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                    batch_size=batch_size, epochs=epochs, optimizer=optimizer,
                    loss=loss, metrics=metrics, callbacks=callbacks)


# In[50]:


plot_train_val(model.history.history)


# In[51]:


evaluate_model(model, 
               X_train, y_train, 
               X_test, y_test, 
               X_val, y_val,
               scaler)


# ###### 18180

# In[269]:


X_y = df.loc[18180, :].reset_index().drop('ship_id', axis=1)
X = X_y.drop(['foc_me', 'utc'], axis=1)
y = X_y['foc_me']
time = X_y['utc']
print('The number of features : ', X.shape[1])


# In[271]:


X_train_, y_train_, X_test_, y_test_ = split_train_test(X, y, .2)
X_train_val, y_train_val, X_test, y_test, scaler = standardize_train_test(X_train_, y_train_, X_test_, y_test_)
X_train, y_train, X_val, y_val = split_train_test(X_train_val, y_train_val, .2, shuffle=True)

print('The numbers of train data : ', X_train.shape[0])
print('The numbers of validate data : ', X_val.shape[0])
print('The numbers of test data : ', X_test.shape[0])


# In[272]:


batch_size = 32
epochs = 1000
optimizer = tf.optimizers.SGD(learning_rate=.001)
# loss = tf.losses.mean_squared_error
loss = tf.losses.mean_absolute_error
metrics = [tf.metrics.mean_absolute_error, tf.metrics.mean_squared_error, tf.metrics.mean_absolute_percentage_error]
callbacks = [keras.callbacks.EarlyStopping(patience=50)]


# In[273]:


model = build_model(X.shape[1], 1, layer_param(128, 'relu', .64))
model = train_model(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                    batch_size=batch_size, epochs=epochs, optimizer=optimizer,
                    loss=loss, metrics=metrics, callbacks=callbacks)


# In[274]:


plot_train_val(model.history.history)


# In[275]:


evaluate_model(model, 
               X_train, y_train, 
               X_test, y_test, 
               X_val, y_val,
               scaler)


# ###### 18180

# In[288]:


X_y = df.loc[18180, :].reset_index().drop('ship_id', axis=1)
X = X_y.drop(['foc_me', 'utc'], axis=1)
y = X_y['foc_me']
time = X_y['utc']
print('The number of features : ', X.shape[1])

X_train_, y_train_, X_test_, y_test_ = split_train_test(X, y, .2)
X_train_val, y_train_val, X_test, y_test, scaler = standardize_train_test(
    X_train_, y_train_, X_test_, y_test_)
X_train, y_train, X_val, y_val = split_train_test(
    X_train_val, y_train_val, .2, shuffle=True)

print('The numbers of train data : ', X_train.shape[0])
print('The numbers of validate data : ', X_val.shape[0])
print('The numbers of test data : ', X_test.shape[0])


# In[289]:


batch_size = 32
epochs = 1000
optimizer = tf.optimizers.SGD(learning_rate=.001)
loss = tf.losses.mean_squared_error
# loss = tf.losses.mean_absolute_error
metrics = [tf.metrics.mean_absolute_error, tf.metrics.mean_squared_error, tf.metrics.mean_absolute_percentage_error]
callbacks = [keras.callbacks.EarlyStopping(patience=50)]


# In[290]:


model = build_model(X.shape[1], 1, layer_param(, 'relu', .64))
model = train_model(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                    batch_size=batch_size, epochs=epochs, optimizer=optimizer,
                    loss=loss, metrics=metrics, callbacks=callbacks)


# In[291]:


plot_train_val(model.history.history)
evaluate_model(model, 
               X_train, y_train, 
               X_test, y_test, 
               X_val, y_val,
               scaler)


# In[ ]:




