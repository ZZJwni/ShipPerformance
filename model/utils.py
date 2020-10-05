from collections import namedtuple
from pathlib import Path
from typing import List, Dict, Tuple
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

ROOT = Path('/home/sprinter/MyProject/ShipPerformance/')
feature_path = ROOT / 'data/processed'
output_figpath = ROOT / 'output/fig'
output_datapath = ROOT / 'output/data'
log_filepath = ROOT / 'log'

def split_train_test(X: np.array, y: np.array, test_size: np.float, shuffle: bool = False, stratify:np.array=None)->Tuple[np.array, np.array, 
                                                                                                                          np.array, np.array]:
    """split train and test.
    """
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, stratify=stratify)
    return (X_train, y_train, X_test, y_test)

def standardize_train_test(scaler, X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array)->Tuple[np.array, np.array, 
                                                                                                                 np.array, np.array]:
    """standardize train and test dataset.
    """
    train_rescaled = scaler.fit_transform(
        np.concatenate((X_train, y_train[:, np.newaxis]), axis=1))
    test_rescaled = scaler.transform(np.concatenate(
        (X_test, y_test[:, np.newaxis]), axis=1))
    return train_rescaled[:, :-1], train_rescaled[:, -1], test_rescaled[:, :-1], test_rescaled[:, -1]

def standardize_traindata(X_train:pd.DataFrame, y_train:pd.DataFrame):
    """Standardize train dataset

    Return
    ------
    X_scalar : scalar of X_train
    y_scalar : scalar of y_train
    """
    pass

def plot_train_val_loss(train_loss, val_loss, fig_name:str, save_dir:str=str(output_figpath)):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(train_loss, label='train')
    ax.plot(val_loss, label='val')
    _ = ax.set_title(fig_name)
    fig.savefig('/'.join([save_dir, fig_name]))

def plot_pred_test(y_pred, y_test, fig_name:str, save_dir:str=str(output_figpath)):
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    sns.lineplot(x=y_test, y=y_test, ax=ax, color='r')
    _ = ax.set_title('y_test vs y_pred')
    fig.savefig('/'.join([save_dir, fig_name]))

def evaluate_model(y_test, y_test_pred,
                   y_train=None, y_train_pred=None, 
                   y_val=None, y_val_pred=None,
                   stdout=None):
    
    if stdout is not None:
        new_std = stdout
        old_std = sys.stdout
    else:
        new_std, old_std = sys.stdout, sys.stdout
    
    sys.stdout = new_std
    if y_train is not None and y_train_pred is not None:
        print('Prediction on train, RMSE : {}, R^2 : {}, MAE : {}'.format(np.sqrt(
            mean_squared_error(y_train, y_train_pred)), r2_score(y_train, y_train_pred), mean_absolute_error(y_train, y_train_pred)))
    if y_val is not None and y_val_pred is not None:
        print('Prediction on validation, RMSE : {}, R^2 : {}, MAE : {}'.format(np.sqrt(
            mean_squared_error(y_val, y_val_pred)), r2_score(y_val, y_val_pred), mean_absolute_error(y_val, y_val_pred)))
    print('Prediction on test, RMSE : {}, R^2 : {}, MAE : {}'.format(np.sqrt(
        mean_squared_error(y_test, y_test_pred)), r2_score(y_test, y_test_pred), mean_absolute_error(y_test, y_test_pred)))
    sys.stdout = old_std
    

def kfold_by_shipid(df: pd.DataFrame, n_splits: int = 6) -> List[Tuple[np.array, np.array]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    yield from skf.split(X=df, y=df['ship_id'])
