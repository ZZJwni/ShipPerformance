#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

from pathlib import Path
ROOT = Path('/home/sprinter/MyProject/ShipPerformance/')
data_path = ROOT / 'data'


# In[2]:


ship_db = pd.read_excel(io=data_path / 'raw/shipdb.xlsx', sheet_name=3, 
                          usecols=['wni_ship_num', 'ship_type', 'date_built_year', 
                                   'length', 'breadth','depth', 'draft', 'dwt',
                                   'disp', 'gt', 'teu', 'power_at_mcr',
                                   'speed_at_mcr', 'rpm_at_mcr'],
                          dtype={'wni_ship_num' : np.int64})

df = pd.read_csv(filepath_or_buffer=data_path / 'processed/leg.csv', 
                 usecols=['ship_id'], dtype={'ship_id' : np.int64})
ship_ids = df['ship_id'].unique()

ship_db = ship_db.rename(columns={'wni_ship_num' : 'ship_id'}, copy=True, inplace=False)
ship_db = ship_db.loc[ship_db['ship_id'].isin(ship_ids), :]
ship_db = ship_db.loc[:, ['ship_id', 'ship_type', 'length', 'breadth', 'depth',
                          'draft', 'dwt', 'gt', 'power_at_mcr', 'speed_at_mcr',
                          'rpm_at_mcr']] # `disp` and `teu` are not used.

del df


# In[3]:


print('The number of ship : ', len(ship_db))


# In[4]:


ship_db.columns


# In[6]:


url = 'http://vpdb-vmg.wni.co.jp/shipdb/cgi/get.cgi?'
result = requests.get(url=url, params={'wni_ship_num': str(69)})


# In[9]:


result.url


# In[5]:


from tqdm.notebook import tqdm
from typing import List
from xml.etree import ElementTree
import requests

ship_ids = ship_db['ship_id'].tolist()


def get_shipdb(url: str = 'http://vpdb-vmg.wni.co.jp/shipdb/cgi/get.cgi?', ship_ids: List[str] = ['69']) -> pd.DataFrame:

    result_list = []
    for ship_id in tqdm(ship_ids):
        result = requests.get(url=url, params={'wni_ship_num': str(ship_id)})
        if int(result.status_code) != 200:
            print(f'Error in {ship_id}.')
            continue
        root = ElementTree.fromstring(result.text)
        result_list.append({str(child.tag): str(child.text) for child in root})

    df = pd.DataFrame.from_records(data=result_list,
                                   columns=['wni_ship_num', 'ship_type', 'date_built',
                                            'length', 'breadth', 'depth', 'draft',
                                            'dwt', 'teu', 'gt', 'bhp', 'rpm_at_mcr', 'speed_at_mcr'])
    df = df.astype({'wni_ship_num': np.int64, 'length': np.float64, 'breadth': np.float64,
                    'depth': np.float64, 'draft': np.float64, 'dwt': np.float64,
                    'teu': np.float64, 'gt': np.float64, 'bhp': np.float64,
                    'rpm_at_mcr': np.float64, 'speed_at_mcr': np.float64})
    df = df.rename(columns={'wni_ship_num': 'ship_id', 'bhp': 'power_at_mcr'})
    return df


# In[61]:


df = get_shipdb(ship_ids=ship_ids)


# In[66]:


df.to_csv(data_path / 'processed/shipdb.csv')


# In[67]:


ship_db = pd.read_csv(data_path / 'processed/shipdb.csv', index_col=0)


# In[75]:


ship_db['ship_type'].value_counts()


# In[68]:


ship_db.dtypes


# In[69]:


# convert dtype frpm str to float
ship_db.loc[:, 'length'] = ship_db['length'].astype(np.float64)


# In[70]:


# check missing values
ship_db.loc[:, ['length', 'breadth', 'depth','draft', 'dwt', 
                'gt', 'power_at_mcr', 'speed_at_mcr','rpm_at_mcr']].apply(lambda col: np.sum(np.isnan(col))) / len(ship_db)


# In[71]:


# drop na
print('Ratios of dropped rows : ', 1 - len(ship_db.dropna()) / len(ship_db))

ship_db_1 = ship_db.dropna()


# In[72]:


cols = ['length', 'breadth', 'depth','draft', 'dwt', 
        'gt', 'power_at_mcr', 'speed_at_mcr','rpm_at_mcr']
fig, axes = plt.subplots(len(cols), 1, figsize=(8, len(cols) * 8))
for i, col in enumerate(cols):
    print(f'{col}, min : {ship_db_1[col].min()}, max : {ship_db_1[col].max()}.')
    sns.distplot(a=ship_db_1[col], ax=axes[i], label=col)


# In[73]:


# remove 0 values
ship_db_2 = ship_db_1.query('depth > 0').query('power_at_mcr > 0')
print('Ratios of dropped rows : ', 1 - len(ship_db_2) / len(ship_db))


# In[74]:


ship_db_2['ship_type'].value_counts()


# ##### ship classification

# In[76]:


from sklearn.cluster import KMeans, DBSCAN
from itertools import combinations


# ###### BULK

# In[77]:


# cols = ['length', 'breadth', 'depth','draft', 'dwt', 
#         'gt', 'power_at_mcr']
# X = ship_db_2.loc[ship_db_2['ship_type'] == 'BULK CARRIER' ,cols]
# dbscan = DBSCAN(eps=5000, n_jobs=-1,)
# dbscan.fit(X)

# labels_count = pd.Series(dbscan.labels_).value_counts()
# print('The number of labels :', labels_count)

# X.loc[:, 'label'] = dbscan.labels_
# iters = list(combinations(cols, 2))
# fig, axes = plt.subplots(len(iters),1,  figsize=(8, len(iters) * 8))
# cmap = sns.color_palette(palette='muted', n_colors=len(labels_count))
# for i, (x, y) in enumerate(iters):
#     sns.scatterplot(x=x, y=y, hue='label', data=X, ax=axes[i], palette=cmap)


# In[79]:


cols = ['length', 'breadth', 'depth','draft', 'dwt', 
        'gt', 'power_at_mcr']
X = ship_db_2.loc[ship_db_2['ship_type'] == 'BULK CARRIER' ,cols]
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, n_init=30, max_iter=1000, n_jobs=-1)
kmeans.fit(X)

labels_count = pd.Series(kmeans.labels_).value_counts()
print('The number of labels :', labels_count)

X.loc[:, 'label'] = kmeans.labels_
iters = list(combinations(cols, 2))
fig, axes = plt.subplots(len(iters),1,  figsize=(8, len(iters) * 8))
cmap = sns.color_palette(palette='muted', n_colors=len(labels_count))
for i, (x, y) in enumerate(iters):
    sns.scatterplot(x=x, y=y, hue='label', data=X, ax=axes[i], palette=cmap)


# In[80]:


bulk_carrier = ship_db_2.loc[ship_db_2['ship_type'] == 'BULK CARRIER', :]
bulk_carrier.loc[:, 'label'] = kmeans.labels_


# In[81]:


bulk_carrier.to_csv(data_path / 'processed/db_bulk_acrrier.csv')


# ###### CONTAINER

# In[82]:


# cols = ['length', 'breadth', 'depth','draft', 'dwt', 
#         'gt', 'power_at_mcr']
# X = ship_db_2.loc[ship_db_2['ship_type'] == 'CONTAINER SHIP' ,cols]
# dbscan = DBSCAN(eps=9000, n_jobs=-1,)
# dbscan.fit(X)

# labels_count = pd.Series(dbscan.labels_).value_counts()
# print('The number of labels :', labels_count)

# X.loc[:, 'label'] = dbscan.labels_
# iters = list(combinations(cols, 2))
# fig, axes = plt.subplots(len(iters),1,  figsize=(8, len(iters) * 8))
# cmap = sns.color_palette(palette='muted', n_colors=len(labels_count))
# for i, (x, y) in enumerate(iters):
#     sns.scatterplot(x=x, y=y, hue='label', data=X, ax=axes[i], palette=cmap)


# In[83]:


cols = ['length', 'breadth', 'depth','draft', 'dwt', 
        'gt', 'power_at_mcr']
X = ship_db_2.loc[ship_db_2['ship_type'] == 'CONTAINER SHIP' ,cols]
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, n_init=30, max_iter=1000, n_jobs=-1)
kmeans.fit(X)

labels_count = pd.Series(kmeans.labels_).value_counts()
print('The number of labels :', labels_count)

X.loc[:, 'label'] = kmeans.labels_
iters = list(combinations(cols, 2))
fig, axes = plt.subplots(len(iters),1,  figsize=(8, len(iters) * 8))
cmap = sns.color_palette(palette='muted', n_colors=len(labels_count))
for i, (x, y) in enumerate(iters):
    sns.scatterplot(x=x, y=y, hue='label', data=X, ax=axes[i], palette=cmap)


# In[84]:


container_ship = ship_db_2.loc[ship_db_2['ship_type'] == 'CONTAINER SHIP', :]
container_ship.loc[:, 'label'] = kmeans.labels_


# In[85]:


container_ship.to_csv(data_path / 'processed/db_container_ship.csv')


# In[ ]:




