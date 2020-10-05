#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas                  as     pd
import numpy                   as     np

# Visualization
import seaborn                 as     sns
import matplotlib.pyplot       as     plt
sns.set_style("whitegrid")

from pathlib import Path
ROOT = Path('/home/sprinter/MyProject/ShipPerformance/')
data_path = ROOT / 'data'


# In[2]:


df = pd.read_csv(data_path / 'processed/record_bulk.csv', index_col=0, parse_dates=['utc'],)
ship_ids = np.unique(df.index)


# In[3]:


df.groupby('label').apply(lambda s: len(s))


# In[3]:


df.columns


# In[9]:


df.query('ship_id == 874').drop(['record_index', 'record_length', 'label'], axis=1).to_csv('874.csv')


# In[ ]:





# In[4]:


# wave_columns = ['pri_ht', 'pri_dir', 'pri_per',
#                 'sec_ht', 'sec_dir', 'sec_per']
# draft_columns = ['draft_aft', 'draft_fore']
# foc_columns = ['foc_me', 'rpm']
# speed_columns = ['og_speed', 'course', 'tw_speed']
# crnt_columns = ['crnt_u', 'crnt_v']
# sea_columns = ['sea_dir', 'sea_ht', 'sea_per', 'sig_ht']
# swell_columns = ['swl_dir', 'swl_ht', 'swl_per']
# wind_columns = ['wind_u', 'wind_v']
# position_columns = ['lat', 'lon']


# #### Feature Engineering

# In[4]:


# modify angles
df_mod = df.assign(
    course=df['course'] / 360 * 2 * np.pi,
    pri_dir=df['pri_dir'] / 360 * 2 * np.pi,
    sec_dir=df['sec_dir'] / 360 * 2 * np.pi,
    sea_dir=df['sea_dir'] / 360 * 2 * np.pi,
    swl_dir=df['swl_dir'] / 360 * 2 * np.pi,
)


# In[5]:


def lg_from_uv(u: np.array, v: np.array, course: np.array) -> np.array:
    """Compute longitudinal components from u, v components
    """
    return u * np.cos(course) + v * np.sin(course)


def lg_from_magnitude(m: np.array, m_dir: np.array, course: np.array) -> np.array:
    """Compute longitudinal components from a magnitude and its dir.
    """
    return m * np.cos(m_dir - course)


def tr_from_magnitude(m: np.array, m_dir: np.array, course: np.array) -> np.array:
    """Compute transverse components from a magnitude and its dir.
    """
    return np.abs(m * np.sin(m_dir - course))


# In[6]:


df_mod = df_mod.assign(
    crnt_lg = lg_from_uv(df_mod['crnt_u'].values, df_mod['crnt_v'].values, df_mod['course'].values),
    wind_lg = lg_from_uv(df_mod['wind_u'].values, df_mod['wind_v'].values, df_mod['course'].values),
    pri_lg = lg_from_magnitude(df_mod['pri_ht'].values, df_mod['pri_dir'].values, df_mod['course'].values),
    sec_lg = lg_from_magnitude(df_mod['sec_ht'].values, df_mod['sec_dir'].values, df_mod['course'].values),
    sea_lg = lg_from_magnitude(df_mod['sea_ht'].values, df_mod['sea_dir'].values, df_mod['course'].values),
    swl_lg = lg_from_magnitude(df_mod['swl_ht'].values, df_mod['swl_dir'].values, df_mod['course'].values),
    pri_tr = tr_from_magnitude(df_mod['pri_ht'].values, df_mod['pri_dir'].values, df_mod['course'].values),
    sec_tr = tr_from_magnitude(df_mod['sec_ht'].values, df_mod['sec_dir'].values, df_mod['course'].values),
    sea_tr = tr_from_magnitude(df_mod['sea_ht'].values, df_mod['sea_dir'].values, df_mod['course'].values),
    swl_tr = tr_from_magnitude(df_mod['swl_ht'].values, df_mod['swl_dir'].values, df_mod['course'].values),
    
)
df_mod = df_mod.assign(trim=df_mod['draft_fore']-df_mod['draft_aft'],
                       draft_mean=(df_mod['draft_fore']+df_mod['draft_aft'])/2)


# In[7]:


df_mod = df_mod.assign(rpm_norm = df_mod['rpm'] / df_mod['rpm_at_mcr'], 
                       speed_norm = df_mod['og_speed'] / df_mod['speed_at_mcr'])


# In[8]:


df_mod.columns


# In[9]:


df_mod.to_csv(data_path / 'processed/feat_bulk.csv')


# In[10]:


cols = ['crnt_lg', 'wind_lg', 
        'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
        'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
        'draft_aft', 'draft_fore', 'trim', 'draft_mean',
        'foc_me', 'rpm', 'og_speed', 'ship_id', 'client',
        'record_length', 'record_index', 'dwt', 'draft', 'breadth',
        'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
        'date_built_year', 'power_at_mcr', 'label', 'utc',
        'speed_norm', 'rpm_norm',
        ]


# In[11]:


df_mod_1 = df_mod[cols]


# In[12]:


all_feats = ['crnt_lg', 'wind_lg', 
             'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
             'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
             'draft_aft', 'draft_fore', 'trim', 'draft_mean',
             'foc_me', 'rpm', 'og_speed', 
             'dwt', 'draft', 'breadth',
             'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
             'date_built_year', 'power_at_mcr',
             'speed_norm', 'rpm_norm']


# ###### Train Test Split

# In[53]:


df_mod_1['label'].value_counts()


# In[26]:


from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple


def kfold_by_shipid(df: pd.DataFrame, n_splits: int = 6) -> List[Tuple[np.array, np.array]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    yield from skf.split(X=df, y=df['ship_id'])


# In[39]:


# 1. Make cv(train, test) of all labels
# 2. Make cv(train, test) of label 0,1,2,3,4, respectively
# 3. Make cv(train, test) of label 0,3,4


# In[40]:


all_feats = ['crnt_lg', 'wind_lg', 
             'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
             'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
             'draft_aft', 'draft_fore', 'trim', 'draft_mean',
             'foc_me', 'rpm', 'og_speed', 
             'dwt', 'draft', 'breadth',
             'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
             'date_built_year', 'power_at_mcr',
             'speed_norm', 'rpm_norm']

feats_1 = ['crnt_lg', 'wind_lg', 
             'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
             'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
             'draft_aft', 'draft_fore', 'trim', 'draft_mean',
             'foc_me', 'rpm', 'og_speed', ]

feats_2 = ['crnt_lg', 'wind_lg', 
           'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
           'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
           'draft_aft', 'draft_fore', 'trim', 'draft_mean',
           'foc_me', 'rpm', 'og_speed', 
           'dwt', 'draft', 'breadth',
           'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
           'power_at_mcr',]

feats_3 = ['crnt_lg', 'wind_lg', 
           'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
           'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
           'draft_aft', 'draft_fore', 'trim', 'draft_mean',
           'foc_me', 
           'dwt', 'draft', 'breadth',
           'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
           'power_at_mcr', 'speed_norm', 'rpm_norm']

feats_4 = ['crnt_lg', 'wind_lg', 
           'pri_lg', 'sec_lg', 'pri_tr', 'sec_tr',
           'sea_lg', 'sea_tr', 'swl_lg', 'swl_tr',
           'draft_aft', 'draft_fore', 'trim', 'draft_mean',
           'foc_me', 'rpm', 'og_speed', 
           'dwt', 'draft', 'breadth',
           'length', 'depth', 'gt', 'rpm_at_mcr', 'speed_at_mcr',
           'power_at_mcr', 'speed_norm', 'rpm_norm']


# In[ ]:




