#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm_notebook


# In[10]:


ROOT = Path('/home/sprinter/MyProject/ShipPerformance/')
data_path = ROOT / 'data'


# In[4]:


def _list_files(file_path: Path = data_path / 'data_error_corrected',
                file_name: str = 'leg.json'):
    """Generator to list all files which name is `file_name`
    inside `file_path`.
    """
    for file_or_dir in file_path.iterdir():
        if file_or_dir.is_file():
            if file_or_dir.name == file_name:
                yield file_or_dir
        else:
            yield from _list_files(file_or_dir, file_name)


# In[5]:


leg_list = list(_list_files(data_path / 'raw/data_error_corrected', 'leg.json')) +    list(_list_files(data_path / 'raw/data', 'leg.json'))
draftlist_list = list(_list_files(data_path / 'raw/data_error_corrected', 'draftlist.json')) +    list(_list_files(data_path / 'raw/data', 'draftlist.json'))


# In[6]:


leg_json_dtypes = {
    'client': np.dtype('O'),
    'course': np.dtype('float32'),
    'crnt_u': np.dtype('float32'),
    'crnt_v': np.dtype('float32'),
    'deadweight': np.dtype('int32'),
    'draft_aft': np.dtype('float32'),
    'draft_fore': np.dtype('float32'),
    'foc_me': np.dtype('float32'),
    'foc_total': np.dtype('float32'),
    'lat': np.dtype('float32'),
    'load': np.dtype('O'),
    'loading_condition': np.dtype('O'),
    'lon': np.dtype('float32'),
    'og_speed': np.dtype('float32'),
    'point_type': np.dtype('O'),
    'power': np.dtype('int32'),
    'pri_dir': np.dtype('float32'),
    'pri_ht': np.dtype('float32'),
    'pri_per': np.dtype('float32'),
    'rpm': np.dtype('O'),
    'sea_dir': np.dtype('float32'),
    'sea_ht': np.dtype('float32'),
    'sea_per': np.dtype('float32'),
    'sec_dir': np.dtype('float32'),
    'sec_ht': np.dtype('float32'),
    'sec_per': np.dtype('float32'),
    'sig_ht': np.dtype('float32'),
    'sst': np.dtype('int32'),
    'swl_dir': np.dtype('float32'),
    'swl_ht': np.dtype('float32'),
    'swl_per': np.dtype('float32'),
    'tw_speed': np.dtype('int32'),
    'utc': np.dtype('O'),
    'wind_u': np.dtype('float32'),
    'wind_v': np.dtype('float32')}


# In[8]:


# read all leg.json
def load_leg_json(data_paths: List[Path],
                  dtype: Dict[str, np.dtype]=None) -> pd.DataFrame:
    """load all leg.json into a pd.DataFrame object.
    """
    df_list = []
    for data_path in data_paths:
        df = (pd
              .read_json(data_path, orient='records', dtype=dtype)
              .assign(ship_id=np.int32(data_path.parent.parent.parent.name))
             )
        df_list.append(df)
    return pd.concat(df_list, axis=0)


# In[9]:


df = load_leg_json(tqdm_notebook(leg_list), leg_json_dtypes)


# In[64]:


df.to_hdf(data_path / 'processed/leg.h5', key='a')


# In[6]:


# read all draftlist.json
def load_draftlist_json(data_paths: List[Path],
                  dtype: Dict[str, np.dtype]=None) -> pd.DataFrame:
    """load all draftlist.json into a pd.DataFrame object.
    """
    df_list = []
    for data_path in data_paths:
        df = (pd
              .read_json(data_path, orient='records', dtype=dtype)
              .assign(ship_id=np.int32(data_path.parent.name))
             )
        df_list.append(df)
    return pd.concat(df_list, axis=0)


# In[7]:


df = load_draftlist_json(tqdm_notebook(draftlist_list))


# In[8]:


df.to_csv(data_path / 'processed/draftlist.csv')


# In[ ]:




