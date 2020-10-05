#!/usr/bin/env python
# coding: utf-8

# In[18]:


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


# In[19]:


df = pd.read_csv(data_path / 'leg_top5_mod.csv', index_col=0, parse_dates=['utc'],)


# In[22]:


df.groupby('ship_id')['utc'].describe()


# In[8]:


wave_columns = ['pri_ht', 'pri_dir', 'pri_per',
                'sec_ht', 'sec_dir', 'sec_per']
draft_columns = ['draft_aft', 'draft_fore']
foc_columns = ['foc_me', 'rpm']
speed_columns = ['og_speed', 'course', 'tw_speed']
crnt_columns = ['crnt_u', 'crnt_v']
sea_columns = ['sea_dir', 'sea_ht', 'sea_per', 'sig_ht']
swell_columns = ['swl_dir', 'swl_ht', 'swl_per']
wind_columns = ['wind_u', 'wind_v']
position_columns = ['lat', 'lon']

ship_ids = np.unique(df.index)


# #### Feature Engineering

# In[9]:


df_mod = df.assign(crnt_speed=np.sqrt(df['crnt_u']**2 + df['crnt_v'] ** 2),
                   crnt_dir=np.arctan2(df['crnt_v'], df['crnt_u']),
                   wind_speed=np.sqrt(df['wind_u']**2 + df['wind_v'] ** 2),
                   wind_dir=np.arctan2(df['wind_v'], df['wind_u']))


def modify_angle(arr):
    return np.where(arr < 0, 2 * np.pi + arr, arr)


df_mod = df_mod.assign(crnt_dir=modify_angle(df_mod['crnt_dir']),
                       wind_dir=modify_angle(df_mod['wind_dir']),
                       course=df_mod['course'] / 360 * 2 * np.pi,
                       pri_dir=df_mod['pri_dir'] / 360 * 2 * np.pi,
                       sec_dir=df_mod['sec_dir'] / 360 * 2 * np.pi,
                       sea_dir=df_mod['sea_dir'] / 360 * 2 * np.pi,
                       swl_dir=df_mod['swl_dir'] / 360 * 2 * np.pi,
                       )

df_mod = df_mod.assign(crnt_effect=df_mod['crnt_speed'] * np.cos(df_mod['crnt_dir']-df_mod['course']),
                       wind_effect=df_mod['wind_speed'] *
                       np.cos(df_mod['wind_dir']-df_mod['course']),
                       )
df_mod = df_mod.assign(trim=df_mod['draft_fore']-df_mod['draft_aft'],
                       draft_mean=(df_mod['draft_fore']+df_mod['draft_aft'])/2)
df_mod = df_mod.assign(crnt_rdir=np.abs(df_mod['crnt_dir']-df_mod['course']),
                       wind_rdir=np.abs(df_mod['wind_dir']-df_mod['course']),
                      )

df_mod = df_mod.assign(crnt_effect_squared=df_mod['crnt_effect']**2,
                       wind_effect_squared=df_mod['wind_effect']**2,
                       rpm_squared=df_mod['rpm']**2,
                       og_speed_squared=df_mod['og_speed']**2)


# #### EDA

# In[10]:


cols = ['pri_ht', 'pri_dir', 'pri_per',
        'sec_ht', 'sec_dir', 'sec_per'] +\
       ['draft_aft', 'draft_fore'] +\
       ['foc_me', 'rpm']+\
       ['og_speed', 'course']+\
       ['crnt_u', 'crnt_v']+\
       ['sea_dir', 'sea_ht', 'sea_per', 'sig_ht']+\
       ['swl_dir', 'swl_ht', 'swl_per']+\
       ['wind_u', 'wind_v']+\
       ['wind_dir', 'wind_speed', 'crnt_dir', 'crnt_speed']+\
       ['crnt_effect', 'wind_effect']+\
       ['trim', 'draft_mean']+\
       ['wind_rdir', 'crnt_rdir']+\
       ['crnt_effect_squared', 'wind_effect_squared', 'rpm_squared', 'og_speed_squared']


# In[11]:


# Investigate multi-collinearity of variables
for ship_id in ship_ids:
    print('----------', ship_id, '----------')
    fig, ax = plt.subplots(figsize=(10,10))
    corr = df_mod.loc[ship_id, cols].corr()
    sns.heatmap(corr, cmap="Blues", ax=ax)
    ax.set_title(ship_id)
    print(corr['foc_me'].sort_values(ascending=False))


# In[17]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.scatterplot(x='og_speed', y='foc_me',
                hue='wind_speed',data=df_mod )


# In[16]:


for ship_id in ship_ids:
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.scatterplot(x='og_speed', y='foc_me',
                    hue='wind_speed',data=df_mod.loc[ship_id], )


# In[181]:


cols = foc_columns + ['og_speed', 'og_speed_squared']
sns.pairplot(data=df_mod.reset_index(), hue='ship_id', 
             vars=cols, kind='reg', diag_kind='hist', diag_kws={'stacked':True}, height=5)

# Investigate multi-collinearity of variables
for ship_id in ship_ids:
    print('----------', ship_id, '----------')
    fig, ax = plt.subplots(figsize=(5, 5))
    corr = df_mod.loc[ship_id, cols].corr()
    sns.heatmap(corr, cmap="Blues", ax=ax, annot=True, fmt=".3f")
    ax.set_title(ship_id)


# In[25]:


cols = draft_columns + ['foc_me'] + ['trim', 'draft_mean']
sns.pairplot(data=df_mod.reset_index(), hue='ship_id', 
             vars=cols, kind='scatter', diag_kind='kde', height=5)

# Investigate multi-collinearity of variables
for ship_id in ship_ids:
    print('----------', ship_id, '----------')
    fig, ax = plt.subplots(figsize=(5, 5))
    corr = df_mod.loc[ship_id, cols].corr()
    sns.heatmap(corr, cmap="Blues", ax=ax, annot=True, fmt=".3f")
    ax.set_title(ship_id)


# In[26]:


cols = ['course']+       ['wind_dir', 'wind_speed', 'crnt_dir', 'crnt_speed']+       ['crnt_effect', 'wind_effect']+       ['foc_me']
sns.pairplot(data=df_mod.reset_index(), hue='ship_id', 
             vars=cols, kind='scatter', diag_kind='kde', height=5)

# Investigate multi-collinearity of variables
for ship_id in ship_ids:
    print('----------', ship_id, '----------')
    fig, ax = plt.subplots(figsize=(8, 8))
    corr = df_mod.loc[ship_id, cols].corr()
    sns.heatmap(corr, cmap="Blues", ax=ax, annot=True, fmt=".3f")
    ax.set_title(ship_id)


# In[27]:


cols = sea_columns + swell_columns + ['foc_me']
sns.pairplot(data=df_mod.reset_index(), hue='ship_id', 
             vars=cols, kind='scatter', diag_kind='kde', height=5)

# Investigate multi-collinearity of variables
for ship_id in ship_ids:
    print('----------', ship_id, '----------')
    fig, ax = plt.subplots(figsize=(8, 8))
    corr = df_mod.loc[ship_id, cols].corr()
    sns.heatmap(corr, cmap="Blues", ax=ax, annot=True, fmt=".3f")
    ax.set_title(ship_id)


# In[28]:


cols = wave_columns + ['foc_me']
sns.pairplot(data=df_mod.reset_index(), hue='ship_id', 
             vars=cols, kind='scatter', diag_kind='kde', height=5)

# Investigate multi-collinearity of variables
for ship_id in ship_ids:
    print('----------', ship_id, '----------')
    fig, ax = plt.subplots(figsize=(8, 8))
    corr = df_mod.loc[ship_id, cols].corr()
    sns.heatmap(corr, cmap="Blues", ax=ax, annot=True, fmt=".3f")
    ax.set_title(ship_id)


# #### MLR

# In[207]:


cols = ['pri_ht','pri_dir','pri_per','sec_ht','sec_dir','sec_per','draft_aft',
        'draft_fore','rpm','og_speed','course','crnt_u','crnt_v','sea_dir',
        'sea_ht','sea_per','sig_ht','swl_dir','swl_ht','swl_per','wind_u',
        'wind_v','wind_dir','wind_speed','crnt_dir','crnt_speed','crnt_effect',
        'wind_effect','trim','draft_mean','wind_rdir','crnt_rdir',
        'crnt_effect_squared', 'wind_effect_squared', 'rpm_squared', 'og_speed_squared']
label = 'foc_me'


# In[183]:


df_mod_1 = df_mod.reset_index().set_index(['ship_id', 'utc']).sort_values(by=['ship_id', 'utc'])


# In[184]:


from sklearn.model_selection import train_test_split

X_train = dict.fromkeys(ship_ids)
X_test = dict.fromkeys(ship_ids)
y_train = dict.fromkeys(ship_ids)
y_test = dict.fromkeys(ship_ids)

for ship_id in ship_ids:
    X_train[ship_id], X_test[ship_id], y_train[ship_id], y_test[ship_id] =    train_test_split(df_mod_1.loc[ship_id, cols], df_mod_1.loc[ship_id, label], test_size=.25, shuffle=False)


# In[190]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from statsmodels.api import graphics
from scipy import stats

def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true-y_pred) / y_true)


# ##### 12796

# In[191]:


ship_id = 12796
s_col = StandardScaler()
X_train_ = s_col.fit_transform(X=X_train[ship_id].values)
l_col = StandardScaler()
y_train_ = l_col.fit_transform(X=y_train[ship_id].values.reshape(-1, 1))
# X_train_ = X_train[ship_id].values
# y_train_ = y_train[ship_id].values

model = RidgeCV(alphas=[1, 3, 5, 7, 9], cv=KFold(n_splits=10, shuffle=False))
model.fit(X_train_, y_train_)
print('alpha: ', model.alpha_)

X_test_ = s_col.transform(X=X_test[ship_id].values)
# X_test_ = X_test[ship_id].values
y_pred_ = model.predict(X_test_)
y_pred = l_col.inverse_transform(y_pred_.reshape(-1, 1))
y_pred = np.ravel(y_pred)
# y_pred = y_pred_

print('R-squared score:         %.3f' %
      r2_score(y_test[ship_id].values, y_pred))
print("Root mean squared error: %.3f" %
      np.sqrt(mean_squared_error(y_test[ship_id].values,  y_pred)))
print("mean absolute error: %.3f" %
      mean_absolute_error(y_test[ship_id].values, y_pred))
print("mean absolute percentage error: %.3f" %
      mean_absolute_percentage_error(y_test[ship_id].values, y_pred))

# Residuals of test set
res = y_test[ship_id].values - y_pred

# Examine normality of residuals
fig, ax = plt.subplots(figsize=(16, 2))
graphics.qqplot(res, dist=stats.norm, line='45', fit=True, ax=ax)

fig, ax = plt.subplots(figsize=(16, 4))
sns.scatterplot(x=y_test[ship_id].values, y=y_pred, ax=ax)
sns.lineplot(x=y_test[ship_id].values,
             y=y_test[ship_id].values, ax=ax, color='r')
ax.set_title('y_test vs y_pred')

# Plot distributions of residuals
fig, ax = plt.subplots(figsize=(16, 1.5))
sns.distplot(res, ax=ax)
ax.set_title('distribution of residuals')

fig, ax = plt.subplots(figsize=(8, 8))
(pd
 .DataFrame({'col': cols, 'coef': np.ravel(model.coef_),
             'coef_abs': np.abs(np.ravel(model.coef_))})
 .sort_values(by=['coef_abs'], ascending=False)
 .set_index('col')['coef_abs']
 .plot(kind='barh', ax=ax))


# ##### 17165

# In[192]:


ship_id = 17165
s_col = StandardScaler()
X_train_ = s_col.fit_transform(X=X_train[ship_id].values)
l_col = StandardScaler()
y_train_ = l_col.fit_transform(X=y_train[ship_id].values.reshape(-1, 1))
# X_train_ = X_train[ship_id].values
# y_train_ = y_train[ship_id].values

model = RidgeCV(alphas=[1, 3, 5, 7, 9], cv=KFold(n_splits=10, shuffle=False))
model.fit(X_train_, y_train_)
print('alpha: ', model.alpha_)

X_test_ = s_col.transform(X=X_test[ship_id].values)
# X_test_ = X_test[ship_id].values
y_pred_ = model.predict(X_test_)
y_pred = l_col.inverse_transform(y_pred_.reshape(-1, 1))
y_pred = np.ravel(y_pred)
# y_pred = y_pred_

print('R-squared score:         %.3f' %
      r2_score(y_test[ship_id].values, y_pred))
print("Root mean squared error: %.3f" %
      np.sqrt(mean_squared_error(y_test[ship_id].values,  y_pred)))
print("mean absolute error: %.3f" %
      mean_absolute_error(y_test[ship_id].values, y_pred))
print("mean absolute percentage error: %.3f" %
      mean_absolute_percentage_error(y_test[ship_id].values, y_pred))

# Residuals of test set
res = y_test[ship_id].values - y_pred

# Examine normality of residuals
fig, ax = plt.subplots(figsize=(16, 2))
graphics.qqplot(res, dist=stats.norm, line='45', fit=True, ax=ax)

fig, ax = plt.subplots(figsize=(16, 4))
sns.scatterplot(x=y_test[ship_id].values, y=y_pred, ax=ax)
sns.lineplot(x=y_test[ship_id].values,
             y=y_test[ship_id].values, ax=ax, color='r')
ax.set_title('y_test vs y_pred')

# Plot distributions of residuals
fig, ax = plt.subplots(figsize=(16, 1.5))
sns.distplot(res, ax=ax)
ax.set_title('distribution of residuals')

fig, ax = plt.subplots(figsize=(8, 8))
(pd
 .DataFrame({'col': cols, 'coef': np.ravel(model.coef_),
             'coef_abs': np.abs(np.ravel(model.coef_))})
 .sort_values(by=['coef_abs'], ascending=False)
 .set_index('col')['coef_abs']
 .plot(kind='barh', ax=ax))


# ##### 18180

# In[193]:


ship_id = 18180
s_col = StandardScaler()
X_train_ = s_col.fit_transform(X=X_train[ship_id].values)
l_col = StandardScaler()
y_train_ = l_col.fit_transform(X=y_train[ship_id].values.reshape(-1, 1))
# X_train_ = X_train[ship_id].values
# y_train_ = y_train[ship_id].values

model = RidgeCV(alphas=[1, 3, 5, 7, 9], cv=KFold(n_splits=10, shuffle=False))
model.fit(X_train_, y_train_)
print('alpha: ', model.alpha_)

X_test_ = s_col.transform(X=X_test[ship_id].values)
# X_test_ = X_test[ship_id].values
y_pred_ = model.predict(X_test_)
y_pred = l_col.inverse_transform(y_pred_.reshape(-1, 1))
y_pred = np.ravel(y_pred)
# y_pred = y_pred_

print('R-squared score:         %.3f' %
      r2_score(y_test[ship_id].values, y_pred))
print("Root mean squared error: %.3f" %
      np.sqrt(mean_squared_error(y_test[ship_id].values,  y_pred)))
print("mean absolute error: %.3f" %
      mean_absolute_error(y_test[ship_id].values, y_pred))
print("mean absolute percentage error: %.3f" %
      mean_absolute_percentage_error(y_test[ship_id].values, y_pred))

# Residuals of test set
res = y_test[ship_id].values - y_pred

# Examine normality of residuals
fig, ax = plt.subplots(figsize=(16, 2))
graphics.qqplot(res, dist=stats.norm, line='45', fit=True, ax=ax)

fig, ax = plt.subplots(figsize=(16, 4))
sns.scatterplot(x=y_test[ship_id].values, y=y_pred, ax=ax)
sns.lineplot(x=y_test[ship_id].values,
             y=y_test[ship_id].values, ax=ax, color='r')
ax.set_title('y_test vs y_pred')

# Plot distributions of residuals
fig, ax = plt.subplots(figsize=(16, 1.5))
sns.distplot(res, ax=ax)
ax.set_title('distribution of residuals')

fig, ax = plt.subplots(figsize=(8, 8))
(pd
 .DataFrame({'col': cols, 'coef': np.ravel(model.coef_),
             'coef_abs': np.abs(np.ravel(model.coef_))})
 .sort_values(by=['coef_abs'], ascending=False)
 .set_index('col')['coef_abs']
 .plot(kind='barh', ax=ax))


# ##### 24881

# In[194]:


ship_id = 24881
s_col = StandardScaler()
X_train_ = s_col.fit_transform(X=X_train[ship_id].values)
l_col = StandardScaler()
y_train_ = l_col.fit_transform(X=y_train[ship_id].values.reshape(-1, 1))
# X_train_ = X_train[ship_id].values
# y_train_ = y_train[ship_id].values

model = RidgeCV(alphas=[1, 3, 5, 7, 9], cv=KFold(n_splits=10, shuffle=False))
model.fit(X_train_, y_train_)
print('alpha: ', model.alpha_)

X_test_ = s_col.transform(X=X_test[ship_id].values)
# X_test_ = X_test[ship_id].values
y_pred_ = model.predict(X_test_)
y_pred = l_col.inverse_transform(y_pred_.reshape(-1, 1))
y_pred = np.ravel(y_pred)
# y_pred = y_pred_

print('R-squared score:         %.3f' %
      r2_score(y_test[ship_id].values, y_pred))
print("Root mean squared error: %.3f" %
      np.sqrt(mean_squared_error(y_test[ship_id].values,  y_pred)))
print("mean absolute error: %.3f" %
      mean_absolute_error(y_test[ship_id].values, y_pred))
print("mean absolute percentage error: %.3f" %
      mean_absolute_percentage_error(y_test[ship_id].values, y_pred))

# Residuals of test set
res = y_test[ship_id].values - y_pred

# Examine normality of residuals
fig, ax = plt.subplots(figsize=(16, 2))
graphics.qqplot(res, dist=stats.norm, line='45', fit=True, ax=ax)

fig, ax = plt.subplots(figsize=(16, 4))
sns.scatterplot(x=y_test[ship_id].values, y=y_pred, ax=ax)
sns.lineplot(x=y_test[ship_id].values,
             y=y_test[ship_id].values, ax=ax, color='r')
ax.set_title('y_test vs y_pred')

# Plot distributions of residuals
fig, ax = plt.subplots(figsize=(16, 1.5))
sns.distplot(res, ax=ax)
ax.set_title('distribution of residuals')

fig, ax = plt.subplots(figsize=(8, 8))
(pd
 .DataFrame({'col': cols, 'coef': np.ravel(model.coef_),
             'coef_abs': np.abs(np.ravel(model.coef_))})
 .sort_values(by=['coef_abs'], ascending=False)
 .set_index('col')['coef_abs']
 .plot(kind='barh', ax=ax))


# #### Multi-Linear Regression Model Vessel Performance

# ##### 24881

# In[227]:


ship_id = 24881
df_ = df_mod_1.loc[ship_id, cols].describe().loc[['min', 'max'], :].T
var_spaces = dict.fromkeys(df_.index)
Var_spaces = dict.fromkeys(df_.index)
for k in var_spaces:
    var_spaces[k] = np.linspace(df_.loc[k, 'min'], df_.loc[k, 'max'], 8)
np.meshgrid(*[v for _, v in var_spaces.items()])


# In[201]:



cols = ['rpm', 'og_speed', 'sea_ht',
        'swl_ht', 'wind_effect', 'crnt_effect',
        'trim',]
df_mod_1.loc[ship_id, cols].describe()

x_rpm = np.linspace(57, 78, 10)
x_trim = np.linspace(-4, 0, 10)
x_og_speed = np.linspace(9, 15, 10)
x_sea_ht = np.linspace(0, 8, 10)
x_swl_ht = np.linspace(0, 7.5, 10)
x_wind = np.linspace(-18, 15, 20)
x_crnt = np.linspace(-2.5, 2.5, 10)
X_rpm, X_trim, X_og_speed, X_sea_ht, X_swl_ht, X_wind, X_crnt = np.meshgrid(x_rpm, x_trim, x_og_speed, x_sea_ht, x_swl_ht, x_wind, x_crnt)
df_s = pd.DataFrame({
    'rpm': X_rpm.flatten(), 
    'trim': X_trim.flatten(), 
    'og_speed':X_og_speed.flatten(), 
    'sea_ht':X_sea_ht.flatten(), 
    'swl_ht':X_swl_ht.flatten(), 
    'wind_effect': X_wind.flatten(), 
    'crnt_effect': X_crnt.flatten()
})


# In[202]:


# Samples
plt.figure(figsize=(16,8))
sns.scatterplot(x    = 'og_speed',
                y    = 'Main Engine Fuel Consumption (MT/day)',
                hue  = 'Draft Mean (meters)',
                data = df_score)


# In[ ]:




