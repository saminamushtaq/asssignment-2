#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# In[2]:


import pandas as pd
df=pd.read_csv('temp.csv')


# In[3]:



df[['Month', 'Day']] = df[['Month', 'Day']].astype('int8')
df[['Year']] = df[['Year']].astype('int16')
df['AvgTemperature'] = df['AvgTemperature'].astype('float16')


# In[4]:



df.reset_index(drop=True, inplace=True)


# In[5]:



df = df[df.Year!=200]
df = df[df.Year!=201]
df = df[df.Year!=2020]


# In[6]:


df = df.drop('State', axis=1)


# In[7]:


df['Date'] = pd.to_datetime(df.Year.astype(str) + '/' + df.Month.astype(str))


# In[8]:


missing = pd.DataFrame(df.loc[df.AvgTemperature == -99, 'Country'].value_counts())
missing['TotalData'] = df.groupby('Country').AvgTemperature.count()
missing['PercentageMissing'] = missing.apply(lambda row: (row.Country/row.TotalData)*100, axis=1)
missing.sort_values(by=['PercentageMissing'], inplace=True, ascending=False)
missing.head(10)


# In[11]:



import numpy as np
df.loc[df.AvgTemperature == -99, 'AvgTemperature'] = np.nan


# In[12]:


print(df.AvgTemperature.isna().sum())


# In[13]:


df['AvgTemperature'] = df['AvgTemperature'].fillna(df.groupby(['City', 'Date']).AvgTemperature.transform('mean'))


# In[14]:


print(df.AvgTemperature.isna().sum())


# In[15]:


print(df.loc[df.AvgTemperature.isna(), 'City'].value_counts()[:10])


# In[16]:


df['AvgTemperature'] = df['AvgTemperature'].fillna(df.groupby(['City']).AvgTemperature.transform('mean'))


# In[17]:



print(df.AvgTemperature.isna().sum())


# In[18]:


df['AvgTempCelsius'] = (df.AvgTemperature -32)*(5/9)
df = df.drop("AvgTemperature", axis=1)


# In[19]:


df['AvgTempCelsius_rounded'] = df.AvgTempCelsius.apply(lambda x: "{0:0.2f}".format(x))
df['AvgTempCelsius_rounded2'] = df.AvgTempCelsius.apply(lambda x: "{0:0.1f}".format(x))


# In[20]:



df['AvgTempCelsius_rounded'] = pd.to_numeric(df['AvgTempCelsius_rounded'])
df['AvgTempCelsius_rounded2'] = pd.to_numeric(df['AvgTempCelsius_rounded2'])


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,4))
sns.lineplot(x='Year', y='AvgTempCelsius', data=df , palette='Set2')
ax.set_title('Average Global Temperatures', fontsize=18)
ax.set_ylabel('Average Temperature (°C)', fontsize=14)
ax.set_xlabel('')
ax.set_xticks(range(1995, 2020))
for index, label in enumerate(ax.xaxis.get_ticklabels()):
    if index % 5 != 0:
        label.set_visible(False)
plt.tight_layout()
plt.show();


# In[23]:


df_mean_month = df.groupby(['Month', 'Year']).AvgTempCelsius_rounded2.mean()
df_mean_month = df_mean_month.reset_index()
df_mean_month = df_mean_month.sort_values(by=['Year'])


# In[24]:


df_pivoted = pd.pivot_table(
    data=df_mean_month,
    index='Month',
    values='AvgTempCelsius_rounded2',
    columns='Year'
)


# In[36]:


fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(data = df_pivoted, cmap='coolwarm', annot=True, fmt=".1f", annot_kws={'size':11})
plt.xlabel('')
plt.ylabel('')
ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.title('Average Global Temperatures (°C)', fontsize=18)
plt.show();


# In[28]:


s = df.groupby(['Region'])['AvgTempCelsius'].mean().reset_index().sort_values(by='AvgTempCelsius', ascending=False)
s.style.background_gradient(cmap="RdBu_r")


# In[35]:



f, ax = plt.subplots(figsize=(10,4))
sns.lineplot(x='Year', y='AvgTempCelsius', hue='Region', data=df , palette='Set2')
plt.title('Average Temperature in Different Regions', fontsize=18)
plt.ylabel('Average Temperature (°C)', fontsize=14)
plt.xlabel('')
plt.xticks(range(1995, 2020))
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1)
for index, label in enumerate(ax.xaxis.get_ticklabels()):
    if index % 5 != 0:
        label.set_visible(False)
plt.tight_layout()
plt.show();


# In[32]:


df_europe = df[df.Region == 'Europe'].copy()
f, ax = plt.subplots(figsize=(10, 5))
sns.distplot(df_europe.AvgTempCelsius_rounded, bins = 20)
plt.title('Distribution of Temperatures in Europe (1995-2019)', fontsize=18)
plt.xlabel('Temperature (°C)', fontsize=14)
ax.axes.yaxis.set_visible(False)
ax.axes.yaxis.set_ticklabels([''])
plt.show()


# In[33]:


f, ax = plt.subplots(figsize=(10, 5))
sns.distplot(df_europe.AvgTempCelsius_rounded, bins = 20)
plt.title('Distribution of Temperatures in Europe (1995-2019)', fontsize=18)
plt.xlabel('Temperature (°C)', fontsize=14)
ax.axes.yaxis.set_visible(False)
ax.axes.yaxis.set_ticklabels([''])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




