#!/usr/bin/env python
# coding: utf-8

# # Question No 1

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df1=pd.read_csv('delivery_time.csv')
df1.head()


# In[3]:


df1.iloc[:, 1]


# In[4]:


df1.columns


# In[5]:


df1.info()


# In[6]:


df1=df1.rename(columns={'Delivery Time': 'dt','Sorting Time': 'st' })
df1.head()


# In[7]:


sns.distplot(df1['dt'])


# In[8]:


sns.distplot(df1['st'])


# In[9]:


plt.scatter(x='st', y='dt',data=df1, color='red')
plt.xlabel("Sorting time")
plt.ylabel("Delivery time")


# In[10]:


sns.regplot(x="st", y="dt", data=df1,color='red')


# In[11]:


plt.boxplot('st',data=df1)


# In[12]:


plt.boxplot('dt',data=df1)


# In[13]:


df1.corr()


# In[14]:


import statsmodels.formula.api as smf
model1 = smf.ols("dt~st",data = df1).fit()


# In[15]:


model1.params


# In[16]:


print(model1.tvalues, '\n', model1.pvalues) 


# In[17]:


(model1.rsquared,model1.rsquared_adj)


# In[18]:


newdata=pd.Series([10,25])


# In[19]:


data_pred=pd.DataFrame(newdata,columns=['st'])


# In[20]:


model1.predict(data_pred)


# In[ ]:





# # Question No 2

# In[21]:


df=pd.read_csv('Salary_Data.csv')
df.head()


# In[22]:


df.info()


# In[23]:


df=df.rename(columns={'YearsExperience': 'ye','Salary': 's' })
df.head()


# In[24]:


sns.distplot(df['ye'])


# In[25]:


sns.distplot(df['s'])


# In[26]:


plt.scatter(x='s', y='s',data=df, color='red')
plt.xlabel("Salary")
plt.ylabel("Years of Experience")


# In[27]:


sns.regplot(x="s", y="ye", data=df,color='red')
plt.xlabel("Salary")
plt.ylabel("Years of Experience")


# In[28]:


plt.boxplot('s',data=df)


# In[29]:


plt.boxplot('ye',data=df)


# In[30]:


df.corr()


# In[31]:


import statsmodels.formula.api as smf
model = smf.ols("s~ye",data = df).fit()


# In[32]:


model.params


# In[33]:


print(model.tvalues, '\n', model.pvalues)  


# In[34]:


(model.rsquared,model.rsquared_adj)


# In[35]:


newdata=pd.Series([5,7])


# In[36]:


data_pred=pd.DataFrame(newdata,columns=['ye'])


# In[37]:


model.predict(data_pred)


# In[ ]:




