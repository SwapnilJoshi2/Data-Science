#!/usr/bin/env python
# coding: utf-8

# # Question No 1

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[2]:


df=pd.read_csv('50_Startups.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.isna().sum()


# # Corelation Matrix

# In[5]:


df.corr()


# # Scatterplot between variables along with histograms

# In[6]:


sns.set_style(style='darkgrid')
sns.pairplot(df)


# In[7]:


df.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace=True)


# # Preparing a Model

# In[8]:


import statsmodels.formula.api as smf 
model = smf.ols('Profit~RnD+Administration+Marketing+State',data=df).fit()


# In[9]:


model.params


# In[10]:


print(model.tvalues, '\n', model.pvalues)


# In[11]:


(model.rsquared,model.rsquared_adj)


# # Simple Linear Regression Models

# In[12]:


ml_v=smf.ols('Profit~RnD',data = df).fit()  
#t and p-Values
print(ml_v.tvalues, '\n', ml_v.pvalues) 


# In[13]:


ml_v=smf.ols('Profit~Marketing',data = df).fit()  
#t and p-Values
print(ml_v.tvalues, '\n', ml_v.pvalues) 


# In[14]:


ml_v=smf.ols('Profit~Administration',data = df).fit()  
#t and p-Values
print(ml_v.tvalues, '\n', ml_v.pvalues) 


# In[15]:


ml_v=smf.ols('Profit~State',data = df).fit()  
#t and p-Values
print(ml_v.tvalues, '\n', ml_v.pvalues) 


# In[16]:


ml_v=smf.ols('Profit~Administration+Marketing+RnD',data = df).fit()  
#t and p-Values
print(ml_v.tvalues, '\n', ml_v.pvalues) 


# In[17]:


ml_v=smf.ols('Profit~Administration+Marketing',data = df).fit()  
#t and p-Values
print(ml_v.tvalues, '\n', ml_v.pvalues) 


# # Calculating VIF

# In[18]:


df['State'].replace({'New York': '1', 'California': '2', 'Florida': '3'},inplace= True)
df.head()


# In[19]:


rsq_hp = smf.ols('RnD~Administration+Marketing+State',data=df).fit().rsquared  
vif_hp = 1/(1-rsq_hp) # 16.33

rsq_wt = smf.ols('Administration~RnD+Marketing+State',data=df).fit().rsquared  
vif_wt = 1/(1-rsq_wt) # 564.98

rsq_vol = smf.ols('Marketing~RnD+Administration+State',data=df).fit().rsquared  
vif_vol = 1/(1-rsq_vol) #  564.84

rsq_sp = smf.ols('State~RnD+Administration+Marketing',data=df).fit().rsquared  
vif_sp = 1/(1-rsq_sp) #  16.35

# Storing vif values in a data frame
d1 = {'Variables':['RnD','Administration','Marketing,State'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# # Residual Analysis

# ## Test for Normality of Residuals (Q-Q Plot)

# In[20]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[21]:


list(np.where(model.resid>1200))


# # Residual Plot for Homoscedasticity

# In[22]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[23]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# # #Residual Vs Regressors

# In[24]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "RnD", fig=fig)
plt.show()


# In[25]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Administration", fig=fig)
plt.show()


# In[26]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Marketing", fig=fig)
plt.show()


# # Model Deletion Diagnostics

# ## Detecting Influencers/Outliers

# ### Cook’s Distance

# In[75]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[28]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[29]:


(np.argmax(c),np.max(c))


# ## High Influence points

# In[30]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[1]:


k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)


# ### From the above plot , it is evident that data point 48 and 49 are the influencers

# In[32]:


df[df.index.isin([48,49])]


# In[33]:


df.head()


# # Improving the Model

# In[34]:


df_new=pd.read_csv('50_Startups.csv')
df_new.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace=True)


# In[35]:


df1=df_new.drop(df_new.index[[48,49]],axis=0).reset_index()
df1.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace=True)


# In[36]:


df1.drop(['index'],axis=1)
df1


# # Build Model

# In[37]:


final_ml_V=smf.ols('Profit~RnD+Administration+Marketing',data=df1).fit()


# In[38]:


(final_ml_V.rsquared,final_ml_V.aic)


# #### Comparing above R-Square and AIC values, model 'final_ml_V' has high R- square and low AIC value hence include variable 'VOL' so that multi collinearity problem would be resolved

# # Cook's Distance

# In[39]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[40]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df1)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[41]:


(np.argmax(c_V),np.max(c_V))


# In[42]:


df1[df1.index.isin([45,46])]


# In[43]:


df2=df1.drop(df1.index[[45,46]],axis=0)
df2


# In[44]:


df3=df2.reset_index()


# In[45]:


df4=df3.drop(['index'],axis=1)


# In[46]:


df4


# In[47]:


final_ml_V=smf.ols('Profit~RnD+Administration+Marketing',data=df4).fit()


# In[48]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[49]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df4)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[50]:


(np.argmax(c_V),np.max(c_V))


# In[51]:


final_ml_V=smf.ols('Profit~RnD+Administration+Marketing',data=df4).fit()


# In[52]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[53]:


df5=df4.drop(df1.index[[3,14,15,4,19,27,36,38]],axis=0)
df5 


# In[54]:


df6=df5.reset_index()


# In[55]:


df7=df6.drop(['index'],axis=1)


# In[56]:


df7


# In[57]:


final_ml_V=smf.ols('Profit~RnD+Administration+Marketing',data=df7).fit()


# In[58]:


model_influence_V = final_ml_V.get_influence()
(c_Vv, _) = model_influence_V.cooks_distance


# In[59]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df7)),np.round(c_Vv,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[60]:


final_ml_V=smf.ols('Profit~RnD+Administration+Marketing',data=df7).fit()


# In[61]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[62]:


df8=df7.drop(df1.index[[2,3,8,9,16,28,35]],axis=0)
df8 


# In[63]:


df9=df8.reset_index()


# In[64]:


df10=df9.drop(['index'],axis=1)
df10


# In[65]:


final_ml_V=smf.ols('Profit~RnD+Administration+Marketing',data=df10).fit()


# In[66]:


model_influence_V = final_ml_V.get_influence()
(c_Vvv, _) = model_influence_V.cooks_distance


# In[67]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df10)),np.round(c_Vvv,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[68]:


final_ml_V=smf.ols('Profit~RnD+Administration+Marketing',data=df10).fit()


# In[69]:


(final_ml_V.rsquared,final_ml_V.aic)


# # Predicting New value

# In[70]:


new_data=pd.DataFrame({'RnD':165349.20,"Administration":136897.80,"Marketing":471784.10,"State":'NewYork'},index=[1])


# In[71]:


final_ml_V.predict(new_data)


# In[72]:


final_ml_V.predict(df_new.iloc[0:5,])


# In[73]:


pred_y = final_ml_V.predict(df_new)


# In[74]:


pred_y


# 
