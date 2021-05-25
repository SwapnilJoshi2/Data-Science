#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[2]:


df=pd.read_csv('Toyoto_Corrola.csv')
df


# In[3]:


df.drop('Id',axis=1,inplace=True)
df.drop('Doors',axis=1,inplace=True)
df.drop('Cylinders',axis=1,inplace=True)
df.drop('Gears',axis=1,inplace=True)
df.drop('Weight',axis=1,inplace=True)
df.info()


# In[4]:


df


# In[5]:


df.isna().sum()


# In[6]:


df.corr()


# In[7]:


sns.set_style(style='darkgrid')
sns.pairplot(df)


# In[8]:


import statsmodels.formula.api as smf
model=smf.ols('Price~+Age_08_04+KM+HP',data=df).fit()


# In[9]:


model.params


# In[10]:


print(model.tvalues,'\n',model.pvalues)


# In[11]:


(model.rsquared,model.rsquared_adj)


# # Simple Regression Model

# In[12]:


ml_v=smf.ols('Price~KM',data=df).fit()
print(ml_v.tvalues,ml_v.pvalues)


# In[13]:


ml_v=smf.ols('Price~HP',data=df).fit()
print(ml_v.tvalues,ml_v.pvalues)


# In[14]:


ml_v=smf.ols('Price~Age_08_04',data=df).fit()
print(ml_v.tvalues,ml_v.pvalues)


# In[15]:


#ml_v=smf.ols('Price~Model',data=df).fit()
#print(ml_v.tvalues,ml_v.pvalues)


# In[16]:


ml_v=smf.ols('Price~HP+KM',data=df).fit()
print(ml_v.tvalues,ml_v.pvalues)


# In[17]:


ml_v=smf.ols('Price~Age_08_04+KM',data=df).fit()
print(ml_v.tvalues,ml_v.pvalues)


# In[18]:


ml_v=smf.ols('Price~Age_08_04+Model',data=df).fit()
print(ml_v.tvalues,ml_v.pvalues)


# ## Calculating VIF

# In[19]:




rsq_wt = smf.ols('Model~Age_08_04+KM+HP',data=df).fit().rsquared  
vif_wt = 1/(1-rsq_wt) # 564.98

rsq_vol = smf.ols('Age_08_04~Model+KM+HP',data=df).fit().rsquared  
vif_vol = 1/(1-rsq_vol) #  564.84

rsq_sp = smf.ols('KM~Age_08_04+Model+HP',data=df).fit().rsquared  
vif_sp = 1/(1-rsq_sp) #  16.35

rsq_sq = smf.ols('HP~Age_08_04+Model+KM',data=df).fit().rsquared  
vif_sq = 1/(1-rsq_sq) #  16.35

# Storing vif values in a data frame
d1 = {'Variables':['Model','Age_08_04','KM,HP'],'VIF':[vif_wt,vif_vol,vif_sp,vif_sq]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# # Residual Analysis

# ## Test for Normality of Residuals (Q-Q Plot)

# In[20]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q')
plt.title('Normal Q-Q Plot Residuals')
plt.show()


# In[21]:


list(np.where(model.resid>2800))


# # Residual Plots for Homoscedasticity

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


# # Residual vs Regressors

# In[24]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "HP", fig=fig)
plt.show()


# In[25]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "KM", fig=fig)
plt.show()


# In[26]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Age_08_04", fig=fig)
plt.show()


# # Model Detection Diagnotics

# ## Detecting Outliers

# ### Cook's Distance

# In[27]:


model_influence = model.get_influence()
(c,_) = model_influence.cooks_distance


# In[28]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(c,3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[29]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[30]:


(np.argmax(c),np.max(c))


# In[31]:


df.boxplot(figsize=('10','10'))


# In[32]:


df.KM.mean()


# In[33]:


df.KM.std()


# In[34]:


df.Price.mean()


# In[35]:


df.Price.std()


# In[36]:


(np.argmax(c),np.max(c))


# In[37]:


df[df.index.isin([2])]


# In[38]:


df1=df.drop(df.index[[2]],axis=0).reset_index()


# In[39]:


df1.drop(['index'],axis=1)
df1


# In[40]:


final_ml_V=smf.ols('Price~Age_08_04+KM+HP',data=df1).fit()


# In[41]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[42]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[43]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df1)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[44]:


(np.argmax(c_V),np.max(c_V))


# In[45]:


df1[df1.index.isin([5])]


# In[46]:


df2=df1.drop(df1.index[[5]],axis=0)
df2


# In[47]:


df3=df2.reset_index()


# In[48]:


df4=df3.drop(['index'],axis=1)


# In[49]:


df4


# In[50]:


final_ml_V=smf.ols('Price~Age_08_04+KM+HP',data=df4).fit()


# In[51]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[52]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[53]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df4)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[54]:


(np.argmax(c_V),np.max(c_V))


# In[55]:


df5=df4.drop(df1.index[[63]],axis=0)
df5 


# In[56]:


df6=df5.reset_index()


# In[57]:


df7=df6.drop(['index'],axis=1)


# In[58]:


df7


# In[63]:


final_ml_V=smf.ols('Price~+Age_08_04+KM+HP',data=df7).fit()


# In[64]:


(final_ml_V.rsquared,final_ml_V.aic)


# # Predicting New Value

# In[65]:


new_data=pd.DataFrame({"Age_08_04":23,"KM":46986,"HP":90},index=[1])


# In[66]:


final_ml_V.predict(new_data)


# In[68]:


final_ml_V.predict(df.iloc[0:5,])


# In[70]:


pred_y = final_ml_V.predict(df)


# In[72]:


pred_y


# In[ ]:




