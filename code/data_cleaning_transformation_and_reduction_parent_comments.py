#!/usr/bin/env python
# coding: utf-8

# In[203]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[204]:


##using absolute path. please change for the rerun
df_comment = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-1-cia/data/vectorized_text_data/parent_comments_data_vectorized.csv')
df_comment


# In[205]:


df_comment = df_comment.drop(df_comment.tail(2).index)
df_comment


# In[206]:


df_comment.isna().sum()


# In[207]:


df_comment['cleaned_parent_comment']


# In[208]:


df_comment.columns


# In[209]:


df_comment.loc[df_comment.isnull().any(axis=1)]


# In[210]:


## droppping the cleaned_comment and comment variable as we only to work with the numerical variables for analysis
df_comment_1 = df_comment.drop(['parent_comment','cleaned_parent_comment','id'],axis=1)


# In[211]:


df_comment_1


# In[212]:


df_comment_1.describe()


# In[213]:


## checking the importance of the each of the terms with respect to their tfidf score

df_comment_1.mean().sort_values()


# In[214]:


df_comment_1.isnull()


# In[215]:


## dropping only 7 records, not gonna affect
df_comment_2 = df_comment_1.dropna()
df_comment_2


# In[216]:


df_comment_2.isnull().sum()


# In[217]:


## we are going ahead and standardizing the variables before PCA
## since it is part of the descriptive process we are not going to split the data before getting the mean and sd

from scipy.stats import zscore
data_scaled=df_comment_2.apply(zscore)
data_scaled


# In[218]:


data_scaled.describe()


# In[219]:


## Building the covariance matrix
import numpy as np

cov_matrix = np.cov(data_scaled.T)
print('Covariance Matrix \n%s', cov_matrix)


# In[220]:


## get the eigen values and the eigen vectors

eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print('Eigen Vectors \n', eig_vecs)
print('\n Eigen Values \n', eig_vals)


# In[221]:


## calculate the variance explained by eigen values and the cumulative variance by the eigen values.

tot = sum(eig_vals)
var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]
print(var_exp)


# In[222]:


## cumulative variance explained

cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)


# In[223]:


cum_var_exp[25]


# In[224]:


len(var_exp)


# In[225]:


plt.plot(var_exp)
plt.grid()


# In[226]:


# Ploting -- trying to check the variance and the cumulative variance with respect to the number of components
plt.figure(figsize=(15 , 8))
plt.bar(range(1, eig_vals.size + 1), var_exp, alpha = 0.5, align = 'center', label = 'Individual explained variance')
plt.step(range(1, eig_vals.size + 1), cum_var_exp, where='mid', label = 'Cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc = 'best')
plt.tight_layout()
plt.grid()
plt.show()


# In[227]:


from statsmodels.multivariate.pca import PCA


# In[228]:


pc = PCA(df_comment_2, 
         ncomp=15,
         standardize=True,  
         normalize=True,    
         missing=None,
         method='eig')


# In[229]:


df_comp = pc.loadings.T
df_comp


# In[230]:


df_reduced_dimension = pc.factors
df_reduced_dimension.head()


# In[231]:


df_reduced_dimension


# In[232]:


df_reduced_dimension = df_reduced_dimension.add_suffix('_p_comments')
df_reduced_dimension


# In[233]:


df_reduced_dimension.insert(loc=0, column='id', value=df_comment['id'])
df_reduced_dimension.insert(loc=1, column='cleaned_parent_comment', value=df_comment['cleaned_parent_comment'])
df_reduced_dimension.insert(loc=2, column='parent_comment', value=df_comment['parent_comment'])


# In[234]:


df_reduced_dimension


# In[235]:


df_reduced_dimension.to_csv('data/pca_data/df_p_comments_PCA_15.csv',index=False)


# In[ ]:




