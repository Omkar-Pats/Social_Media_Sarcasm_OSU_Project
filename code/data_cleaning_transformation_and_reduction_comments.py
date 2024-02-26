#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[96]:


##using absolute path. please change for the rerun
df_comment = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-1-cia/data/vectorized_text_data/comments_data_vectorized.csv')
df_comment


# In[97]:


df_comment = df_comment.drop(df_comment.tail(2).index)
df_comment


# In[98]:


df_comment.isna().sum()


# In[99]:


df_comment['cleaned_comment']


# In[100]:


df_comment.columns


# In[101]:


df_comment.loc[df_comment.isnull().any(axis=1)]


# In[102]:


## droppping the cleaned_comment and comment variable as we only to work with the numerical variables for analysis
df_comment_1 = df_comment.drop(['comment','cleaned_comment','id'],axis=1)


# In[103]:


df_comment_1


# In[104]:


df_comment_1.describe()


# In[105]:


## checking the importance of the each of the terms with respect to their tfidf score

df_comment_1.mean().sort_values()


# In[106]:


## dropping only 2 records, not gonna affect
df_comment_2 = df_comment_1.dropna()
df_comment_2


# In[107]:


df_comment_2.isnull().sum()


# In[108]:


## we are going ahead and standardizing the variables before PCA
## since it is part of the descriptive process we are not going to split the data before getting the mean and sd

from scipy.stats import zscore
data_scaled=df_comment_2.apply(zscore)
data_scaled


# In[109]:


data_scaled.describe()


# In[110]:


## Building the covariance matrix
import numpy as np

cov_matrix = np.cov(data_scaled.T)
print('Covariance Matrix \n%s', cov_matrix)


# In[111]:


## get the eigen values and the eigen vectors

eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print('Eigen Vectors \n', eig_vecs)
print('\n Eigen Values \n', eig_vals)


# In[112]:


## calculate the variance explained by eigen values and the cumulative variance by the eigen values.

tot = sum(eig_vals)
var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]
print(var_exp)


# In[113]:


## cumulative variance explained

cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)


# In[114]:


cum_var_exp[30]


# In[115]:


plt.plot(var_exp)
plt.grid()


# In[116]:


# Ploting -- trying to check the variance and the cumulative variance with respect to the number of components
plt.figure(figsize=(15 , 8))
plt.bar(range(1, eig_vals.size + 1), var_exp, alpha = 0.5, align = 'center', label = 'Individual explained variance')
plt.step(range(1, eig_vals.size + 1), cum_var_exp, where='mid', label = 'Cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()


# In[117]:


from statsmodels.multivariate.pca import PCA


# In[118]:


pc = PCA(df_comment_2, 
         ncomp=20,
         standardize=True,  
         normalize=True,    
         missing=None,
         method='eig')


# In[119]:


df_comp = pc.loadings.T
df_comp


# In[120]:


df_reduced_dimension = pc.factors
df_reduced_dimension.head()


# In[121]:


df_reduced_dimension = df_reduced_dimension.add_suffix('_comments')
df_reduced_dimension


# In[122]:


df_reduced_dimension.insert(loc=0, column='id', value=df_comment['id'])
df_reduced_dimension.insert(loc=1, column='cleaned_comment', value=df_comment['cleaned_comment'])
df_reduced_dimension.insert(loc=2, column='comment', value=df_comment['comment'])


# In[123]:


df_reduced_dimension


# In[124]:


df_reduced_dimension.to_csv('data/pca_data/df_comments_PCA_20.csv',index=False)


# In[ ]:




