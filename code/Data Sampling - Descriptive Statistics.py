#!/usr/bin/env python
# coding: utf-8

# In[120]:


import pandas as pd


# In[121]:


df_X_train_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_train_transformed.csv')
df_X_train_transformed


# In[122]:


df_X_train_transformed.describe().T


# In[123]:


df_X_train_transformed['subreddit'].value_counts(1)


# In[124]:


df_Y_train_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_train_transformed.csv')
df_Y_train_transformed


# In[125]:


df_Y_train_transformed['label'].value_counts(1)


# In[126]:


df_X_test_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_test_transformed.csv')
df_X_test_transformed


# In[127]:


df_X_test_transformed.describe().T


# In[128]:


df_Y_test_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_test_transformed.csv')
df_Y_test_transformed


# In[129]:


df_X_val_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_val_transformed.csv')
df_X_val_transformed


# In[130]:


df_X_val_transformed.describe().T


# In[131]:


df_Y_val_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_val_transformed.csv')
df_Y_val_transformed


# In[132]:


# print('Transformed Data Subr')
print(df_X_train_transformed['subreddit'].value_counts(1))
print(df_X_test_transformed['subreddit'].value_counts(1))
print(df_X_val_transformed['subreddit'].value_counts(1))


# In[133]:


df_X_train_transformed['subreddit'].value_counts(1)[0]


# In[150]:


import numpy as np
df_transformed_temp = pd.DataFrame({'AskReddit subreddit proportion': [df_X_train_transformed['subreddit'].value_counts(1)[0],
                                          df_X_val_transformed['subreddit'].value_counts(1)[0],
                                          df_X_test_transformed['subreddit'].value_counts(1)[0]],
                                    'politics subreddit proportion': [df_X_train_transformed['subreddit'].value_counts(1)[1],
                                          df_X_val_transformed['subreddit'].value_counts(1)[1],
                                          df_X_test_transformed['subreddit'].value_counts(1)[1]],
                                   'worldnews subreddit proportion': [df_X_train_transformed['subreddit'].value_counts(1)[2],
                                          df_X_val_transformed['subreddit'].value_counts(1)[2],
                                          df_X_test_transformed['subreddit'].value_counts(1)[2]]}, 
                                   index=['Training', 'Validation','Test'])
ax = df_transformed_temp.plot.bar(figsize=(19,8))

for container in ax.containers:
    ax.bar_label(container)


# In[149]:


import numpy as np
df_transformed_temp = pd.DataFrame({'Target Variable 1s proportion': [df_Y_train_transformed['label'].value_counts(1)[0],
                                          df_Y_val_transformed['label'].value_counts(1)[0],
                                          df_Y_test_transformed['label'].value_counts(1)[0]],
                                    'Target Variable 0s proportion': [df_Y_train_transformed['label'].value_counts(1)[1],
                                          df_Y_val_transformed['label'].value_counts(1)[1],
                                          df_Y_test_transformed['label'].value_counts(1)[1]]}, 
                                   index=['Training', 'Validation','Test'])
ax = df_transformed_temp.plot.bar(figsize=(20,12))

for container in ax.containers:
    ax.bar_label(container)



                                   


# In[93]:


np.random.rand(2)


# In[76]:


df_X_train_transformed['subreddit'].value_counts(1).plot(kind='barh', figsize=(8, 6))


# In[77]:


df_X_train_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_train_untransformed.csv')
df_X_train_untransformed


# In[78]:


from IPython.display import display


# In[79]:


with pd.option_context('display.max_rows', df_X_train_untransformed.shape[1]):
    display(df_X_train_untransformed.describe().T)


# In[80]:


df_Y_train_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_train_untransformed.csv')
df_Y_train_untransformed


# In[81]:


df_X_test_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_test_untransformed.csv')
df_X_test_untransformed


# In[82]:


with pd.option_context('display.max_rows', df_X_test_untransformed.shape[1]):
    display(df_X_test_untransformed.describe().T)


# In[83]:


df_Y_test_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_test_untransformed.csv')
df_Y_test_untransformed


# In[84]:


df_X_val_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_val_untransformed.csv')
df_X_val_untransformed


# In[85]:


with pd.option_context('display.max_rows', df_X_val_untransformed.shape[1]):
    display(df_X_val_untransformed.describe().T)


# In[86]:


df_Y_val_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_val_untransformed.csv')
df_Y_val_untransformed


# In[117]:


import numpy as np
df_untransformed_temp = pd.DataFrame({'AskReddit subreddit proportion': [df_X_train_untransformed['subreddit'].value_counts(1)[0],
                                          df_X_val_untransformed['subreddit'].value_counts(1)[0],
                                          df_X_test_untransformed['subreddit'].value_counts(1)[0]],
                                    'politics subreddit proportion': [df_X_train_untransformed['subreddit'].value_counts(1)[1],
                                          df_X_val_untransformed['subreddit'].value_counts(1)[1],
                                          df_X_test_untransformed['subreddit'].value_counts(1)[1]],
                                   'worldnews subreddit proportion': [df_X_train_untransformed['subreddit'].value_counts(1)[2],
                                          df_X_val_untransformed['subreddit'].value_counts(1)[2],
                                          df_X_test_untransformed['subreddit'].value_counts(1)[2]],
                                   'Target Variable 1s proportion': [df_Y_train_untransformed['label'].value_counts(1)[0],
                                          df_Y_val_untransformed['label'].value_counts(1)[0],
                                          df_Y_test_untransformed['label'].value_counts(1)[0]],
                                    'Target Variable 0s proportion': [df_Y_train_untransformed['label'].value_counts(1)[1],
                                          df_Y_val_untransformed['label'].value_counts(1)[1],
                                          df_Y_test_untransformed['label'].value_counts(1)[1]]}, 
                                   index=['Training', 'Validation','Test'])
ax = df_untransformed_temp.plot.bar(figsize=(33,14))

for container in ax.containers:
    ax.bar_label(container)


# In[153]:


import numpy as np
df_untransformed_temp = pd.DataFrame({'AskReddit subreddit proportion': [df_X_train_untransformed['subreddit'].value_counts(1)[0],
                                          df_X_val_untransformed['subreddit'].value_counts(1)[0],
                                          df_X_test_untransformed['subreddit'].value_counts(1)[0]],
                                    'politics subreddit proportion': [df_X_train_untransformed['subreddit'].value_counts(1)[1],
                                          df_X_val_untransformed['subreddit'].value_counts(1)[1],
                                          df_X_test_untransformed['subreddit'].value_counts(1)[1]],
                                   'worldnews subreddit proportion': [df_X_train_untransformed['subreddit'].value_counts(1)[2],
                                          df_X_val_untransformed['subreddit'].value_counts(1)[2],
                                          df_X_test_untransformed['subreddit'].value_counts(1)[2]]}, 
                                   index=['Training', 'Validation','Test'])
ax = df_untransformed_temp.plot.bar(figsize=(19,8))

for container in ax.containers:
    ax.bar_label(container)


# In[154]:


import numpy as np
df_untransformed_temp = pd.DataFrame({'Target Variable 1s proportion': [df_Y_train_untransformed['label'].value_counts(1)[0],
                                          df_Y_val_untransformed['label'].value_counts(1)[0],
                                          df_Y_test_untransformed['label'].value_counts(1)[0]],
                                    'Target Variable 0s proportion': [df_Y_train_untransformed['label'].value_counts(1)[1],
                                          df_Y_val_untransformed['label'].value_counts(1)[1],
                                          df_Y_test_untransformed['label'].value_counts(1)[1]]}, 
                                   index=['Training', 'Validation','Test'])
ax = df_untransformed_temp.plot.bar(figsize=(20,12))

for container in ax.containers:
    ax.bar_label(container)



                                   


# In[ ]:




