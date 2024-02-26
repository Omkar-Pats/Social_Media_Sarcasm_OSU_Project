#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


## reading the transformed datasets

df_X_train_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_train_transformed.csv')
df_Y_train_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_train_transformed.csv')
df_X_test_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_test_transformed.csv')
df_Y_test_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_test_transformed.csv')
df_X_val_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_val_transformed.csv')
df_Y_val_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_val_transformed.csv')


# In[3]:


df_X_train_transformed.columns


# In[4]:


## Functions to maintain equity in the data prepreocessing step

def transformation_before_modeling_helper(_df):
    '''
    Function for basic transformation before modeling - One Hot Encoding/Dummies
    '''
    _df_v2 = _df.drop(
        ['author', 'cleaned_comment', 'cleaned_parent_comment', 'comment', 'parent_comment', 'id', 'date', 'subreddit', 'topic'],
        axis=1
    )

    _sub_temp_df = pd.get_dummies(
        _df['subreddit'],
        prefix='sub_name',
    )

    _topic_temp_df = pd.get_dummies(
        _df['topic'],
        prefix='topic_label',
    )

    transformed_df = pd.concat([
        _df_v2, _sub_temp_df, _topic_temp_df
    ], axis=1)

    return transformed_df


def plot_roc_curve(y_actual, y_preds_probab, set_name='', pos_label=''):
    '''
    Function for plotting the ROC Curve
    '''
    fpr, tpr, threshold = roc_curve(y_actual, y_preds_probab, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    sns.set_theme(style="white", palette=None)
    plt.title(set_name + ' - ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
def plot_conf_matrix(actual_labels, predicted_labels, labels = ['NT', 'T'], cmap='viridis', title=''):
    '''
    Function for plotting the confusion matrix
    '''
    conf_matrix = confusion_matrix(
        actual_labels,
        predicted_labels
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(conf_matrix, cmap=cmap)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actuals")

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()


# In[21]:


# df_X_model_train_trans = transformation_before_modeling_helper(df_X_train_transformed)
# df_X_model_val_trans = transformation_before_modeling_helper(df_X_val_transformed)
# df_X_model_test_trans = transformation_before_modeling_helper(df_X_test_transformed)


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix


# In[18]:


## Getting the best logistic regression parameters

def choose_best_lr_model(X_train_df, X_val_df, Y_train_df, Y_val_df, params_dict_search):
    '''
    Function for hyperparameter tuning of Logistic Regression Model and getting the best model.
    Uses Randomized Search CV. Best model is chosen based on the performance on the validation set.

    Returns:
        model estimator object
    '''

    lr_model = LogisticRegression(
        n_jobs=-1,
        random_state=1234,
        max_iter=10000
    )

    X_train_val_combined = pd.concat([X_train_df, X_val_df]).reset_index(drop=True)
    Y_train_val_combined = pd.concat([Y_train_df['label'], Y_val_df['label']]).reset_index(drop=True)
    train_val_indices = [-1]*X_train_df.shape[0] + ([1] * X_val_df.shape[0])

    ps = PredefinedSplit(train_val_indices)


    lr_cv = RandomizedSearchCV(
        estimator=lr_model,
        param_distributions=params_dict_search,
        n_jobs=-1,
        random_state=1234,
        cv=ps,
        verbose=3,
        scoring=['roc_auc', 'f1'],
        refit='roc_auc'
    )
    print("Parameter Options:",)
    print(lr_cv)
    print("---"*30, end="\n\n")

    hyper_param_results = lr_cv.fit(
        X_train_val_combined,
        Y_train_val_combined
    )
    
    lr_model_tuning_results_v1 = pd.DataFrame(hyper_param_results.cv_results_)
    lr_model_tuning_results_v1.to_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/lr_model_v1_param_tuning_results.csv', index=False)

    print(
        "Best Hyperparameters for LR Model 1 (PCA Transformed Dataset)\n\n",
        hyper_param_results.best_estimator_
    )

    print("---"*30, end="\n\n")

    print(
        "Best AUC Score for LR Model 1 (PCA Transformed Dataset)\n\n",
        hyper_param_results.best_score_
    )
    print("---"*30, end="\n\n")
    
    return hyper_param_results.best_estimator_


# In[66]:


## Building the logistic regresison model and getting the most optimum parmeters
## for the validation set performance

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

X_train_df = transformation_before_modeling_helper(df_X_train_transformed)
X_val_df = transformation_before_modeling_helper(df_X_val_transformed)
X_test_df = transformation_before_modeling_helper(df_X_test_transformed)


X_train_df = pd.DataFrame(SS.fit_transform(X_train_df),columns=X_train_df.columns)
X_val_df = pd.DataFrame(SS.transform(X_val_df),columns=X_val_df.columns)
X_test_df = pd.DataFrame(SS.transform(X_test_df),columns=X_test_df.columns)

print("Shape of Training, Validation, and Test Datasets Respectively: ")
print(X_train_df.shape, X_val_df.shape, X_test_df.shape)
print("---"*30, end="\n\n")

params_list_rf = {
    'penalty': ['l2','none'],
    'C': np.linspace(1, 5, num=1),
    'solver': ['newton-cg','lbfgs','saga']
}

best_lr_model_v1_transformed_data = choose_best_lr_model(
    X_train_df = X_train_df, 
    X_val_df = X_val_df, 
    Y_train_df = df_Y_train_transformed,
    Y_val_df = df_Y_val_transformed, 
    params_dict_search = params_list_rf
)

print("Selecting Best Model and Making Predictions on Test Set...")

final_lr_model_transf = best_lr_model_v1_transformed_data
test_transf_preds_probab = final_lr_model_transf.predict_proba(X_test_df)[:,1]
test_transf_preds_labels = final_lr_model_transf.predict(X_test_df)

print("---"*30, end="\n\n")


# In[67]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[68]:


## Evaluation on the test set

print("Classification Evaluation Report - Test Set Metrics: (1 = Sarcasm)")
print(classification_report(df_Y_test_transformed['label'], test_transf_preds_labels))
print("---"*30, end="\n\n")


plt.figure(figsize=(6,6))
plot_roc_curve(
    df_Y_test_transformed['label'],
    test_transf_preds_probab, 
    'Test Set (Model with PCA Transformed Data)', 
    1
)
print("---"*30, end="\n\n")


plt.figure(figsize=(6,6))
plot_conf_matrix(
    df_Y_test_transformed['label'].map({0:'Not Sarcasm', 1:'Sarcasm'}),
    np.where(test_transf_preds_labels==1, 'Sarcasm', 'Not Sarcasm'), 
    ['Not Sarcasm', 'Sarcasm'],
    'Blues',
    'Test Set (Model with PCA Transformed Data) - Confusion Matrix'
)

plt.show()


# In[81]:


pd.DataFrame({'Feature Names':final_lr_model_transf.feature_names_in_,'Values':np.abs(final_lr_model_transf.coef_[0])}).sort_values(by='Values',ascending=False).head(30).set_index('Feature Names').plot(kind='bar', figsize=(16, 7), rot=90,grid=True);


# In[76]:


plt.figure(figsize=(16, 8))
plt.bar(final_lr_model_transf.feature_names_in_[0:30],np.abs(final_lr_model_transf.coef_[0])[0:30],)


# # Model 2 - Untransformed data

# In[57]:


## reading the transformed datasets

df_X_train_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_train_untransformed.csv')
df_Y_train_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_train_untransformed.csv')
df_X_test_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_test_untransformed.csv')
df_Y_test_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_test_untransformed.csv')
df_X_val_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_val_untransformed.csv')
df_Y_val_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_val_untransformed.csv')


# In[62]:


## Building the logistic regresison model and getting the most optimum parmeters
## for the validation set performance

## untransformed variables

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

X_train_df = transformation_before_modeling_helper(df_X_train_untransformed)
X_val_df = transformation_before_modeling_helper(df_X_val_untransformed)
X_test_df = transformation_before_modeling_helper(df_X_test_untransformed)


X_train_df = pd.DataFrame(SS.fit_transform(X_train_df),columns=X_train_df.columns)
X_val_df = pd.DataFrame(SS.transform(X_val_df),columns=X_val_df.columns)
X_test_df = pd.DataFrame(SS.transform(X_test_df),columns=X_test_df.columns)

print("Shape of Training, Validation, and Test Datasets Respectively: ")
print(X_train_df.shape, X_val_df.shape, X_test_df.shape)
print("---"*30, end="\n\n")

params_list_rf = {
    'penalty': ['l2','none'],
    'C': np.linspace(1, 5, num=1),
    'solver': ['newton-cg','lbfgs','saga']
}

best_lr_model_v2_transformed_data = choose_best_lr_model(
    X_train_df = X_train_df, 
    X_val_df = X_val_df, 
    Y_train_df = df_Y_train_untransformed,
    Y_val_df = df_Y_val_untransformed, 
    params_dict_search = params_list_rf
)

print("Selecting Best Model and Making Predictions on Test Set...")

final_lr_model_untransf = best_lr_model_v2_transformed_data
test_untransf_preds_probab = final_lr_model_untransf.predict_proba(X_test_df)[:,1]
test_untransf_preds_labels = final_lr_model_untransf.predict(X_test_df)

print("---"*30, end="\n\n")


# In[63]:


## Evaluation on the test set 

print("Classification Evaluation Report - Test Set Metrics: (1 = Sarcasm)")
print(classification_report(df_Y_test_untransformed['label'], test_untransf_preds_labels))
print("---"*30, end="\n\n")


plt.figure(figsize=(6,6))
plot_roc_curve(
    df_Y_test_untransformed['label'],
    test_untransf_preds_probab, 
    'Test Set (RF Model 2 with Untransformed Data)', 
    1
)
print("---"*30, end="\n\n")


plt.figure(figsize=(6,6))
plot_conf_matrix(
    df_Y_test_untransformed['label'].map({0:'Not Sarcasm', 1:'Sarcasm'}),
    np.where(test_untransf_preds_labels==1, 'Sarcasm', 'Not Sarcasm'), 
    ['Not Sarcasm', 'Sarcasm'],
    'Blues',
    'Test Set (RF Model 2 with Untransformed Data) - Confusion Matrix'
)


# In[64]:


pd.DataFrame({'Feature Names':final_lr_model_untransf.feature_names_in_,'Values':np.abs(final_lr_model_untransf.coef_[0])}).sort_values(by='Values',ascending=False)


# In[82]:


pd.DataFrame({'Feature Names':final_lr_model_untransf.feature_names_in_,'Values':np.abs(final_lr_model_untransf.coef_[0])}).sort_values(by='Values',ascending=False)


# In[84]:


pd.DataFrame({'Feature Names':final_lr_model_untransf.feature_names_in_,'Values':np.abs(final_lr_model_untransf.coef_[0])}).sort_values(by='Values',ascending=False).head(30).set_index('Feature Names').plot(kind='bar', figsize=(16, 7), rot=90,grid=True);


# In[ ]:




