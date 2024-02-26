'''
This file contains the code for building a Random Forest Binary Classification Model
for predicting whether a tweet is related to sarcasm or not

We will be building models for two diffrent datasets(transformed and untransformed) 
and also tuning the hyperprameters using the validation set
Final evaluation is done on the test set
'''

##########################################################################################
###################################### Imports ###########################################

import shap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix


############################################################################################
###################################### Load Data ###########################################

### load transformed datasets
train_df_transf = pd.read_csv('data/predictive_modeling_data/X_train_transformed.csv')
val_df_transf = pd.read_csv('data/predictive_modeling_data/X_val_transformed.csv')
test_df_transf = pd.read_csv('data/predictive_modeling_data/X_test_transformed.csv')

Y_train_transf = pd.read_csv('data/predictive_modeling_data/Y_train_transformed.csv')
Y_val_transf = pd.read_csv('data/predictive_modeling_data/Y_val_transformed.csv')
Y_test_transf = pd.read_csv('data/predictive_modeling_data/Y_test_transformed.csv')


### load untransformed datasets
train_df_untransf = pd.read_csv('data/predictive_modeling_data/X_train_untransformed.csv')
val_df_untransf = pd.read_csv('data/predictive_modeling_data/X_val_untransformed.csv')
test_df_untransf = pd.read_csv('data/predictive_modeling_data/X_test_untransformed.csv')

Y_train_untransf = pd.read_csv('data/predictive_modeling_data/Y_train_untransformed.csv')
Y_val_untransf = pd.read_csv('data/predictive_modeling_data/Y_val_untransformed.csv')
Y_test_untransf = pd.read_csv('data/predictive_modeling_data/Y_test_untransformed.csv')


###################################################################################################
################## Helper Functions for the Predictive Modeling Process ###########################

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



########################################################################################################################
################## Helper Functions for the Hyperparameter Tuning and Getting the Best Model ###########################

def choose_best_rf_model(X_train_df, X_val_df, Y_train_df, Y_val_df, params_dict_search):
    '''
    Function for hyperparameter tuning of Random Forest Model and getting the best model.
    Uses Randomized Search CV. Best model is chosen based on the performance on the validation set.

    Returns:
        model estimator object
    '''

    rf_model = RandomForestClassifier(
        n_jobs=-1,
        random_state=1234
    )

    X_train_val_combined = pd.concat([X_train_df, X_val_df]).reset_index(drop=True)
    Y_train_val_combined = pd.concat([Y_train_df['label'], Y_val_df['label']]).reset_index(drop=True)
    train_val_indices = [-1]*X_train_df.shape[0] + ([1] * X_val_df.shape[0])

    ps = PredefinedSplit(train_val_indices)


    rf_cv = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=params_dict_search,
        n_jobs=-1,
        random_state=1234,
        cv=ps,
        verbose=3,
        scoring=['roc_auc', 'f1'],
        refit='roc_auc'
    )
    print("Parameter Options:",)
    print(rf_cv)
    print("---"*30, end="\n\n")

    hyper_param_results = rf_cv.fit(
        X_train_val_combined,
        Y_train_val_combined
    )
    
    rf_model_tuning_results_v1 = pd.DataFrame(hyper_param_results.cv_results_)
    rf_model_tuning_results_v1.to_csv('data/predictive_modeling_data/intermediate_data/rf_model_v1_param_tuning_results.csv', index=False)

    print(
        "Best Hyperparameters for RF Model 1 (PCA Transformed Dataset)\n\n",
        hyper_param_results.best_estimator_
    )

    print("---"*30, end="\n\n")

    print(
        "Best AUC Score for RF Model 1 (PCA Transformed Dataset)\n\n",
        hyper_param_results.best_score_
    )
    print("---"*30, end="\n\n")
    
    return hyper_param_results.best_estimator_



########################################################################################################################
######################### Model V1 - Random Forest Model with PCA Transformed Variables ################################

X_train_df = transformation_before_modeling_helper(train_df_transf)
X_val_df = transformation_before_modeling_helper(val_df_transf)
X_test_df = transformation_before_modeling_helper(test_df_transf)

print("Shape of Training, Validation, and Test Datasets Respectively: ")
print(X_train_df.shape, X_val_df.shape, X_test_df.shape)
print("---"*30, end="\n\n")

params_list_rf = {
    'max_depth': np.linspace(3, 40, dtype=int, num=7),
    'min_weight_fraction_leaf': np.linspace(0.000001, 0.001, num=7),
    'min_samples_leaf': np.linspace(10, 100, dtype=int, num=5),
    'n_estimators': [30, 100, 150]
}

best_rf_model_v1_transformed_data = choose_best_rf_model(
    X_train_df = X_train_df, 
    X_val_df = X_val_df, 
    Y_train_df = Y_train_transf,
    Y_val_df = Y_val_transf, 
    params_dict_search = params_list_rf
)

print("Selecting Best Model and Making Predictions on Test Set...")

final_rf_model_transf = best_rf_model_v1_transformed_data
test_transf_preds_probab = final_rf_model_transf.predict_proba(X_test_df)[:,1]
test_transf_preds_labels = final_rf_model_transf.predict(X_test_df)

print("---"*30, end="\n\n")


########################################################################################################################
################################### Model V1 - Evaluation Metrics on Test Set ##########################################

print("Classification Evaluation Report - Test Set Metrics: (1 = Sarcasm)")
print(classification_report(Y_test_transf['label'], test_transf_preds_labels))
print("---"*30, end="\n\n")


plt.figure(figsize=(6,6))
plot_roc_curve(
    Y_test_transf['label'],
    test_transf_preds_probab, 
    'Test Set (Model with PCA Transformed Data)', 
    1
)
print("---"*30, end="\n\n")


plt.figure(figsize=(6,6))
plot_conf_matrix(
    Y_test_transf['label'].map({0:'Not Sarcasm', 1:'Sarcasm'}),
    np.where(test_transf_preds_labels==1, 'Sarcasm', 'Not Sarcasm'), 
    ['Not Sarcasm', 'Sarcasm'],
    'Blues',
    'Test Set (Model with PCA Transformed Data) - Confusion Matrix'
)


#############################################################################################################################################
################################### Model V1 - Model Explanation using Feature Importance and SHAP ##########################################

### feature importance
features = pd.DataFrame()
features['feature'] = X_train_df.columns.values
features['importance'] = final_rf_model_transf.feature_importances_
temp = list(features.feature.str.replace('__|_', ' ', regex = True).str.split(' '))
features['feature'] = [' '.join(map(lambda x: x.capitalize(), i)) for i in temp]
features = features.sort_values(by=['importance'], ascending=False).head(30)
features.rename(columns = {'feature':'Feature Name'}, inplace=True)
features.set_index('Feature Name', inplace=True)
features.plot(kind='bar', figsize=(16, 10), rot=90)
plt.xticks(fontsize=12)
plt.yticks([], [])
plt.ylabel('Relative Importance')
plt.xlabel("Feature Names", rotation=0)
plt.title('RF Model with PCA Transformed Data - Top 30 Features')
plt.show()

### Random Forest Model Explanation using SHAP (Shapely Additive Explanations)
shap_explainer = shap.TreeExplainer(final_rf_model_transf)

analyze_indices = [0, 682, 4, 14767] # picking two positive and two negative cases and checking feature contributions

shap_values_list = []
for i in analyze_indices:
    print("\n\n\n")
    print("-------" * 25)
    print("Actual Label: ", Y_test_transf['label'][i])
    choosen_instance = X_test_df.loc[i]
    shap_values = shap_explainer.shap_values(choosen_instance)
    shap_values_list.append(shap_values)
    shap.force_plot(shap_explainer.expected_value[1], shap_values[1], choosen_instance, matplotlib=True, text_rotation=45, figsize=(16, 5))



########################################################################################################################
########################################################################################################################


########################################################################################################################
######################### Model V2 - Random Forest Model with Untransformed Variables ##################################

del X_train_df, X_val_df, X_test_df, final_rf_model_transf

X_train_df = transformation_before_modeling_helper(train_df_untransf)
X_val_df = transformation_before_modeling_helper(val_df_untransf)
X_test_df = transformation_before_modeling_helper(test_df_untransf)

print("Shape of Training, Validation, and Test Datasets Respectively: ")
print(X_train_df.shape, X_val_df.shape, X_test_df.shape)
print("---"*30, end="\n\n")

params_list_rf = {
    'max_depth': np.linspace(10, 100, dtype=int, num=5),
    'min_weight_fraction_leaf': np.linspace(0.00001, 0.01, num=7),
    'min_samples_leaf': np.linspace(10, 100, dtype=int, num=5),
    'n_estimators': [30, 100, 150]
}

best_rf_model_v2_untransformed_data = choose_best_rf_model(
    X_train_df = X_train_df, 
    X_val_df = X_val_df, 
    Y_train_df = Y_train_transf,
    Y_val_df = Y_val_transf,
    params_dict_search = params_list_rf
)

print("Selecting Best Model and Making Predictions on Test Set...")

final_rf_model_untransf = best_rf_model_v2_untransformed_data
test_untransf_preds_probab = final_rf_model_untransf.predict_proba(X_test_df)[:,1]
test_untransf_preds_labels = final_rf_model_untransf.predict(X_test_df)

print("---"*30, end="\n\n")


########################################################################################################################
################################### Model V2 - Evaluation Metrics on Test Set ##########################################

print("Classification Evaluation Report - Test Set Metrics: (1 = Sarcasm)")
print(classification_report(Y_test_untransf['label'], test_untransf_preds_labels))
print("---"*30, end="\n\n")


plt.figure(figsize=(6,6))
plot_roc_curve(
    Y_test_untransf['label'],
    test_untransf_preds_probab, 
    'Test Set (RF Model 2 with Untransformed Data)', 
    1
)
print("---"*30, end="\n\n")


plt.figure(figsize=(6,6))
plot_conf_matrix(
    Y_test_untransf['label'].map({0:'Not Sarcasm', 1:'Sarcasm'}),
    np.where(test_untransf_preds_labels==1, 'Sarcasm', 'Not Sarcasm'), 
    ['Not Sarcasm', 'Sarcasm'],
    'Blues',
    'Test Set (RF Model 2 with Untransformed Data) - Confusion Matrix'
)


#############################################################################################################################################
################################### Model V2 - Model Explanation using Feature Importance and SHAP ##########################################

### Feature importance values
features = pd.DataFrame()
features['feature'] = X_train_df.columns.values
features['importance'] = final_rf_model_untransf.feature_importances_
temp = list(features.feature.str.replace('__|_', ' ', regex = True).str.split(' '))
features['feature'] = [' '.join(map(lambda x: x.capitalize(), i)) for i in temp]
features = features.sort_values(by=['importance'], ascending=False).head(30)
features.rename(columns = {'feature':'Feature Name'}, inplace=True)
features.set_index('Feature Name', inplace=True)
features.plot(kind='bar', figsize=(16, 10), rot=90)
plt.xticks(fontsize=12)
plt.yticks([], [])
plt.ylabel('Relative Importance')
plt.xlabel("Feature Names", rotation=0)
plt.title('RF Model 2 with Untransformed Data - Top 30 Features')
plt.show()


### Random Forest Model Explanation using SHAP (Shapely Additive Explanations)
shap_explainer = shap.TreeExplainer(final_rf_model_untransf)
analyze_indices = [5, 682, 569, 14767] # picking two positive and two negative cases and checking feature contributions

shap_values_list = []
for i in analyze_indices:
    print("\n\n\n")
    print("-------" * 25)
    print("Actual Label: ", Y_test_transf['label'][i])
    choosen_instance = X_test_df.loc[i]
    shap_values = shap_explainer.shap_values(choosen_instance)
    shap_values_list.append(shap_values)
    shap.force_plot(shap_explainer.expected_value[1], shap_values[1], choosen_instance, matplotlib=True, text_rotation=45, figsize=(16, 5))
