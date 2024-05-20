import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from autogluon.tabular import TabularPredictor

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost as xgb

from utility import *

import random
np.random.seed(2)
random.seed(2)

"""
This is the main script that is used to run machine learning models on SUVR values to predict tau positivity status. 

(C) Robel K. Gebre, PhD
Aging and Dementia Imaging Research (ADIR)
Mayo Clinic, Rochester, MN
2024

"""



features = ["angular","calcarine","cingulum_ant","cingulum_mid","cingulum_post","cuneus","frontal_inf_oper","frontal_inf_orb", "frontal_inf_tri", "frontal_med_orb","frontal_mid","frontal_mid_orb",
            "frontal_sup","frontal_sup_medial", "frontal_sup_orb", "fusiform", "herschl", "insula","lingual","occipital_inf","occipital_mid","occipital_sup","olfactory","parahippocampal", "paracentral", "parietal_inf",
            "parietal_sup","postcentral", "precentral","precuneus","rectus","retrosplenial","rolandic_oper",  "supp_motor_area","supramarginal","temporal_pole_mid","temporal_pole_sup", "amygdala", "entorhinal",
            "temporal_inf","temporal_mid"]

##Note: The features given here are based on the MCALT-ADIR122 atlas. Check README for reference. Example csv is also provided with this Repo.

#read your data here
data = pd.read_csv('replace with your data.csv')

X = data[features].astype(np.float64)
y = data[['True_Label']].astype(int) #True label is a place holder, replace by your correct column name for the binary class

#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #This split is rudimentary, based on your needs, stratify split gives better results.

##### Train a new ag model --- (from here) ----------------
# train_data = pd.concat([X_train, y_train], axis=1)
# test_data = pd.concat([X_test, y_test], axis=1)

# predictor = TabularPredictor(
#     label="True_Label", path=os.path.join(os.getcwd(),'trained models/from scratch'), eval_metric="accuracy", problem_type="binary"
# ).fit(
#     train_data.reset_index(drop=True),
#     # auto_stack=True,
#     # holdout_frac=0.20,
#     # hyperparameters="default",  
#     # use_bag_holdout=True,  
#     # num_bag_sets=4,  
#     # num_bag_folds=10,  
#     # num_stack_levels=2,  
#     # verbosity=2,
#     # presets='best_quality'
# )

##### Or load a pretrained model
predictor = TabularPredictor.load(os.path.join(os.getcwd(), 'trained models/on Mayo data'), require_version_match=False) # Change the path to your directory containing the predictor

##### A voting classifier with a grid search #####
## Note: These models have to be trained each time and are not saved on disk like the autogluon predictor shown above. 
estimators = [
    ("rf", RandomForestClassifier()),
    ("xgb", xgb.XGBClassifier()),
    ("lgb", lgb.LGBMClassifier()),
]

voting_clf = VotingClassifier(estimators=estimators, voting="soft")

params = {
    "rf__n_estimators": [500,800,1200],
    "rf__criterion": ['gini', 'entropy', 'log_loss'],

    "lgb__extra_trees": [True], 
    "lgb__objective": ["binary"],

    "xgb__n_estimators": [100,500,800,1200], 
}

voting_grid = GridSearchCV(
    estimator=voting_clf, 
    param_grid = params,
    cv=5, 
    scoring="accuracy", 
    verbose=1, 
    n_jobs=-1,
)

voting_grid.fit(X_train, y_train.values.ravel())

y_pred = voting_grid.predict(X_test)
performance_stats_voting = classification_report(y_test.values.ravel(), y_pred, output_dict=True)
print('voting',performance_stats_voting)

##### get the individual models
voting_clf_trained = voting_grid.best_estimator_
rf_model = voting_clf_trained.named_estimators_['rf']
lgb_model = voting_clf_trained.named_estimators_['lgb']
xgb_model = voting_clf_trained.named_estimators_['xgb']

trained_models = {
    "RF": rf_model,
    "XGB": xgb_model,
    "LGB": lgb_model,
    "Ensemble_voting": voting_grid,
    'AG': predictor,
    }

df_theta_dict = {}
for model_name, trained_model in trained_models.items():
    print(f"Calculating SHAP values for {model_name}")
    if model_name == 'AG':
        shap_values, explainer = calculate_shap_values(trained_model, features, X_train, X, 'ag')
    else:
        shap_values, explainer = calculate_shap_values(trained_model, features, X_train, X, 'general')
    df_theta_dict[model_name] = model_THETA(data, shap_values, features, 'THETA_' + model_name)

####### Stat and visualization
df_0 = pd.concat([df_theta_dict['AG'],
                  df_theta_dict['RF'],
                  df_theta_dict['XGB'],
                  df_theta_dict['LGB'],
                  df_theta_dict['Ensemble_voting'],
                  
                ], 
                axis=1)

df_0.to_csv('THETAs.csv')

##### ----------- Training done (to here) ---------------

#### If you've run it already and have df_0 on disk, then comment the section within (from here) to (to here) and use below
# df_0 = pd.read_csv(os.path.join(os.getcwd(),'THETAs.csv'))[['THETA_AG','THETA_RF','THETA_XGB','THETA_LGB','THETA_Ensemble_voting']]

corr = df_0.corr('spearman')
print(corr)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.set(font_scale=1.3)
plt.figure(figsize=(15, 15))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('corr.png')

first_column = df_0.columns[0]
sns.set(font_scale=1.2)
sns.set_style('whitegrid')
plt.figure(figsize=(12, 10))  
for i, column in enumerate(df_0.columns[1:], 1):
    plt.subplot(2, 2, i) 
    sns.regplot(x=df_0[first_column], y=df_0[column], data=df_0)
    plt.title(f'{first_column} vs {column}')
plt.tight_layout()
plt.savefig('regression_plots.png')