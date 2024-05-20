import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import shap
import random

np.random.seed(2)
random.seed(2)

def get_iqrs(sep_csv_suvrs, sep_csv_shaps, tau_suvrs, n_quant_upper, n_quant_lower):
    """
    This function calculates interquartile ranges (IQRs) and isolates regions in SUVR data based on corresponding SHAP values.

    Main steps:
    1. Calculates upper and lower quantiles of SHAP values based on provided quantile percentages.
    2. Identifies SUVR values that fall above the upper quantile, below the lower quantile, and within the IQR.
    3. Creates separate datasets for each region (top, lower, between).
    4. Calculates the number of regions in the "between" category for each sample.

    Returns: 
        * suvr_top, suvr_lower, suvr_between: SUVr dataframes for the top, lower, and between regions.
        * shap_top, shap_lower, shap_between: Corresponding SHAP value dataframes.
        * _suvrs, _shap_quantiles_top_values, _shap_quantiles_bottom_values, _shaps: Original SUVr data, SHAP values, and calculated quantiles.
        * remaning_regions_csv: Dataframe containing the count of "between" regions for each sample.
    """
    n_quant_upper *= 100
    n_quant_lower *= 100

    _suvrs = sep_csv_suvrs[tau_suvrs]
    _shaps = sep_csv_shaps[tau_suvrs] 
    
    _shap_quantiles_top = np.percentile(_shaps, n_quant_upper, axis=1)
    _shap_quantiles_bottom = np.percentile(_shaps, n_quant_lower, axis=1)

    _shap_quantiles_top_values = _shap_quantiles_top 
    _shap_quantiles_bottom_values = _shap_quantiles_bottom 

    _shap_quantiles_bottom_values = np.where(np.isnan(_shap_quantiles_bottom_values), 0, _shap_quantiles_bottom_values)
    _shap_quantiles_top_values = np.where(np.isnan(_shap_quantiles_top_values), 0, _shap_quantiles_top_values)

    shap_top = _shaps[_shaps > _shap_quantiles_bottom_values[:, None]].fillna(0)
    shap_lower = _shaps[_shaps < _shap_quantiles_top_values[:, None]].fillna(0)
    shap_between = _shaps[(_shaps > _shap_quantiles_bottom_values[:,None]) & (_shaps < _shap_quantiles_top_values[:,None])].fillna(0)

    mask_top = (shap_top == 0)
    suvr_top = _suvrs.copy()
    suvr_top[mask_top] = 0

    mask_bottom = (shap_lower == 0)
    suvr_lower = _suvrs.copy()
    suvr_lower[mask_bottom] = 0

    mask_between = (shap_between == 0)
    suvr_between = _suvrs.copy()
    suvr_between[mask_between] = 0

    non_zero_between = shap_between.apply(lambda x: (x != 0).sum(), axis=1)
    remaning_regions_csv = pd.DataFrame(non_zero_between, columns=['number of middle regions'])
    
    return suvr_top, suvr_lower, suvr_between, shap_top, shap_lower, shap_between, _suvrs, _shap_quantiles_top_values, _shap_quantiles_bottom_values, _shaps, remaning_regions_csv

def calculate_regional_THETAindices(_shaps, suvr_between, shap_between):
    """
    This function calculates regional THETA indices

    Main steps:
    1. Calculates an intermediate index by multiplying region-specific SUVr values (`suvr_between`) 
    with corresponding SHAP values (`shap_between`). This indicates the combined influence of a feature within  the interquartile range.
    2. Adds the original SHAP values (`_shaps`) to the intermediate index, resulting in the final regional THETA index.

    Returns:
        * regional_THETAindices: The final calculated regional THETA indices.
        * regional_THETAindices_iqr: The intermediate index used in the interquartile range.
    """

    regional_THETAindices_iqr = suvr_between.reset_index(drop=True) * shap_between.reset_index(drop=True)

    regional_THETAindices = regional_THETAindices_iqr.reset_index(drop=True) + _shaps.reset_index(drop=True)

    return regional_THETAindices, regional_THETAindices_iqr

def calculate_THETA_scores(regional_THETAindices_iqr,_shaps):
    """
    This function calculates THETA scores.

    Main steps:
    1. Calculates the sum across all regions for the original SHAP values (`_shaps`).
    2. Calculates the sum across all regions for the regional THETA indices (`regional_THETAindices_iqr`)
    3. Takes the absolute values of both sums and adds them together to arrive at the final THETA score.

    Returns:
        * THETA: An array or Series of THETA scores.
    """
    sum_A = np.sum(_shaps.reset_index(drop=True), axis=1)

    sum_B = np.sum(regional_THETAindices_iqr.reset_index(drop=True), axis=1)

    THETA = np.abs(sum_A.values) + np.abs(sum_B.values) 

    return THETA

def model_THETA(df_suvr, shap_values_df, tau_suvrs, col_name):
    """
    A helper function to fascilitate THETA calculateion. This functions gets called from the outside "main.py" 
    Returns:
        * A single dataframe containing THETA
    """
    suvr_top, suvr_lower, suvr_between, shap_top, shap_lower, shap_between, _suvrs, _shap_quantiles_top_values, _shap_quantiles_bottom_values, _shaps, remaning_regions_csv = get_iqrs(df_suvr, shap_values_df, tau_suvrs, 0.99, 1-0.99)

    regional_THETAindices, regional_THETAindices_iqr = calculate_regional_THETAindices(shap_values_df, suvr_between, shap_between)

    THETA = calculate_THETA_scores(regional_THETAindices_iqr, shap_values_df)

    return pd.DataFrame(THETA,columns=[col_name])

class GeneralWrapper:
    """A general wrapper for sklearn depedent models."""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    
    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        predictions = self.model.predict_proba(X)[:, 1]  
        return predictions

class AutogluonWrapper:
    """Reference: https://github.com/autogluon/autogluon/blob/master/examples/tabular/interpret/SHAP%20with%20AutoGluon-Tabular%20Census%20income%20classification.ipynb """
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names
    
    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X, as_multiclass=False)

def calculate_shap_values(model, features, background_data, X, model_type): 
    """
    This function calculates the SHAP values that are used as weights to the SUVRs.
    
    "SHAP is intended to explain how much each feature contributes to a particular prediction. 
    In this binary classification context, "how much" is quantified in terms of the deviation 
    between predicted probability of the positive class from a baseline reference value." 
    Documentation: https://github.com/autogluon/autogluon/blob/master/examples/tabular/interpret/SHAP%20with%20AutoGluon-Tabular%20Census%20income%20classification.ipynb 

    """

    if model_type=='general':
        """
        Some models can return a perfect predicted probability which will make the odds -inf or inf so to prevent that we can increase the background 
        data by a small amount by taking quantiles around the median. This will make the odds finite.
        """

        quantile_index = np.linspace(0.35, 0.55, 3) 
        # quantile_index = np.linspace(0.25, 0.75, 3) #Use if you want more spread but keeping it close to the median is important
        quantiles = background_data[features].quantile(quantile_index)
        med = quantiles

        wrapper = GeneralWrapper(model, features)
        explainer = shap.KernelExplainer(wrapper.predict_binary_prob, med)
        shap_values = explainer.shap_values(X)
        shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
    
    if model_type=='ag':
        """
        Taking the median is fine here since unlike other models, AG models do not not suffer from a perfect predicted probability.
        """

        med = pd.DataFrame([background_data.median()], columns=background_data.columns) 

        ag_wrapper = AutogluonWrapper(model, features)
        explainer = shap.KernelExplainer(ag_wrapper.predict_binary_prob, med)
        shap_values = explainer.shap_values(X)
        shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
   
    return shap_values_df,explainer