# -*- coding: utf-8 -*-

"""The methods to load ML models and predict the heat capacity using a set of ML features."""

import numpy as np
import pandas as pd
from .descriptors import simple_descriptors
import joblib
import glob
import copy



def predict_G_ensemble_dataset(models: list, FEATURES: list, df_features: pd.DataFrame, temperature: str) -> list:
    """Predict G using an ensemble of ML models for a dataset.

    :param models: ensemble of ML models
    :param FEATURES: features for ML model
    :param df_features: pandas dataframe containing the features
    :param temperature: target temperature 

    """
    df_site_structure=copy.deepcopy(df_features)
    for model_idx,model in enumerate(models):
        df_site_structure["G_{}_predicted_{}".format(temperature,model_idx)]=model.predict(df_site_structure[FEATURES])
    results=[]
    for name in df_site_structure["MP_ID"].unique():
        predicted_mol=[]
        for model_idx in range(len(models)):
            sites=df_site_structure.loc[df_site_structure["MP_ID"]==name]
            predicted_mol.append(sites["G_{}_predicted_{}".format(temperature,model_idx)])
        results.append({
            "name":name,
            "predicted_G_molar_{}_mean".format(temperature): np.mean(predicted_mol),
            "predicted_G_molar_{}_std".format(temperature): np.std(predicted_mol),
        })
    return results


