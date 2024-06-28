import numpy as np
from sklearn.pipeline import Pipeline

from feature_engine.imputation import MeanMedianImputer, RandomSampleImputer
from feature_engine.discretisation import ArbitraryDiscretiser
from feature_engine.encoding import OneHotEncoder


def dc_no_encoding_pipeline(df):
    """
    Fit and transform data to a data cleaning pipeline
    without encoding and map discretised values
    """

    pipeline = Pipeline([
        ("median_imputation", MeanMedianImputer(
            imputation_method="median",
            variables=["RestingBP"])),
        ("random_sample_imputation", RandomSampleImputer(
            random_state=1,
            seed='general',
            variables=["Cholesterol"])),
        ("arbitrary_discretisation", ArbitraryDiscretiser(
            binning_dict={"Oldpeak": [-np.inf, 0, 1.5, np.inf]})),
        ])

    clean_df = pipeline.fit_transform(df)

    map_dict = {0: "≤ 0", 1: "0 ≤ 1.5", 2: "> 1.5"}
    clean_df["Oldpeak"] = clean_df["Oldpeak"].replace(to_replace=map_dict)

    return clean_df


def dc_all_cat_pipeline(df):
    """
    Fit and transform data to a data cleaning pipeline that
    discretises all categorical data
    """

    pipeline = Pipeline([
        ("median_imputation", MeanMedianImputer(
            imputation_method="median",
            variables=["RestingBP"])),
        ("random_sample_imputation", RandomSampleImputer(
            random_state=1,
            seed='general',
            variables=["Cholesterol"])),
        ("arbitrary_discretisation", ArbitraryDiscretiser(binning_dict={
            "Age": [-np.inf, 40, 50, 60, np.inf],
            "RestingBP": [-np.inf, 120, 130, 140, np.inf],
            "Cholesterol": [-np.inf, 173, 223, 267, np.inf],
            "MaxHR": [-np.inf, 120, 138, 156, np.inf],
            "Oldpeak": [-np.inf, 0, 1.5, np.inf],
        })),
    ])

    clean_df = pipeline.fit_transform(df)

    return clean_df


def map_discretisation(df):
    """
    Map discretised data to values
    """

    map_dict = {
        "Age": {0: "≤ 40", 1: "40 ≤ 50", 2: "50 ≤ 60", 3: "> 60"},
        "RestingBP": {0: "≤ 120", 1: "120 ≤ 130", 2: "130 ≤ 140", 3: "> 140"},
        "Cholesterol": {
            0: "≤ 173", 1: "173 ≤ 223", 2: "223 ≤ 267", 3: "> 267"
            },
        "MaxHR": {0: "≤ 120", 1: "120 ≤ 138", 2: "138 ≤ 156", 3: "> 156"},
        "Oldpeak": {0: "≤ 0", 1: "0 ≤ 1.5", 2: "> 1.5"},
        }

    for k in map_dict:
        df[k].replace(to_replace=map_dict[k], inplace=True)

    return df


def one_hot_encode(df):
    """
    Transform data with a one hot encoder
    """

    encoder = OneHotEncoder(
        variables=df.columns[df.dtypes == "object"].to_list(),
        drop_last=False)
    df_ohe = encoder.fit_transform(df)

    return df_ohe
