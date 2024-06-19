import numpy as np
from sklearn.pipeline import Pipeline

from feature_engine.imputation import MeanMedianImputer, RandomSampleImputer
from feature_engine.discretisation import ArbitraryDiscretiser
from feature_engine.encoding import OneHotEncoder


def dc_no_encoding_pipeline(df):

    pipeline = Pipeline([
        ("median_imputation", MeanMedianImputer(imputation_method="median",
                                                variables=["RestingBP"])),
        ("random_sample_imputation", RandomSampleImputer(random_state=1,
                                                         seed='general',
                                                         variables=["Cholesterol"])),
        ("arbitrary_discretisation", ArbitraryDiscretiser(binning_dict={"Oldpeak":[-np.inf, 0, 1.5, np.inf]})),
        ])

    clean_df = pipeline.fit_transform(df)

    map_dict = {0: "≤ 0", 1: "0 ≤ 1.5", 2: "> 1.5"}
    clean_df["Oldpeak"] = clean_df["Oldpeak"].replace(to_replace=map_dict)

    return clean_df

def one_hot_encode(df):

    encoder = OneHotEncoder(variables=df.columns[df.dtypes=="object"].to_list(), drop_last=False)
    df_ohe = encoder.fit_transform(df)

    return df_ohe
    
