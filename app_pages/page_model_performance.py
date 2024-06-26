import streamlit as st
import pandas as pd

from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_model_performance_body():

    version = "v2"
    dc_fe_pipeline = load_pkl_file(
        f"outputs/ml_pipeline/classification_model/{version}/data_cleaning_and_feat_engineering_pipeline.pkl"
    )
    model_pipeline = load_pkl_file(
        f"outputs/ml_pipeline/classification_model/{version}/classification_pipeline.pkl"
    )
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/classification_model/{version}/X_train.csv"
    )
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/classification_model/{version}/X_test.csv"
    )
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/classification_model/{version}/y_train.csv"
    )
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/classification_model/{version}/y_test.csv"
    )

    st.write("### ML Pipeline: Binary Classification")

    st.info(
        f"The model success metrics are:\n"
        f"* At least 90% recall for heart disease (the model minimises the chances of missing a posititve diagnosis).\n\n"
        f"The model will be considered a failure if:\n"
        f"* The model fails to achieve 90% recall for heart disease.\n"
        f"* The model fails to achieve 70% precision for no heart disease (false positives).\n"
    )

    st.write("---")
    st.write(f"#### ML Pipelines")
    st.write(f"For this model there were 2 ML Pipelines arrange in series:\n")

    st.write(f"* The first pipeline is responsible for data cleaning and feature engineering.\n")
    st.write(dc_fe_pipeline)

    st.write(f"* The second pipeline is responsible for feature scaling and modelling.\n")
    st.write(model_pipeline)

    st.write("---")
    st.write(f"#### Model Performance")
    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=model_pipeline,
                    label_map=["No Heart Disease", "Heart Disease"])
