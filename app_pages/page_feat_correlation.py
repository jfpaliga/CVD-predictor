import streamlit as st
import pandas as pd


DATASET_DF = pd.read_csv(f"./inputs/datasets/raw/heart.csv").head(5)

def page_feat_correlation_body():

    st.write("### Feature Correlation Study")

    st.info(
        f"#### **Business Requirement 1**: Data Visualisation and Correlation Study\n\n"
        f"* We need to perform a correlation study to determine which features correlate most closely to the target.\n\n"
        f"* A Pearson's correlation will indicate linear relationships between numerical variables.\n\n"
        f"* A Spearman's correlation will measure the monotonic relationships between variables.\n\n"
        f"* A Predictive Power Score study can also be used to determine relationships between attributes regardless of data type (6/11 features are categorical).\n\n"
    )

    st.dataframe(DATASET_DF)
    st.write("The dataset contains 918 observations with 12 attributes.")

    st.write("---")

    st.success(
        f"#### **Summary of Correlation Analysis**\n\n"
        f"* Correlations within the dataset were analysed using Spearman and Pearson correlations followed by a Predictive Power Score (PPS) analysis.\n\n"
        f"* For the Spearman and Pearson correlations, all categorical features from the cleaned dataset were one hot encoded.\n\n"
        f"* Both methods found the same correlation between features and target.\n\n"
        f"* The PPS analyis also indicated the same features had the greatest correlation to the target.\n\n"
        f"* These features were: ChestPainType, ExerciseAngina and ST_Slope.\n\n"
        f"* The PPS analysis also indicated that the Oldpeak feature had some weak correlation to the target."
    )

