import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


DATASET_DF = pd.read_csv(f"./inputs/datasets/raw/heart.csv")

def page_feat_correlation_body():

    st.write("### Feature Correlation Study")

    st.info(
        f"#### **Business Requirement 1**: Data Visualisation and Correlation Study\n\n"
        f"* We need to perform a correlation study to determine which features correlate most closely to the target.\n"
        f"* A Pearson's correlation will indicate linear relationships between numerical variables.\n"
        f"* A Spearman's correlation will measure the monotonic relationships between variables.\n"
        f"* A Predictive Power Score study can also be used to determine relationships between attributes regardless of data type (6/11 features are categorical).\n"
    )

    if st.checkbox("Inspect heart disease dataset"):
        st.dataframe(DATASET_DF.head(5))
        st.write(f"The dataset contains {DATASET_DF.shape[0]} observations with {DATASET_DF.shape[1]} attributes.")

    st.write("---")

    st.write(
        f"#### **Summary of Correlation Analysis**\n"
        f"* Correlations within the dataset were analysed using Spearman and Pearson correlations followed by a Predictive Power Score (PPS) analysis.\n"
        f"* For the Spearman and Pearson correlations, all categorical features from the cleaned dataset were one hot encoded.\n"
        f"* Both methods found the same correlation between features and target.\n"
    )

    pearson_corr_results = plt.imread("outputs/images/pearson_correlation.png")
    spearman_corr_results = plt.imread("outputs/images/spearman_correlation.png")

    if st.checkbox("View Pearson correlation results"):
        st.image(pearson_corr_results)
    if st.checkbox("View Spearman correlation results"):
        st.image(spearman_corr_results)

    st.write("---")

    st.write(
        f"#### **Summary of PPS Analysis**\n"
        f"* The PPS analyis also indicated the same features had the greatest correlation to the target.\n"
        f"* These features were: ChestPainType, ExerciseAngina and ST_Slope.\n"
        f"* The PPS analysis also indicated that the Oldpeak feature had some weak correlation to the target.\n"
    )

    pps_heatmap = plt.imread("outputs/images/pps_heatmap.png")

    if st.checkbox("View PPS heatmap"):
        st.image(pps_heatmap)

    st.write("---")

    chestpaintype_dist = plt.imread("outputs/images/ChestPainType_distribution.png")
    exerciseangina_dist = plt.imread("outputs/images/ExerciseAngina_distribution.png")
    oldpeak_dist = plt.imread("outputs/images/Oldpeak_distribution.png")
    st_slope_dist = plt.imread("outputs/images/ST_Slope_distribution.png")

    st.write(
        f"#### **Analysis of Most Correlated Features**\n"
        f"* An asymptomatic (ASY) chest pain type is typically associated with heart disease.\n"
        f"* Exercise-induced angina is typically associated with heart disease, although a significant portion of patients without exercise-induced angina also had heart disease.\n"
        f"* An ST depression of >1.5 mm is typically associated with heart disease, and roughly 50% of patients with an ST depression between 0 and 1.5 had heart disease.\n"
        f"* A flat ST slope is typically associated with heart disease. A down ST slope is generally more related to heart disease as well, however there is less data on this.\n"
    )

    feature_distribution = st.selectbox(
        "Select feature to view distribution:",
        ("ChestPainType", "ExerciseAngina", "Oldpeak", "ST_Slope")
    )

    if feature_distribution == "ChestPainType":
        st.image(chestpaintype_dist)
    elif feature_distribution == "ExerciseAngina":
        st.image(exerciseangina_dist)
    elif feature_distribution == "Oldpeak":
        st.image(oldpeak_dist)
    elif feature_distribution == "ST_Slope":
        st.image(st_slope_dist)

    st.write("---")

    st.success(
        f"#### **Conclusions**\n\n"
        f"* We stated in **hypothesis 1** that we expected cholesterol and maximum heart rate to be the greatest risk factors related to heart disease.\n"
        f"* The findings from the exploratory data analysis found this hypothesis to be **incorrect.**\n"
        f"* From the correlation studies, it was found that the most highly correlating features were:\n"
        f"  * A patient's chest pain type.\n"
        f"  * Chest pain (angina) induced from exercise.\n"
        f"  * An ST depression of >1.5 mm on an ECG.\n"
        f"  * A flat or downward sloping ST slope on an ECG.\n"
    )
