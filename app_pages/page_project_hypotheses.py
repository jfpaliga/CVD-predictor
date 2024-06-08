import streamlit as st


def page_project_hypotheses_body():

    st.write("### Project Hypotheses")

    st.write(
        f"* [Hypothesis 1](#hypothesis-1)\n"
        f"* [Hypothesis 2](#hypothesis-2)\n"
        f"* [Hypothesis 3](#hypothesis-3)\n"
    )

    st.write(f"#### **Hypothesis 1**\n\n")
    st.info(
        f"* We suspect that the highest risk factors involved in heart disease are cholesterol and maximum heart rate.\n\n"
    )
    st.write(f"##### **Findings**:\n\n")
    st.info(
        f"* This hypothesis was incorrect.\n"
        f"* Chest pain type, exercise-induced angina, ST slope and ST depression were the most highly correlated features with heart disease.\n"
        f"* Maximum heart rate and cholesterol were found to have only very weak correlations using Pearson and Spearman correlations.\n"
        f"* Predictive power score found no correlation between heart disease and maximum heart rate or cholesterol.\n"
    )

    st.write(f"#### **Hypothesis 2**\n\n")
    st.warning(
        f"* We suspect that a successful prediction will rely on a large number of parameters.\n"
    )
    st.write(f"##### **Findings**:\n\n")
    st.warning(
        f"* Enter findings from analysis of feature importance after hyperparameter optimisation.\n"
    )

    st.write(f"#### **Hypothesis 3**\n\n")
    st.success(
        f"* We suspect that men over 50 with high cholesterol are the most at-risk patient group.\n\n"
    )
    st.write(f"##### **Findings**:\n\n")
    st.success(
        f"* Enter findings here."
    )
