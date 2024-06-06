import streamlit as st


def page_project_hypotheses_body():

    st.write("### Project Hypotheses")

    st.write(
        f"* [Hypothesis 1](#hypothesis-1)\n"
        f"* [Hypothesis 2](#hypothesis-2)\n"
        f"* [Hypothesis 3](#hypothesis-3)"
    )

    st.info(
        f"#### **Hypothesis 1**\n\n"
        f"* We suspect that the highest risk factors involved in heart disease are cholesterol and maximum heart rate.\n\n"
        f"##### **Findings**:\n\n"
        f"* Enter findings from correlation analysis here."
    )

    st.warning(
        f"#### **Hypothesis 2**\n\n"
        f"* We suspect that a successful prediction will rely on a large number of parameters.\n\n"
        f"##### **Findings**:\n\n"
        f"* Enter findings from analysis of feature importance after hyperparameter optimisation."
    )

    st.success(
        f"#### **Hypothesis 3**\n\n"
        f"* We suspect that men over 50 with high cholesterol are the most at-risk patient group.\n\n"
        f"##### **Findings**:\n\n"
        f"* Enter findings here."
    )
