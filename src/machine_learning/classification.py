import streamlit as st


def predict_live_heart_disease(X_live, dc_fe_pipeline, model_pipeline):
    """
    Filter live data to relevant features then process the data through
    the ML pipelines and provide a prediction on heart disease
    """

    features = ["ST_Slope", "ChestPainType", "MaxHR", "Age", "Cholesterol"]
    X_live = X_live.filter(features)

    X_live_dc_fe = dc_fe_pipeline.transform(X_live)

    cvd_prediction = model_pipeline.predict(X_live_dc_fe)
    cvd_prediction_proba = model_pipeline.predict_proba(X_live_dc_fe)

    cvd_proba = cvd_prediction_proba[0, cvd_prediction][0]*100
    if cvd_prediction == 1:
        cvd_result = "will"
    else:
        cvd_result = "will not"

    statement = (
        f"### There is a {cvd_proba.round(1)}% probability "
        f"that this patient **{cvd_result}** develop heart disease."
    )

    st.write(statement)

    return cvd_prediction
