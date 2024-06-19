import streamlit as st
import pandas as pd
import joblib


@st.cache_data
def load_patient_data_raw():
    df = pd.read_csv("outputs/datasets/collection/HeartDiseasePrediction.csv")
    return df

@st.cache_data
def load_cleaned_test_set():
    df = pd.read_csv("outputs/datasets/cleaned/TestSetCleaned.csv")
    return df

@st.cache_data
def load_cleaned_train_set():
    df = pd.read_csv("outputs/datasets/cleaned/TrainSetCleaned.csv")
    return df

def load_pkl_file(file_path):
    return joblib.load(filename=file_path)
