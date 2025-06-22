import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import shap
import pycaret

token = st.secrets["DAGSHUB_TOKEN"]
dagshub.auth.add_app_token(token=token)

dagshub.init(repo_owner='oliviaosterlund', repo_name='finalprojectapp', mlflow=True)

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics

st.set_page_config(
    page_title="New York Housing",
    layout="centered",
    page_icon="🍎",
)

df = pd.read_csv("NY-House-Dataset.csv")
st.sidebar.title("New York Housing")
page = st.sidebar.selectbox("Select Page",["Introduction","Data Visualization", "Automated Report","Predictions", "Explainability", "Pycaret", "MLFlow Runs"])


