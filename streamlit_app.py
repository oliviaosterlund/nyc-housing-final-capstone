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
    page_title="New York Housing Market",
    layout="centered",
    page_icon="🍎",
)

dfo = pd.read_csv("NY-House-Dataset.csv")
df = dfo.drop(["BROKERTITLE", "LONG_NAME"], axis = 1)
st.sidebar.title("New York Housing Market")
page = st.sidebar.selectbox("Select Page",["Introduction","Data Visualization", "Automated Report","Predictions", "Explainability", "MLFlow Runs"])
df_numeric = df.select_dtypes(include=np.number)
if page == "Introduction":
    st.subheader("New York Housing Market")
    st.markdown("""
    #### What this app does:
    - *Analyzes* key factors
    - *Visualizes* trends
    - *Predicts* housing prices using a selection of regression models
    """)
    st.image("housingimage.png")
    
    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("#####  Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())


elif page == "Data Visualization":
    st.subheader("Data Viz")

    tab1, tab2 = st.tabs(["Scatter Plot", "Correlation Heatmap"])
    with tab1:
        st.subheader("Scatter Plot")
        fig_bar, ax_bar = plt.subplots(figsize=(12,6))
        x_col = st.selectbox("Select x-axis variable", df_numeric.columns.drop("PRICE"))
        sns.scatterplot(df_numeric, x = x_col, y = "PRICE")
        st.pyplot(fig_bar)
    with tab2:
        st.subheader("Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(18,14))
        sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
        st.pyplot(fig_corr)

elif page == "Automated Report":
    st.subheader("Automated Report")
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(df,title="New York Housing Market",explorative=True,minimal=True)
            st_profile_report(profile)
        export = profile.to_html()
        st.download_button(label="📥 Download full Report",data=export,file_name="New_York_Housing_Market.html",mime='text/html')

elif page == "Predictions":
    st.subheader("Predictions")

    list_var = list(df_numeric.columns.drop("PRICE"))
    features_selection = st.sidebar.multiselect("Select Features (X)",list_var,default=list_var)
    if not features_selection:
        st.warning("Please select at least one feature")
        st.stop()
    
    model_name = st.sidebar.selectbox(
        "Choose Model",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
    )

    params = {}
    if model_name == "Decision Tree":
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "Random Forest":
        params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "XGBoost":
        params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)

    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"],default=["Mean Absolute Error (MAE)"])

    
    X = df_numeric[features_selection]
    y = df_numeric["PRICE"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

    
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(**params, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(**params, random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(objective='reg:squarederror', **params, random_state=42)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = metrics.mean_squared_error(y_test, predictions)
        mae = metrics.mean_absolute_error(y_test, predictions)
        r2 = metrics.r2_score(y_test, predictions)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE** {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R2** {r2:,.3f}")
    
    fig, ax = plt.subplots()
    ax.scatter(y_test,predictions,alpha=0.5)
    ax.plot([y_test.min(),y_test.max()],
           [y_test.min(),y_test.max() ],"--r",linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

elif page == "Explainability":
    st.subheader("Explainability")

    X_shap, y_shap = df_numeric.drop(columns = "PRICE"), df_numeric["PRICE"]
    # Train default XGBoost model for explainability
    model_exp = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_exp.fit(X_shap, y_shap)

    # Create SHAP explainer and values
    explainer = shap.Explainer(model_exp)
    shap_values = explainer(X_shap)

    # SHAP Waterfall Plot for first prediction
    st.markdown("### SHAP Waterfall Plot for First Prediction")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())


    # SHAP Scatter Plot for 'PROPERTYSQFT'
    st.markdown("### SHAP Scatter Plot for 'PROPERTYSQFT'")
    shap.plots.scatter(shap_values[:, "PROPERTYSQFT"], color=shap_values, show=False)
    st.pyplot(plt.gcf())

elif page == "MLflow Runs":
    st.subheader("MLflow Runs")
    # Fetch runs
    runs = mlflow.search_runs(order_by=["start_time desc"])
    st.dataframe(runs)
    st.markdown(
        "View detailed runs on DagsHub: [oliviaosterlund/finalprojectapp MLflow](https://dagshub.com/oliviaosterlund/finalprojectapp.mlflow)"
    )
