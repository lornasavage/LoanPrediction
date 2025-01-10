import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
#import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from shapash.explainer.smart_explainer import SmartExplainer
import io

sage_green = "#B2AC88" 

st.markdown(
    f"""
    <style>
    body {{
        background-color: {sage_green};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("Loan Prediction")

try:
    df = pd.read_csv("credit.csv")
except FileNotFoundError:
    st.error("The dataset file 'credit.csv' was not found. Please ensure it's in the correct path.")

app_page = st.sidebar.selectbox('Select Page', ['Business Case Presentation and Data Description', 'Data Visualization', 'Deployment','Prediction Models', 'Feature Importance and Driving Variables', 'Hyperparameter Tuning Experiences and Best Performing Model', 'Conclusion'])



st.sidebar.header("Automated Data Preprocessing")
drop_unknowns = st.sidebar.checkbox("Drop rows with unknown values", value=True)
    
# Columns to encode
columns_to_encode = [
    "checking_balance", "credit_history", "savings_balance", 
    "employment_duration", "housing", "job", "phone", "default", "other_credit"
]

# Drop the 'purpose' column
if "purpose" in df.columns:
    df.drop("purpose", axis=1, inplace=True)

# Encode the specified columns
label_encoders = {}
for col in columns_to_encode:
    if col in df.columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

# Handle unknown values
if drop_unknowns:
    df.replace("unknown", np.nan, inplace=True)
    df.dropna(inplace=True)


if app_page == 'Business Case Presentation and Data Description':
    st.header('Business Case Pres and data overview')
    st.write('Dataset')
    st.dataframe(df)

    st.write('Header of dataset: ')
    st.dataframe(df.head())

    st.write("Information about the dataframe: ")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("Statistics on the dataset: ")
    st.dataframe(df.describe())



if app_page == 'Data Visualization':
    #'Graphs and stuff'
    data = {'credit_history': ['critical', 'poor', 'good', 'very good', 'perfect']}
    df2 = pd.DataFrame(data)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['ch-encoded'] = label_encoder.fit_transform(df['credit_history']) 
    st.dataframe(df.head())

    list_columns = df.columns
    values = st.multiselect("Select two variables: ", list_columns, ["amount", "percent_of_income"])
    st.bar_chart(df, x = values[0], y=values[1])
    sns.swarmplot(data=df, x = values [0], y = values[1], palette='Set2')


    # Dropdowns for selecting variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    selected_x = st.selectbox("Select a categorical variable for the x-axis:", categorical_columns)
    selected_y = st.selectbox("Select a numeric variable for the y-axis:", numeric_columns)

    # Check if user has selected both variables
    if selected_x and selected_y:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.swarmplot(data=df, x=selected_x, y=selected_y, palette='Set2', ax=ax)
            st.pyplot(fig)
        except ValueError as e:
            st.error(f"Error creating swarmplot: {e}")


if app_page == 'Prediction Models':
    MODELS = {
        #"Logistic Regression":LogisticRegression,
        "KNN":KNeighborsClassifier,
        "Decision Tree":DecisionTreeClassifier,
    }

    model_mode=st.sidebar.selectbox("Select a model of your choice",['KNN','Decision Tree'])

    #df.State=pd.Categorical(df['State']).codes
    #df['International plan']=pd.Categorical(df['International plan']).codes
    #df['Voice mail plan']=pd.Categorical(df['Voice mail plan']).codes
    #df['Churn']=pd.Categorical(df['Churn']).codes


    X = df.drop(["default"], axis=1)
    y = df.default

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


    #if model_mode == 'Logistic Regression':
    #log = LogisticRegression()
    #model = log.fit(X_train,y_train)
    #prediction = log.predict(X_test)
    #st.write("Accuracy Logistic Regression:",metrics.accuracy_score(y_test,prediction))

    if model_mode == 'KNN':
        knn=KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train,y_train)
        prediction = knn.predict(X_test)
        st.write("Accuracy KNN:",metrics.accuracy_score(y_test,prediction))


    elif model_mode == 'Decision Tree':
        tree=DecisionTreeClassifier()
        tree.fit(X_train,y_train)
        prediction = tree.predict(X_test)
        st.write("Accuracy Tree:",metrics.accuracy_score(y_test,prediction))




if app_page == 'Feature Importance and Driving Variables':
    from shapash.explainer.smart_explainer import SmartExplainer

    explainer = SmartExplainer(model=tree)
    explainer.compile(x=X_test, y_pred=y_pred)
    explainer.plot.features_importance() 

    st.header('feature importance')
    xpl = SmartExplainer(tree)
    y_pred = pd.Series(prediction)
    X_test = X_test.reset_index(drop=True)
    xpl.compile(x=X_test, y_pred=y_pred)
    xpl.plot.features_importance()


if app_page == 'Hyperparameter Tuning Experiences and Best Performing Model':
    st.header('Parameter Tuning')
    import mlflow
    from mlflow import log_metric, log_param, log_artifact
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics
    import joblib


    import dagshub
    dagshub.init(repo_owner='lorriesavage', repo_name='finalProj', mlflow=True)

    # Start MLflow run
    with mlflow.start_run():

        mlflow.log_param("parameter name","value")
        mlflow.log_metric("Accuracy",0.9)

        mlflow.end_run()


        st.title("ML Flow Visualization")

        ui.link_button(text="👉 Go to ML Flow",url="__________[put your dagshub link here pls]",key="link_btnmlflow")


        X = df.drop(["default"], axis=1)
        y = df.default

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create a decision tree classifier
        dt = DecisionTreeClassifier(random_state=42)

        # Define a parameter grid to search over
        param_grid = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2, 4]}

        # Create GridSearchCV object
        grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)

        # Perform grid search to find the best parameters
        grid_search.fit(X_train, y_train)

        # Log the best parameters
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)

        # Evaluate the model
        best_dt = grid_search.best_estimator_
        test_score = best_dt.score(X_test, y_test)

        # Log the performance metric
        y_pred = pd.Series(prediction)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)

        log_metric("accuracy", accuracy)
        log_metric("precision", precision)
        log_metric("recall", recall)
        log_metric("f1", f1)


        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        # Log the best model in MLflow
        mlflow.sklearn.log_model(best_dt, "best_dt")

        # Save the model to the MLflow artifact store
        mlflow.sklearn.save_model(best_dt, "best_dt_model")



    
if app_page == 'Deployment 🚀':
    st.markdown("# :violet[Deployment 🚀]")
    
    # Model Selection
    select_ds = "Loan Default Prediction 🏦"
    #id = st.text_input('ID Model', '00ffae4993044a5d9cb369a46dbc1e01')  
    
    # Load the model from MLflow
    logged_model = f'./mlruns/0/{id}/artifacts/top_model_v1'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Load dataset (for feature names and structure)
    df = pd.read_csv("credit.csv") 
    
    # Select columns for deployment 
    deploy_df = df.drop(columns=['default'], axis=1)  # Drop target variable
    feature_columns = deploy_df.columns  # Get feature names

    # Create interactive inputs for deployment
    input_values = {}
    for col in feature_columns:
        default_value = float(df[col].mean())  
        input_values[col] = st.number_input(f"Enter value for {col}", value=default_value)

    # Create DataFrame from input values
    data_new = pd.DataFrame({col: [value] for col, value in input_values.items()})
    
    # Display inputs for verification
    st.write("Input Data Preview:")
    st.dataframe(data_new)

    # Prediction
    try:
        prediction = loaded_model.predict(data_new)[0]  # Predict default status
        st.write(f"Prediction: {'Default' if prediction >= 0.5 else 'No Default'} (Score: {np.round(prediction, 2)})")
    except Exception as e:
        st.error(f"Error during prediction: {e}")



if app_page == 'Conclusion':
    st.header('Conclusion:')
    
