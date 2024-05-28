import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Streamlit app title
st.title('Customer Churn Prediction')

# File uploader
train_file = st.file_uploader("Upload the training data CSV", type="csv")
test_file = st.file_uploader("Upload the test data CSV", type="csv")

if train_file is not None and test_file is not None:
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Show the first few rows of the data
    st.write("Training Data Preview:")
    st.write(train_data.head())
    st.write("Missing Values in Training Data:")
    st.write(train_data.isnull().sum())

    # Remove duplicates
    train_data.drop_duplicates(inplace=True, keep="first")

    # Data info
    st.write("Training Data Information:")
    st.write(train_data.info())

    # Label encoding
    labelencode = LabelEncoder()
    def encode(data):
        for col in data.columns:
            if data[col].dtypes == 'object':
                data[col] = labelencode.fit_transform(data[col])
        return data

    train_data = encode(train_data)
    test_data = encode(test_data)

    # Identify categorical variables
    categories_train = [col for col in train_data.columns if train_data[col].dtype == 'object' or train_data[col].nunique() < 10]
    categories_test = [col for col in test_data.columns if test_data[col].dtype == 'object' or test_data[col].nunique() < 10]

    st.write("Categorical variables in Training Data:", categories_train)
    st.write("Categorical variables in Test Data:", categories_test)

    # One-Hot Encoding
    train_data = pd.get_dummies(train_data, columns=['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'])
    test_data = pd.get_dummies(test_data, columns=['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'])

    # Correlation heatmap
    mask = np.triu(np.ones_like(train_data.corr()))
    plt.figure(figsize=(20, 12))
    sns.heatmap(train_data.corr(), cmap="RdYlBu", annot=True, mask=mask, vmin=-1, vmax=1)
    st.pyplot(plt)

    # Exited distribution
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    sns.countplot(data=train_data, x="Exited", palette="RdYlBu")
    plt.title("Distribution of 'Exited'")
    st.pyplot(fig)

    # Splitting data
    X = train_data.drop("Exited", axis=1)
    y = train_data.Exited

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Decision Tree Model
    model1 = DecisionTreeClassifier()
    model1.fit(X_train, y_train)
    
    # Random Forest Model
    model2 = RandomForestClassifier()
    model2.fit(X_train, y_train)

    # Evaluate models
    roc_auc_dt = roc_auc_score(y_val, model1.predict(X_val))
    roc_auc_rf = roc_auc_score(y_val, model2.predict(X_val))

    st.write("ROC AUC of Decision Tree:", roc_auc_dt)
    st.write("ROC AUC of Random Forest:", roc_auc_rf)

    # Make predictions on validation set using Random Forest model
    y_pred_model2 = model2.predict(X_val)
    st.write("Predictions on validation set by Random Forest:", y_pred_model2)

    # Predict on test data
    y_pred = model2.predict(test_data)

    # Prepare submission file
    sub_path = '/kaggle/input/playground-series-s4e1/sample_submission.csv'
    submission = pd.read_csv(sub_path)
    submission['Exited'] = y_pred
    submission.to_csv('submission.csv', index=False)

    st.write("Submission file created successfully!")

    # Download link for submission file
    with open("submission.csv", "rb") as file:
        st.download_button("Download Submission File", file, file_name="submission.csv")

else:
    st.write("Please upload both training and test data files.")