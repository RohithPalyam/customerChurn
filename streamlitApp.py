import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.echo_expander import echo_expander
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
customer_churn = pd.read_csv('customer_churn.csv')

# Sidebar with option menu
with st.sidebar:
    selected = option_menu(
        "Tasks Menu",
        ["Data Manipulation", "Data Visualization", "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest"],
        icons=["file", "bar-chart", "trend-up", "trend-down", "tree", "grid"],
        menu_icon="menu-app",
        default_index=0
    )

# Task 1: Data Manipulation
if selected == "Data Manipulation":
    with st.echo():
        # Extract 5th column
        customer_5 = customer_churn.iloc[:, 4]
        st.write("5th Column:", customer_5.head())

        # Extract 15th column
        customer_15 = customer_churn.iloc[:, 14]
        st.write("15th Column:", customer_15.head())

        # Extract male senior citizens with electronic check payment
        senior_male_electronic = customer_churn[(customer_churn['gender'] == 'Male') &
                                                (customer_churn['SeniorCitizen'] == 1) &
                                                (customer_churn['PaymentMethod'] == 'Electronic check')]
        st.write("Male Senior Citizens with Electronic Check Payment:", senior_male_electronic.head())

        # Extract customers with tenure > 70 or monthly charges > $100
        customer_total_tenure = customer_churn[(customer_churn['tenure'] > 70) |
                                               (customer_churn['MonthlyCharges'] > 100)]
        st.write("Customers with Tenure > 70 or Monthly Charges > $100:", customer_total_tenure.head())

        # Extract customers with 2-year contract, mailed check, and churned
        two_mail_yes = customer_churn[(customer_churn['Contract'] == 'Two year') &
                                       (customer_churn['PaymentMethod'] == 'Mailed check') &
                                       (customer_churn['Churn'] == 'Yes')]
        st.write("Churned Customers with 2-Year Contract and Mailed Check:", two_mail_yes.head())

        # Extract 333 random records
        customer_333 = customer_churn.sample(n=333, random_state=42)
        st.write("333 Random Records:", customer_333.head())

        # Count levels in 'Churn' column
        churn_count = customer_churn['Churn'].value_counts()
        st.write("Churn Count:", churn_count)

# Task 2: Data Visualization
if selected == "Data Visualization":
    with st.echo():
        # Bar plot for InternetService
        st.write("### Bar Plot: Distribution of Internet Service")
        plt.figure(figsize=(8, 6))
        customer_churn['InternetService'].value_counts().plot(kind='bar', color='orange')
        plt.xlabel('Categories of Internet Service')
        plt.ylabel('Count of Categories')
        plt.title('Distribution of Internet Service')
        st.pyplot(plt)

        # Histogram for tenure
        st.write("### Histogram: Distribution of Tenure")
        plt.figure(figsize=(8, 6))
        plt.hist(customer_churn['tenure'], bins=30, color='green')
        plt.title('Distribution of Tenure')
        st.pyplot(plt)

        # Scatter plot for MonthlyCharges vs Tenure
        st.write("### Scatter Plot: Tenure vs Monthly Charges")
        plt.figure(figsize=(8, 6))
        plt.scatter(customer_churn['tenure'], customer_churn['MonthlyCharges'], color='brown')
        plt.xlabel('Tenure of customer')
        plt.ylabel('Monthly Charges of customer')
        plt.title('Tenure vs Monthly Charges')
        st.pyplot(plt)

        # Box plot for Tenure vs Contract
        st.write("### Box Plot: Tenure vs Contract")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Contract', y='tenure', data=customer_churn)
        plt.title('Tenure vs Contract')
        st.pyplot(plt)

# Task 3: Linear Regression
if selected == "Linear Regression":
    with st.echo():
        X = customer_churn[['tenure']]
        y = customer_churn['MonthlyCharges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        y_pred = model_lr.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write("Root Mean Squared Error:", rmse)

# Task 4: Logistic Regression
if selected == "Logistic Regression":
    with st.echo():
        X = customer_churn[['MonthlyCharges']]
        y = (customer_churn['Churn'] == 'Yes').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
        model_log = LogisticRegression()
        model_log.fit(X_train, y_train)
        y_pred = model_log.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Confusion Matrix:", conf_matrix)
        st.write("Accuracy:", accuracy)

# Task 5: Decision Tree
if selected == "Decision Tree":
    with st.echo():
        X = customer_churn[['tenure']]
        y = (customer_churn['Churn'] == 'Yes').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_tree = DecisionTreeClassifier(random_state=42)
        model_tree.fit(X_train, y_train)
        y_pred_tree = model_tree.predict(X_test)
        conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
        accuracy_tree = accuracy_score(y_test, y_pred_tree)
        st.write("Confusion Matrix:", conf_matrix_tree)
        st.write("Accuracy:", accuracy_tree)

# Task 6: Random Forest
if selected == "Random Forest":
    with st.echo():
        X = customer_churn[['tenure', 'MonthlyCharges']]
        y = (customer_churn['Churn'] == 'Yes').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write("Confusion Matrix:", conf_matrix_rf)
        st.write("Accuracy:", accuracy_rf)
