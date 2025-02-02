import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load data
def load_data():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Display general information about the dataset
def data_set_info(data):
    st.subheader("Dataset Overview:")
    st.write(data.head())
    st.write("\\n")
    st.write(f"Data dimensions: {data.shape}")
    st.write(f"Data types: {data.dtypes}")
    st.write(f"Missing values: {data.isnull().sum()}")

# Check unique values for categorical features
def check_unique_values(data, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
        
    st.subheader("Unique values in columns:")
    for column in data.columns:
        if column in exclude_columns:
            continue  # Skip unnecessary columns
        if data[column].dtype == 'object':  # Check only categorical data
            unique_values = data[column].unique()
            unique_count = len(unique_values)
            st.write(f"{column}: {unique_values} (Unique: {unique_count})")

# Handle missing data
def handle_missing_data(data):
    st.subheader("Handling missing values:")
    method = st.selectbox("Choose a method to handle missing values", ["Drop rows", "Fill with median", "Fill with mean"])
    
    if method == "Drop rows":
        data = data.dropna()
        st.write("Missing rows have been dropped.")
    elif method == "Fill with median":
        data = data.fillna(data.median())
        st.write("Missing values have been filled with the median.")
    elif method == "Fill with mean":
        data = data.fillna(data.mean())
        st.write("Missing values have been filled with the mean.")
    
    return data

# Remove duplicates
def remove_duplicates(data):
    st.subheader("Removing duplicates:")
    if st.button("Remove duplicates"):
        initial_shape = data.shape[0]
        data = data.drop_duplicates()
        st.write(f"Removed {initial_shape - data.shape[0]} duplicates.")
    return data

# Encode categorical features
def encode_categorical_data(data):
    st.subheader("Encoding categorical features:")
    categorical_columns = data.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        encoding_method = st.selectbox("Choose encoding method", ["One-Hot Encoding", "Label Encoding"])
        for column in categorical_columns:
            if encoding_method == "One-Hot Encoding":
                data = pd.get_dummies(data, columns=[column], prefix=[column])
                st.write(f"{column} has been transformed using One-Hot Encoding.")
            elif encoding_method == "Label Encoding":
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                st.write(f"{column} has been transformed using Label Encoding.")
    else:
        st.write("No categorical features for encoding.")
    return data

# Remove outliers (using IQR method)
def remove_outliers(data):
    st.subheader("Removing outliers:")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_size = data.shape[0]
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        st.write(f"For {column}, outliers have been removed. Data reduced from {initial_size} to {data.shape[0]}")
    return data

# Visualize data
def visualize_data(data):
    st.subheader("Charts for numerical features:")

    # Histograms for numerical features
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        st.write(f"Histogram for {column}:")
        fig, ax = plt.subplots()
        data[column].hist(bins=20, ax=ax)
        ax.set_title(f'Histogram: {column}')
        st.pyplot(fig)
    
    # Scatter plot for two features
    st.write("Scatter plot for two features:")
    col1, col2 = st.selectbox("Choose two numerical features for scatter plot", numeric_columns, index=(0, 1))
    fig, ax = plt.subplots()
    ax.scatter(data[col1], data[col2])
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"Scatter plot: {col1} vs {col2}")
    st.pyplot(fig)

    # Correlation matrix
    st.write("Correlation matrix:")
    corr_matrix = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

# Main function for the Streamlit app
def main():
    st.title("Data Preprocessing and Analysis")
    
    # Load data
    data = load_data()
    
    if data is not None:
        # Display general information
        data_set_info(data)

        # Handle missing data
        data = handle_missing_data(data)

        # Remove duplicates
        data = remove_duplicates(data)

        # Encode categorical features
        data = encode_categorical_data(data)

        # Remove outliers
        data = remove_outliers(data)

        # Check unique values
        check_unique_values(data)

        # Visualize data
        visualize_data(data)

if __name__ == "__main__":
    main()
