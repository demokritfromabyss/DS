import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 📌 URL to the dataset on GitHub (replace with your own link!)
DATASET_URL = "https://github.com/demokritfromabyss/DS/blob/main/streamlit_app/contract_new_streamlit.csv"

# Function to load data from GitHub
@st.cache_data
def load_data(url):
    """Load dataset from GitHub and handle errors."""
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to display dataset overview
def data_set_info(data):
    """Display dataset overview: head, shape, data types, and missing values."""
    st.subheader("Dataset Overview 📊")
    st.write(data.head())  # Display first 5 rows
    st.write(f"📏 Shape: {data.shape}")
    st.write(f"📊 Data types: {data.dtypes}")
    st.write("❓ Missing values:", data.isnull().sum())

# Function to check unique values in categorical columns
def check_unique_values(data):
    """Check and display unique values for categorical columns."""
    st.subheader("Unique values in categorical features 🔍")
    for column in data.select_dtypes(include=['object']).columns:
        unique_values = data[column].unique()
        st.write(f"🔹 {column}: {len(unique_values)} unique values")

# Function to handle missing data
def handle_missing_data(data):
    """Handle missing data by dropping rows or filling with mean/median."""
    st.subheader("Handling Missing Values ❓")
    method = st.selectbox("Choose method", ["Drop rows", "Fill with median", "Fill with mean"])
    
    if method == "Drop rows":
        data = data.dropna()
        st.write("❌ Missing rows removed.")
    elif method == "Fill with median":
        data.fillna(data.median(), inplace=True)
        st.write("📉 Missing values filled with median.")
    elif method == "Fill with mean":
        data.fillna(data.mean(), inplace=True)
        st.write("📈 Missing values filled with mean.")
    
    return data

# Function to remove duplicate rows
def remove_duplicates(data):
    """Remove duplicate rows from the dataset."""
    st.subheader("Removing Duplicates 🚀")
    initial_size = data.shape[0]
    data = data.drop_duplicates()
    st.write(f"✅ Removed {initial_size - data.shape[0]} duplicate rows.")
    return data

# Function to encode categorical data
def encode_categorical_data(data):
    """Convert categorical data into numerical format (One-Hot or Label Encoding)."""
    st.subheader("Encoding Categorical Features 🔢")
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    if categorical_columns.any():
        encoding_method = st.selectbox("Choose encoding method", ["One-Hot Encoding", "Label Encoding"])
        
        for column in categorical_columns:
            if encoding_method == "One-Hot Encoding":
                data = pd.get_dummies(data, columns=[column], prefix=[column])
                st.write(f"🎭 {column} encoded using One-Hot Encoding.")
            elif encoding_method == "Label Encoding":
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                st.write(f"🔢 {column} encoded using Label Encoding.")
    else:
        st.write("✅ No categorical features to encode.")
    
    return data

# Function to remove outliers using IQR method
def remove_outliers(data):
    """Remove outliers using the Interquartile Range (IQR) method."""
    st.subheader("Removing Outliers ✂")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        Q1, Q3 = data[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        initial_size = data.shape[0]
        data = data[(data[column] >= lower) & (data[column] <= upper)]
        st.write(f"📉 {column}: Outliers removed. Rows reduced from {initial_size} to {data.shape[0]}")
    
    return data

# Function to visualize data
def visualize_data(data):
    """Generate histograms and correlation matrix for numerical features."""
    st.subheader("Data Visualization 📊")

    # Histograms for numerical columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        fig, ax = plt.subplots()
        data[column].hist(bins=20, ax=ax)
        ax.set_title(f"Histogram: {column}")
        st.pyplot(fig)

    # Correlation matrix heatmap
    st.subheader("Correlation Matrix 🔗")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Main function for Streamlit app
def main():
    """Main function to run the Streamlit app."""
    st.title("Automated Data Preprocessing App 🚀")
    
    # Load dataset from GitHub
    data = load_data(DATASET_URL)
    
    if data is not None:
        data_set_info(data)           # Display dataset overview
        data = handle_missing_data(data)  # Handle missing values
        data = remove_duplicates(data)    # Remove duplicate rows
        data = encode_categorical_data(data)  # Encode categorical features
        data = remove_outliers(data)    # Remove outliers
        check_unique_values(data)       # Check unique values
        visualize_data(data)            # Generate data visualizations

# Run the Streamlit app
if __name__ == "__main__":
    main()
