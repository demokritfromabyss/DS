import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# URL to dataset on GitHub (replace with your actual link)
DATASET_URL = "https://raw.githubusercontent.com/demokritfromabyss/DS/refs/heads/main/streamlit_app/contract_new_streamlit.csv"

@st.cache_data
def load_data(url):
    """Load dataset from GitHub and convert column names to lowercase."""
    try:
        data = pd.read_csv(url)
        data.columns = data.columns.str.lower()  # Convert column names to lowercase
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to convert data types
def convert_data_types(data):
    """Allow user to select columns and change their data types."""
    st.subheader("Change Data Types 🔄")
    selected_columns = st.multiselect("Select columns to convert", data.columns)
    target_type = st.selectbox("Select target data type", ["int", "float", "object", "datetime"])
    
    if st.button("Convert Data Type"):  
        for col in selected_columns:
            if target_type == "int":
                data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
            elif target_type == "float":
                data[col] = pd.to_numeric(data[col], errors='coerce')
            elif target_type == "object":
                data[col] = data[col].astype(str)
            elif target_type == "datetime":
                data[col] = pd.to_datetime(data[col], errors='coerce')
        st.success("Data types converted successfully!")
    return data

# Function to visualize data
def visualize_data(data):
    """Generate histograms and correlation matrix for numerical features."""
    if st.button("Show Histograms 📊"):
        st.subheader("Feature Histograms")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            fig, ax = plt.subplots()
            data[column].hist(bins=20, ax=ax)
            ax.set_title(f"Histogram: {column}")
            st.pyplot(fig)
    
    if st.button("Show Correlation Matrix 🔗"):
        st.subheader("Correlation Matrix")
        corr_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Main function
def main():
    """Main function to run Streamlit app."""
    st.title("Automated Data Preprocessing App 🚀")
    
    # Load dataset
    data = load_data(DATASET_URL)
    
    if data is not None:
        # Slider for selecting number of rows to process
        row_count = st.slider("Select number of rows to process", min_value=10, max_value=len(data), value=10)
        data = data.iloc[:row_count]  # Process only selected number of rows
        
        st.write(data.head())
        
        # Convert data types
        data = convert_data_types(data)
        
        # Show updated dataset
        st.subheader("Updated Dataset")
        st.write(data.head())
        
        # Visualizations
        visualize_data(data)

# Run the app
if __name__ == "__main__":
    main()
