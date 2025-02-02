import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seab as sns
from phik import phik_matrix
from sklearn.preprocessing import LabelEncoder

# URL to dataset on GitHub
DATASET_URL = "https://raw.githubusercontent.com/demokritfromabyss/DS/refs/heads/main/streamlit_app/contract_new_streamlit.csv"

@st.cache_data
def load_data(url):
    """Load dataset from GitHub and convert column names to lowercase."""
    try:
        data = pd.read_csv(url)
        data.columns = data.columns.str.lower()  # Convert column names to lowercase
        return data.copy()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def display_dataset_info(data):
    """Display general dataset information for all columns."""
    st.subheader("Dataset Overview")
    st.write("**First 10 Rows:**")
    st.write(data.head(10))
    st.write("**Dataset Info:**")
    buffer = []
    data.info(buf=buffer.append)
    st.text("\n".join(buffer))
    st.write("**Descriptive Statistics:**")
    st.write(data.describe(include='all', percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))
    
    st.write("**Missing Values:**")
    st.write(data.isnull().sum())
    
    st.write("**Duplicate Rows:**")
    st.write(data.duplicated().sum())

def main():
    """Main function to run Streamlit app."""
    st.title("Automated Data Preprocessing and EDA App 🚀")
    st.write("""
        **Application Purpose**: Data preprocessing and exploratory data analysis (EDA).
        Load a dataset, perform preprocessing, and visualize the data to gain insights.
    """)

    if "app_started" not in st.session_state:
        st.session_state.app_started = False
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None

    if not st.session_state.app_started:
        if st.button("Start Application"):
            st.session_state.app_started = True
            st.session_state.processed_data = load_data(DATASET_URL)
            st.rerun()
        return

    data = st.session_state.processed_data.copy()

    if data is not None:
        # Slider for selecting number of rows to process
        row_count = st.slider("Select number of rows to process", min_value=10, max_value=len(data), value=10)
        data = data.iloc[:row_count]

        display_dataset_info(data)

        # Column deletion
        st.subheader("Column Deletion 🗑️")
        columns_to_delete = st.multiselect("Select columns to delete", options=data.columns.tolist())
        if st.button("Delete Selected Columns"):
            data.drop(columns=columns_to_delete, inplace=True)
            st.session_state.processed_data = data.copy()
            st.success("Selected columns deleted.")
            display_dataset_info(data)

        # Data type conversion
        st.subheader("Change Data Types 🔄")
        selected_columns = st.multiselect("Select columns to convert", options=data.columns.tolist())
        target_type = st.selectbox("Select target data type", ["int", "float", "object", "datetime", "category", "bool", "string"])

        if st.button("Convert Data Type"):
            for col in selected_columns:
                if target_type == "int":
                    data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
                elif target_type == "float":
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                elif target_type == "object":
                    data[col] = data[col].astype(str)
                elif target_type == "datetime":
                    data[col] = pd.to_datetime(data[col], errors='coerce').dt.date
                elif target_type == "category":
                    data[col] = data[col].astype('category')
                elif target_type == "bool":
                    data[col] = data[col].astype(bool)
                elif target_type == "string":
                    data[col] = data[col].astype(str)
            st.session_state.processed_data = data.copy()
            st.success("Data types successfully converted.")
            display_dataset_info(data)

        # Visualization
        st.subheader("Data Visualization 📊")

        if st.button("Generate Histograms for All Features"):
        st.subheader("Feature Histograms")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(8, 5 * len(numeric_columns)))
        if len(numeric_columns) == 1:
            axes = [axes]
        for ax, column in zip(axes, numeric_columns):
            data[column].hist(bins=20, ax=ax)
            ax.set_title(f"Histogram: {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
        st.pyplot(fig)
    st.subheader("Feature Histograms")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(8, 5 * len(numeric_columns)))
    if len(numeric_columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, numeric_columns):
        data[column].hist(bins=20, ax=ax)
        ax.set_title(f"Histogram: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
    st.pyplot(fig)
            st.subheader("Feature Histograms")
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                fig, ax = plt.subplots()
                data[column].hist(bins=20, ax=ax)
                ax.set_title(f"Histogram: {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

        # Correlation matrix
        if st.button("Show Correlation Matrix 🔗"):
            st.subheader("Correlation Matrix")
            try:
                corr_matrix = data.phik_matrix(interval_cols=data.select_dtypes(include=[np.number]).columns)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("PhiK Correlation Matrix")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating correlation matrix: {e}")

if __name__ == "__main__":
    main()
