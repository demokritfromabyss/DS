import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from phik import phik_matrix
from sklearn.preprocessing import LabelEncoder
import io
from fpdf import FPDF

# URL to dataset on GitHub
DATASET_URL = "https://raw.githubusercontent.com/demokritfromabyss/DS/refs/heads/main/streamlit_app/contract_new_streamlit.csv"

@st.cache_data
def load_data(url):
    """Load dataset from GitHub and convert column names to lowercase."""
    try:
        data = pd.read_csv(url)
        data.columns = data.columns.str.lower()
        return data.copy()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def display_dataset_info(data):
    """Display general dataset information for all columns."""
    st.subheader("Dataset Overview")
    st.write("**First 10 Rows:**")
    st.write(data.head(10))
    buffer = io.StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write("**Descriptive Statistics:**")
    st.write(data.describe(include='all', percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))
    st.write("**Missing Values:**")
    st.write(data.isnull().sum())
    st.write("**Duplicate Rows:**")
    st.write(data.duplicated().sum())

def reset_app():
    """Reset all changes and reload original dataset."""
    st.session_state.processed_data = load_data(DATASET_URL)
    st.experimental_rerun()

.read(), file_name="dataset_report.pdf", mime="application/pdf")

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
    
    st.sidebar.button("🔄 Reset", on_click=reset_app)
    
    data = st.session_state.processed_data.copy()
    if data is not None:
        row_count = st.slider("Select number of rows to process", min_value=10, max_value=len(data), value=10)
        data = data.iloc[:row_count]
        display_dataset_info(data)
        

        st.subheader("Data Visualization 📊")
        if st.button("Generate Histograms for All Features"):
    st.session_state.show_histograms = True

if "show_histograms" in st.session_state and st.session_state.show_histograms:
    st.subheader("Feature Histograms")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        fig, axes = plt.subplots(nrows=len(numeric_columns), figsize=(8, 5 * len(numeric_columns)))
        if len(numeric_columns) == 1:
            axes = [axes]
        elif len(numeric_columns) > 1:
            axes = axes.flatten()
        for ax, column in zip(axes, numeric_columns):
            data[column].hist(bins=20, ax=ax)
            ax.set_title(f"Histogram: {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for histograms.")

        if st.button("Show Correlation Matrix 🔗"):
    st.session_state.show_correlation_matrix = True

if "show_correlation_matrix" in st.session_state and st.session_state.show_correlation_matrix:
    st.subheader("Correlation Matrix")
    try:
        corr_matrix = data.phik_matrix()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("PhiK Correlation Matrix")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating correlation matrix: {e}")

if __name__ == "__main__":
    main()
