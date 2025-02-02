import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from phik import phik_matrix
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

def main():
    """Main function to run Streamlit app."""
    st.title("Automated Data Preprocessing and EDA App 🚀")
    st.write("""
        **Application Purpose**: Data preprocessing and exploratory data analysis (EDA).
        Load a dataset, perform preprocessing, and visualize the data to gain insights.
    """)

    if "app_started" not in st.session_state:
        st.session_state.app_started = False

    if not st.session_state.app_started:
        if st.button("Start Application"):
            st.session_state.app_started = True
            st.rerun()
        return

    # Load dataset
    data = load_data(DATASET_URL)

    if data is not None:
        # Slider for selecting number of rows to process
        row_count = st.slider("Select number of rows to process", min_value=10, max_value=len(data), value=10)
        data = data.iloc[:row_count]  # Process only selected number of rows

        st.write("**Preview of the dataset:**")
        st.write(data.head())

        # Column deletion
        st.subheader("Column Deletion 🗑️")
        columns_to_delete = st.multiselect("Select columns to delete", data.columns)
        if st.button("Delete Selected Columns"):
            data.drop(columns=columns_to_delete, inplace=True)
            st.success("Selected columns deleted.")
            st.write(data.head())

        # Data type conversion
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
            st.success("Data types successfully converted.")
            st.write(data.head())

        # Visualization
        st.subheader("Data Visualization 📊")

        if st.button("Generate Histograms for All Features"):
            st.subheader("Feature Histograms")
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                fig, ax = plt.subplots()
                data[column].hist(bins=20, ax=ax)
                ax.set_title(f"Histogram: {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

        st.subheader("Custom Graphs")
        selected_features = st.multiselect("Select features for visualization", data.columns)
        plot_type = st.selectbox("Select chart type", ["Histogram", "Boxplot", "Scatter"])
        color = st.color_picker("Choose chart color")
        grid = st.checkbox("Show grid")

        if plot_type == "Scatter":
            alpha = st.slider("Select point transparency", 0.0, 1.0, 0.5)

        if st.button("Generate Selected Chart"):
            if plot_type == "Histogram":
                for feature in selected_features:
                    fig, ax = plt.subplots()
                    data[feature].hist(bins=20, ax=ax, color=color)
                    ax.set_title(f"Histogram: {feature}")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Frequency")
                    if grid:
                        ax.grid(True)
                    st.pyplot(fig)

            elif plot_type == "Boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(data=data[selected_features], ax=ax, palette=[color])
                ax.set_title("Boxplot")
                if grid:
                    ax.grid(True)
                st.pyplot(fig)

            elif plot_type == "Scatter":
                if len(selected_features) == 2:
                    fig, ax = plt.subplots()
                    data.plot.scatter(x=selected_features[0], y=selected_features[1], ax=ax, color=color, alpha=alpha)
                    ax.set_title(f"Scatter plot: {selected_features[0]} vs {selected_features[1]}")
                    ax.set_xlabel(selected_features[0])
                    ax.set_ylabel(selected_features[1])
                    if grid:
                        ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.error("For scatter plots, please select exactly 2 features.")

        # Correlation matrix
        if st.button("Show Correlation Matrix 🔗"):
            st.subheader("Correlation Matrix")
            corr_matrix = data.phik_matrix(interval_cols=data.select_dtypes(include=[np.number]).columns)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("PhiK Correlation Matrix")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
