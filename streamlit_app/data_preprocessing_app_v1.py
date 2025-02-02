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
        **Назначение программы**: Предобработка данных и разведочный анализ (EDA).
        Загрузите набор данных, выполните предобработку и визуализируйте данные для получения инсайтов.
    """)

    if st.button("Запустить программу"):
        # Load dataset
        data = load_data(DATASET_URL)

        if data is not None:
            # Slider for selecting number of rows to process
            row_count = st.slider("Выберите количество строк для обработки", min_value=10, max_value=len(data), value=10)
            data = data.iloc[:row_count]  # Process only selected number of rows

            st.write("**Предварительный просмотр данных:**")
            st.write(data.head())

            # Column deletion
            st.subheader("Удаление столбцов 🗑️")
            columns_to_delete = st.multiselect("Выберите столбцы для удаления", data.columns)
            if st.button("Удалить выбранные столбцы"):
                data.drop(columns=columns_to_delete, inplace=True)
                st.success("Выбранные столбцы удалены.")
                st.write(data.head())

            # Data type conversion
            st.subheader("Изменение типов данных 🔄")
            selected_columns = st.multiselect("Выберите столбцы для преобразования", data.columns)
            target_type = st.selectbox("Выберите целевой тип данных", ["int", "float", "object", "datetime"])

            if st.button("Преобразовать тип данных"):
                for col in selected_columns:
                    if target_type == "int":
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
                    elif target_type == "float":
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    elif target_type == "object":
                        data[col] = data[col].astype(str)
                    elif target_type == "datetime":
                        data[col] = pd.to_datetime(data[col], errors='coerce')
                st.success("Типы данных успешно преобразованы.")
                st.write(data.head())

            # Visualization
            st.subheader("Визуализация данных 📊")

            if st.button("Построить гистограммы для всех признаков"):
                st.subheader("Гистограммы признаков")
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for column in numeric_columns:
                    fig, ax = plt.subplots()
                    data[column].hist(bins=20, ax=ax)
                    ax.set_title(f"Гистограмма: {column}")
                    ax.set_xlabel(column)
                    ax.set_ylabel("Частота")
                    st.pyplot(fig)

            st.subheader("Построение пользовательских графиков")
            selected_features = st.multiselect("Выберите признаки для визуализации", data.columns)
            plot_type = st.selectbox("Выберите тип графика", ["Гистограмма", "Boxplot", "Scatter"])
            color = st.color_picker("Выберите цвет графика")
            grid = st.checkbox("Показать сетку")

            if plot_type == "Scatter":
                alpha = st.slider("Выберите прозрачность точек", 0.0, 1.0, 0.5)

            if st.button("Построить выбранный график"):
                if plot_type == "Гистограмма":
                    for feature in selected_features:
                        fig, ax = plt.subplots()
                        data[feature].hist(bins=20, ax=ax, color=color)
                        ax.set_title(f"Гистограмма: {feature}")
                        ax.set_xlabel(feature)
                        ax.set_ylabel("Частота")
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
                        st.error("Для scatter-графика необходимо выбрать ровно 2 признака.")

            # Correlation matrix
            if st.button("Показать матрицу корреляции 🔗"):
                st.subheader("Матрица корреляции")
                corr_matrix = data.phik_matrix(interval_cols=data.select_dtypes(include=[np.number]).columns)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("PhiK Матрица корреляции")
                st.pyplot(fig)

if __name__ == "__main__":
    main()
