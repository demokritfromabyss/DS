
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
def load_data():
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Вывод общей информации по датасетам
def data_set_info(data):
    st.subheader("Общая информация о датасете:")
    st.write(data.head())
    st.write("\n")
    st.write(f"Размер данных: {data.shape}")
    st.write(f"Типы данных: {data.dtypes}")
    st.write(f"Информация о пропусках: {data.isnull().sum()}")

# Подсчет уникальных значений для строковых признаков
def check_unique_values(data, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
        
    st.subheader("Уникальные значения в столбцах:")
    for column in data.columns:
        if column in exclude_columns:
            continue  # Пропускаем ненужные столбцы
        if data[column].dtype == 'object':  # Проверяем только строковые данные
            unique_values = data[column].unique()
            unique_count = len(unique_values)
            st.write(f"{column}: {unique_values} (Уникальных: {unique_count})")

# Обработка пропусков
def handle_missing_data(data):
    st.subheader("Обработка пропусков:")
    method = st.selectbox("Выберите метод обработки пропусков", ["Удалить строки", "Заполнить медианой", "Заполнить средним"])
    
    if method == "Удалить строки":
        data = data.dropna()
        st.write("Пропущенные строки были удалены.")
    elif method == "Заполнить медианой":
        data = data.fillna(data.median())
        st.write("Пропущенные значения были заполнены медианой.")
    elif method == "Заполнить средним":
        data = data.fillna(data.mean())
        st.write("Пропущенные значения были заполнены средним значением.")
    
    return data

# Удаление дубликатов
def remove_duplicates(data):
    st.subheader("Удаление дубликатов:")
    if st.button("Удалить дубликаты"):
        initial_shape = data.shape[0]
        data = data.drop_duplicates()
        st.write(f"Удалено {initial_shape - data.shape[0]} дубликатов.")
    return data

# Преобразование категориальных признаков в числовые
def encode_categorical_data(data):
    st.subheader("Преобразование категориальных признаков:")
    categorical_columns = data.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        encoding_method = st.selectbox("Выберите метод кодирования", ["One-Hot Encoding", "Label Encoding"])
        for column in categorical_columns:
            if encoding_method == "One-Hot Encoding":
                data = pd.get_dummies(data, columns=[column], prefix=[column])
                st.write(f"{column} был преобразован с помощью One-Hot Encoding.")
            elif encoding_method == "Label Encoding":
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                st.write(f"{column} был преобразован с помощью Label Encoding.")
    else:
        st.write("Нет категориальных признаков для кодирования.")
    return data

# Очистка выбросов (IQR метод)
def remove_outliers(data):
    st.subheader("Удаление выбросов:")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_size = data.shape[0]
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        st.write(f"Для {column} выбросы были удалены. Уменьшение размерности с {initial_size} до {data.shape[0]}")
    return data

# Визуализация данных
def visualize_data(data):
    st.subheader("Графики для числовых признаков:")

    # Гистограммы для числовых признаков
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        st.write(f"Гистограмма для {column}:")
        fig, ax = plt.subplots()
        data[column].hist(bins=20, ax=ax)
        ax.set_title(f'Гистограмма: {column}')
        st.pyplot(fig)
    
    # Диаграмма рассеяния (scatter plot) для пар признаков
    st.write("Диаграмма рассеяния для двух признаков:")
    col1, col2 = st.selectbox("Выберите два числовых признака для диаграммы рассеяния", numeric_columns, index=(0, 1))
    fig, ax = plt.subplots()
    ax.scatter(data[col1], data[col2])
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"Диаграмма рассеяния: {col1} vs {col2}")
    st.pyplot(fig)

    # Корреляционная матрица
    st.write("Корреляционная матрица:")
    corr_matrix = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Корреляционная матрица")
    st.pyplot(fig)

# Основная функция для приложения Streamlit
def main():
    st.title("Предобработка и анализ данных")
    
    # Загрузка данных
    data = load_data()
    
    if data is not None:
        # Вывод общей информации
        data_set_info(data)

        # Обработка пропусков
        data = handle_missing_data(data)

        # Удаление дубликатов
        data = remove_duplicates(data)

        # Преобразование категориальных признаков
        data = encode_categorical_data(data)

        # Удаление выбросов
        data = remove_outliers(data)

        # Проверка уникальных значений
        check_unique_values(data)

        # Визуализация данных
        visualize_data(data)

if __name__ == "__main__":
    main()
