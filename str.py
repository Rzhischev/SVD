import numpy as np
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt


def compute_svd(image, k):
    """Выполняет сингулярное разложение (SVD) и возвращает сжатое изображение"""
    img = np.array(image)
    img = img / 255  # Нормализуем значения изображения
    U, sing_vals, V = np.linalg.svd(img)  # Выполняем сингулярное разложение

    # Создаём двумерный массив sigma
    sigma = np.zeros((U.shape[0], V.shape[0]))
    np.fill_diagonal(sigma, sing_vals)  # Заполняем диагональные элементы матрицы sigma

    # Оставляем только топ-K сингулярных чисел
    trunc_U = U[:, :k]
    trunc_sigma = sigma[:k, :k]
    trunc_V = V[:k, :]

    trunc_img = trunc_U @ trunc_sigma @ trunc_V  # Получаем сжатое изображение

    # Нормализуем данные изображения
    trunc_img = trunc_img - np.min(trunc_img)
    trunc_img = trunc_img / np.max(trunc_img)

    return trunc_img


def main():
    st.title("Сжатие изображения с использованием SVD")

    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('L'))  # Переводим изображение в оттенки серого

        st.image(image, caption='Оригинальное изображение.', use_column_width=True)

        k = st.slider('Количество сингулярных чисел', min_value=1, max_value=min(img_array.shape), value=5)

        trunc_img = compute_svd(img_array, k)
        st.image(trunc_img, caption=f'Сжатое изображение с {k} сингулярными числами.', use_column_width=True)

if __name__ == "__main__":
    main()
