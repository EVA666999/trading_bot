import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping

import time

class ProgressBar(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()  # Время начала эпохи

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time  # Время, прошедшее за эпоху
        print(f"Эпоха {epoch + 1}: Потери: {logs['loss']:.6f}, Время: {elapsed_time:.2f} сек")

    def on_batch_end(self, batch, logs=None):
        total_batches = self.params['steps']  # Общее количество батчей
        batch_num = batch + 1  # Номер текущего батча (начинаем с 1)
        if batch_num % 100 == 0:  # Печать прогресса каждые 100 шагов
            percent = (batch_num / total_batches) * 100
            print(f"\rПроцесс: {percent:.2f}% - BATCH {batch_num}/{total_batches}", end='')


def preprocess_data(df, feature_col='Close', time_steps=50):
    """
    Подготовка данных для модели, включая сглаживание.
    """
    if feature_col not in df.columns:
        raise KeyError(f"Колонка '{feature_col}' не найдена. Доступные колонки: {df.columns.tolist()}")

    # Добавляем сглаживание
    df['Smoothed_Close'] = df[feature_col].ewm(span=10).mean()

    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Smoothed_Close']].values)  # Преобразуем в numpy массив

    # Создание временных окон
    X, y = [], []
    for i in range(time_steps, len(df_scaled)):
        X.append(df_scaled[i - time_steps:i, 0])
        y.append(df_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Преобразуем в форму для LSTM
    return X, y, scaler


def build_model(input_shape):
    """
    Оптимизированная LSTM модель
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape),  # Сокращаем до 50
        tf.keras.layers.LSTM(units=30, return_sequences=False),  # Сокращаем до 30
        tf.keras.layers.Dense(units=64, activation='relu'),  # Добавляем Dense слой
        tf.keras.layers.Dropout(0.1),  # Уменьшаем Dropout
        tf.keras.layers.Dense(units=1)  # Последний слой
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')  # MSE вместо Huber
    return model


def predict_for_all(df, model, scaler, feature_col, time_steps):
    """
    Оптимизированное предсказание для всех данных.
    """
    print("Начинаем предсказания...")
    # Извлекаем данные для предсказания
    df_scaled = scaler.transform(df[[feature_col]].values)
    X_pred = [
        df_scaled[i - time_steps:i, 0]
        for i in range(time_steps, len(df_scaled))
    ]
    X_pred = np.array(X_pred).reshape(-1, time_steps, 1)

    print("Массив данных для предсказаний подготовлен. Запускаем модель...")
    # Одновременное предсказание для всех данных
    predictions_scaled = model.predict(X_pred, verbose=1)
    predictions = scaler.inverse_transform(predictions_scaled)

    print("Предсказания завершены.")
    return predictions


def save_with_progress(df, output_file, chunk_size=1000):
    """
    Сохраняем DataFrame в файл по частям и выводим прогресс.
    """
    total_rows = len(df)
    num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size > 0 else 0)

    print(f"Всего строк в DataFrame: {total_rows}. Будет записано в {num_chunks} частей.")

    # Записываем заголовки только один раз
    with open(output_file, 'w', newline='') as f:
        # Пишем заголовки (колонки) только один раз
        print("Начинаем запись заголовков...")
        df.iloc[:0].to_csv(f, index=False)  # Записываем только заголовки
        print("Заголовки записаны. Начинаем запись данных...")

        # Записываем данные по частям
        for chunk_index in range(num_chunks):
            start_row = chunk_index * chunk_size
            end_row = min((chunk_index + 1) * chunk_size, total_rows)
            chunk = df.iloc[start_row:end_row]

            chunk.to_csv(f, header=False, index=False)  # Записываем части данных (без заголовков)

            # Вычисляем процент выполнения
            progress = ((chunk_index + 1) / num_chunks) * 100
            print(f"\rЗапись в файл: {progress:.2f}% завершено (Часть {chunk_index + 1}/{num_chunks})", end='')

    print("\nРезультаты сохранены в файл:", output_file)


def train_and_predict(df, feature_col='Close', time_steps=50, epochs=60, batch_size=32, output_file='predictions.csv'):
    """
    Основная функция для обучения, предсказания и сохранения результатов.
    """
    print("Подготовка данных для обучения...")
    X, y, scaler = preprocess_data(df, feature_col, time_steps)
    print(f"Данные подготовлены. Размеры X: {X.shape}, y: {y.shape}")

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f"Обучающие данные: {X_train.shape}, Тестовые данные: {X_test.shape}")

    # Построение модели
    print("Создаем модель...")
    model = build_model((X_train.shape[1], 1))
    print("Модель создана. Начинаем обучение...")

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=13,
        restore_best_weights=True
    )

    progress_bar = ProgressBar()
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[progress_bar, early_stopping],
        verbose=0
    )
    print("Обучение завершено. Начинаем предсказания...")

    # Предсказания
    predictions = np.full((len(df),), np.nan)
    predictions[time_steps:] = predict_for_all(df, model, scaler, 'Smoothed_Close', time_steps).flatten()

    print("Предсказания завершены. Добавляем их в DataFrame...")

    # Итоговый DataFrame с оригинальными, сглаженными и предсказанными данными
    df['Real Price'] = df[feature_col]  # Оригинальные данные
    df['Smoothed Price'] = df['Smoothed_Close']  # Сглаженные данные
    df['Predicted Price'] = predictions  # Предсказания модели

    print("Вычисляем метрики...")
    valid_indices = ~df['Predicted Price'].isna()  # Учитываем только строки с предсказаниями
    real_prices = df.loc[valid_indices, 'Real Price']
    predicted_prices = df.loc[valid_indices, 'Predicted Price']
    
    # Метрики
    avg_diff = np.mean(np.abs(real_prices - predicted_prices))
    rms_diff = np.sqrt(np.mean((real_prices - predicted_prices) ** 2))
    
    print(f"Средняя разница (Avg Difference): {avg_diff:.6f}")
    print(f"Среднеквадратичная разница (RMS Difference): {rms_diff:.6f}")

    print("Начинаем запись результатов в файл...")
    save_with_progress(df, output_file, chunk_size=1000)
    print("Запись завершена.")

    return model, df


# Пример использования
if __name__ == "__main__":
    # Загрузите свои данные
    df = pd.read_csv('btc_15m_data_2018_to_2024-2024-10-10.csv')  # Замените на ваш файл
    print(f"Количество строк в DataFrame: {len(df)}")
    model, result_df = train_and_predict(df, feature_col='Close', output_file='predictions.csv')