import os
import logging
import yaml
import joblib

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    logging.info("Загрузка и предобработка данных...")

    # чтение параметров
    params_path = "params.yaml"
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
    else:
        params = {}

    dataset_path = params.get('data_processing', {}).get('dataset_path', 'dataset.csv')
    test_size = params.get('data_processing', {}).get('test_size', 0.2)
    random_state = params.get('data_processing', {}).get('random_state', 42)
    drop_columns = params.get('data_processing', {}).get('drop_columns', ['user_id', 'task_id', 'time_of_day'])

    logging.info(f"Параметры: dataset_path={dataset_path}, test_size={test_size}, random_state={random_state}")

    # чтение датасета
    if not os.path.exists(dataset_path):
        logging.error(f"Файл {dataset_path} не найден.")
        raise FileNotFoundError(f"{dataset_path} not found")

    df = pd.read_csv(dataset_path)

    # базовая очистка
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    # приведение даты
    if 'deadline' in df.columns:
        df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
    else:
        logging.error("В датафрейме отсутствует колонка 'deadline'.")
        raise KeyError("Missing 'deadline' column")

    # извлечение компонент даты/времени
    df['year'] = df['deadline'].dt.year
    df['month'] = df['deadline'].dt.month
    df['day'] = df['deadline'].dt.day
    df['hour'] = df['deadline'].dt.hour
    df['minute'] = df['deadline'].dt.minute

    # периодические признаки
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # выбрасываем оригинальную дату
    df = df.drop(columns=['deadline'])

    # кодирование категорий (OrdinalEncoder)
    encoder = OrdinalEncoder()
    if 'category' in df.columns:
        # OrdinalEncoder требует 2D
        df['category'] = encoder.fit_transform(df[['category']])
    else:
        logging.warning("Колонка 'category' не найдена — кодирование пропущено.")

    # формирование X и y
    if 'actual_duration' not in df.columns:
        logging.error("В датафрейме отсутствует колонка 'actual_duration'.")
        raise KeyError("Missing 'actual_duration' column")

    if 'fatigue_level' not in df.columns:
        logging.error("В датафрейме отсутствует колонка 'fatigue_level'.")
        raise KeyError("Missing 'fatigue_level' column")

    X = df.drop(columns=['actual_duration', 'fatigue_level'])
    y_duration = df['actual_duration']
    y_fatigue = df['fatigue_level']

    # разделение на train/valid
    X_train, X_valid, y_d_train, y_d_valid, y_f_train, y_f_valid = train_test_split(
        X, y_duration, y_fatigue, test_size=test_size, random_state=random_state
    )

    # сохраняем результаты
    os.makedirs('datatt/processed', exist_ok=True)
    X_train.to_csv('datatt/processed/X_train.csv', index=False)
    X_valid.to_csv('datatt/processed/X_valid.csv', index=False)
    y_d_train.to_csv('datatt/processed/y_duration_train.csv', index=False)
    y_d_valid.to_csv('datatt/processed/y_duration_valid.csv', index=False)
    y_f_train.to_csv('datatt/processed/y_fatigue_train.csv', index=False)
    y_f_valid.to_csv('datatt/processed/y_fatigue_valid.csv', index=False)

    os.makedirs('ddm', exist_ok=True)
    joblib.dump(encoder, "ddm/ordinal_encoder.joblib")

    logging.info("Данные успешно обработаны и сохранены в data/processed; encoder сохранён в ddm/ordinal_encoder.joblib")

    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'y_duration_train': y_d_train,
        'y_duration_valid': y_d_valid,
        'y_fatigue_train': y_f_train,
        'y_fatigue_valid': y_f_valid,
        'encoder': encoder
    }


if __name__ == "__main__":
    load_and_preprocess_data()
