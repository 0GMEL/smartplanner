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

    params_path = "params.yaml"
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
    else:
        params = {}

    dataset_path = params.get('data_processing', {}).get('dataset_path', 'datas.csv')
    test_size = params.get('data_processing', {}).get('test_size', 0.2)
    random_state = params.get('data_processing', {}).get('random_state', 42)
    drop_columns = params.get('data_processing', {}).get('drop_columns', ['user_id', 'task_id', 'time_of_day'])

    logging.info(f"Параметры: dataset_path={dataset_path}, test_size={test_size}, random_state={random_state}")

    if not os.path.exists(dataset_path):
        logging.error(f"Файл {dataset_path} не найден.")
        raise FileNotFoundError(f"{dataset_path} not found")

    df = pd.read_csv(dataset_path)

    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    if 'deadline' in df.columns:
        df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
    else:
        logging.error("В датафрейме отсутствует колонка 'deadline'.")
        raise KeyError("Missing 'deadline' column")

    df['year'] = df['deadline'].dt.year
    df['month'] = df['deadline'].dt.month
    df['day'] = df['deadline'].dt.day
    df['hour'] = df['deadline'].dt.hour
    df['minute'] = df['deadline'].dt.minute

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df = df.drop(columns=['deadline'])

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if 'category' in df.columns:
        df['category'] = encoder.fit_transform(df[['category']])
    else:
        logging.warning("Колонка 'category' не найдена — кодирование пропущено.")

    X = df.drop(columns=['actual_duration', 'fatigue_level'])
    y_duration = df['actual_duration']
    y_fatigue = df['fatigue_level']

    X_train, X_valid, y_d_train, y_d_valid, y_fa_train, y_fa_valid = train_test_split(
        X, y_duration, y_fatigue, test_size=test_size, random_state=random_state
    )

    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_valid.to_csv('data/processed/X_valid.csv', index=False)
    y_d_train.to_csv('data/processed/y_duration_train.csv', index=False)
    y_d_valid.to_csv('data/processed/y_duration_valid.csv', index=False)
    y_fa_train.to_csv('data/processed/y_fatigue_train.csv', index=False)
    y_fa_valid.to_csv('data/processed/y_fatigue_valid.csv', index=False)

    os.makedirs('ddm', exist_ok=True)
    joblib.dump(encoder, "ddm/ordinal_encoder.joblib")

    logging.info("Данные успешно обработаны и сохранены; encoder сохранён в ddm/ordinal_encoder.joblib")

    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'y_duration_train': y_d_train,
        'y_duration_valid': y_d_valid,
        'y_fatigue_train': y_fa_train,
        'y_fatigue_valid': y_fa_valid,
        'encoder': encoder
    }

if __name__ == "__main__":
    load_and_preprocess_data()